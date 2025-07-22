# -*- coding: utf-8 -*-
"""
Final adapted script for training and evaluating three different
repulsion potentials (xTB, PTBP, Gamma) using 5-fold cross-validation.
Includes robust error handling, method-specific start parameters, and geometry logging.
Output is now logged to 'out.log'.
"""
import os
import pickle
import logging
from collections import Counter
from typing import List, Dict, Type
from itertools import combinations_with_replacement

import h5py
import torch
import torch.nn as nn
from torch.nn import Parameter
import matplotlib.pyplot as plt
import numpy as np

from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import (
    SkFeed, SkfOccupationFeed, HubbardFeed, PairwiseRepulsiveEnergyFeed
)
from tbmalt.ml.loss_function import mse_loss
from tbmalt.common.exceptions import ConvergenceError

# Import the repulsion classes and the updated pairwise_repulsive function
from new_feeds_simon import (
    PTBPRepulsive, xTBRepulsive, DFTBGammaRepulsive, Feed, pairwise_repulsive
)

# =============================================================================
# --- 1. Global Settings and Constants ---
# =============================================================================
torch.set_default_dtype(torch.float64)
Tensor = torch.Tensor
DEVICE = torch.device('cpu')

# --- Logging Setup ---
# Configure logging to write to 'out.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("out.log", mode='w'), # 'w' for overwrite, 'a' for append
        logging.StreamHandler()  # Keep printing to console as well
    ]
)

# --- File Paths ---
DATASET_PATH = 'ani_kfold_dataset_with_formation_energies.h5'
SKF_FILE = 'mio.h5'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True) # Create folder for results

# --- Model Parameters ---
ORBITAL_BASIS = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
SPECIES = [1, 6, 7, 8]
ATOM_MAP = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
ATOM_MAP_REV = {v: k for k, v in ATOM_MAP.items()}

# --- Training Configuration ---
NUMBER_OF_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.01


# =============================================================================
# --- 2. Helper and Calculation Functions ---
# =============================================================================

def log_convergence_error(molecule: str, conf_index: int, method: str, cv_iter: int, geometry: Geometry):
    """Logs a convergence error, including geometry, to the central log file."""
    logging.warning(f"Convergence Error: Method={method}, CV_Iter={cv_iter}, Molecule={molecule}, Conformation_Index={conf_index}")
    coords_str = str(geometry.positions.tolist())
    logging.warning(f"  Geometry (Angstrom):\n  {coords_str}")


def load_data_from_cv_iteration(file_path: str, iteration_num: int, dataset_type: str) -> tuple:
    path_in_h5 = f'iteration_{iteration_num}/{dataset_type}'
    logging.info(f"--- Loading dataset '{dataset_type}' from path: {path_in_h5} ---")
    all_molecule_names, all_species, all_coordinates, all_formation_energies = [], [], [], []
    try:
        with h5py.File(file_path, 'r') as f:
            if path_in_h5 not in f:
                logging.error(f"Error: Path '{path_in_h5}' not found in HDF5 file.")
                return [], [], [], []
            data_group = f[path_in_h5]
            for molecule_name, molecule_group in data_group.items():
                if all(k in molecule_group for k in ['species', 'coordinates', 'formation_energies']):
                    all_molecule_names.append(molecule_name)
                    species_symbols = [s.decode('utf-8') for s in molecule_group['species'][()]]
                    all_species.append([ATOM_MAP[symbol] for symbol in species_symbols])
                    all_coordinates.append(molecule_group['coordinates'][()])
                    all_formation_energies.append(molecule_group['formation_energies'][()])
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return [], [], [], []
    logging.info(f"Successfully loaded {len(all_molecule_names)} molecules.")
    return all_molecule_names, all_species, all_coordinates, all_formation_energies


def load_or_calculate_dftb_energies(
    molecule_names, atomic_numbers_list, coords_list, target_energies_list, cache_file, cv_iter: int):
    """
    Loads or calculates DFTB energies. It now also filters the target energies
    to ensure consistency when conformations are skipped.
    Returns: electronic energies, geometries, and the corresponding filtered target energies.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f: data = pickle.load(f)
        if data.get('status') == 'complete':
            logging.info(f"Loading final DFTB results for training set from '{cache_file}'.")
            return data['total_energies'], data['geometries'], data['target_energies']

    logging.info("Starting computationally intensive DFTB energy calculation for the training set.")
    all_total_energies, all_geometries, all_target_energies, start_index = [], [], [], 0
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f: data = pickle.load(f)
        if data.get('status') != 'complete':
            all_total_energies = data.get('total_energies', [])
            all_geometries = data.get('geometries', [])
            all_target_energies = data.get('target_energies', [])
            start_index = data.get('last_index', -1) + 1

    h_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'hamiltonian', device=DEVICE)
    s_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'overlap', device=DEVICE)
    o_feed = SkfOccupationFeed.from_database(SKF_FILE, SPECIES, device=DEVICE)
    u_feed = HubbardFeed.from_database(SKF_FILE, SPECIES, device=DEVICE)
    calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, filling_scheme='fermi', filling_temp=0.001)

    for i in range(start_index, len(molecule_names)):
        logging.info(f"--- Processing training molecule {i+1}/{len(molecule_names)}: {molecule_names[i]} ---")
        for j, single_conformation_coords in enumerate(coords_list[i]):
            geometry = Geometry(
                torch.tensor(atomic_numbers_list[i], device=DEVICE, dtype=torch.long),
                torch.tensor(single_conformation_coords, device=DEVICE, dtype=torch.float64), units='a')
            orbs = OrbitalInfo(geometry.atomic_numbers, ORBITAL_BASIS)
            try:
                calculator(geometry, orbs)
                all_geometries.append(geometry)
                all_total_energies.append(calculator.total_energy)
                all_target_energies.append(target_energies_list[i][j])
            except ConvergenceError:
                log_convergence_error(molecule_names[i], j, 'dftb-baseline', cv_iter, geometry)
                logging.warning(f"    -> WARNING: Convergence error for {molecule_names[i]}, conformation {j+1}. Skipping.")

        with open(cache_file, 'wb') as f:
            pickle.dump({'status': 'in_progress', 'last_index': i,
                         'total_energies': all_total_energies, 'geometries': all_geometries,
                         'target_energies': all_target_energies}, f)
    with open(cache_file, 'wb') as f:
        pickle.dump({'status': 'complete', 'total_energies': all_total_energies, 'geometries': all_geometries,
                     'target_energies': all_target_energies}, f)
    logging.info("All DFTB calculations for the training set completed successfully.")
    return all_total_energies, all_geometries, all_target_energies


def get_chemical_potentials(ref_data, alpha, Z, repulsive_class, cutoff_dict):
    potentials = {}
    for symbol, data in ref_data.items():
        pair_rep_feed = pairwise_repulsive(data['geom'], alpha, Z, repulsive_class, cutoff_dict)
        repulsive_energy = PairwiseRepulsiveEnergyFeed(pair_rep_feed)(data['geom'])
        total_energy_ref = data['elec_energy'] + repulsive_energy
        potentials[symbol] = total_energy_ref / data['n_atoms']
    return potentials


def calculate_formation_energies(geometries, elec_energies, chem_potentials, alpha, Z, repulsive_class, cutoff_dict):
    predicted_energies = []
    for geom, e_elec in zip(geometries, elec_energies):
        pair_rep_feed = pairwise_repulsive(geom, alpha, Z, repulsive_class, cutoff_dict)
        e_repulsive = PairwiseRepulsiveEnergyFeed(pair_rep_feed)(geom)
        e_total = e_elec + e_repulsive
        atom_counts = Counter(geom.atomic_numbers.tolist())
        e_atomic_ref = sum(count * chem_potentials[ATOM_MAP_REV[element]] for element, count in atom_counts.items())
        predicted_energies.append(e_total - e_atomic_ref)
    return torch.stack(predicted_energies)


def plot_loss_vs_epochs(loss_history: List[float], filename: str):
    plt.figure(figsize=(10, 6)); plt.plot(loss_history, label='MSE Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)'); plt.title('Training Loss over Epochs')
    plt.grid(True, which="both", ls="--"); plt.legend(); plt.yscale('log'); plt.tight_layout()
    plt.savefig(filename); plt.close()
    logging.info(f"Loss history saved to '{filename}'")


def plot_test_results(target: Tensor, predicted: Tensor, labels: List[str], filename: str):
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(list(set(labels)))
    colors = plt.get_cmap('tab10', len(unique_labels))
    for i, label in enumerate(unique_labels):
        indices = [idx for idx, l in enumerate(labels) if l == label]
        plt.scatter(target[indices].numpy(), predicted[indices].numpy(),
                    color=colors(i), alpha=0.5, s=20, label=label)
    x_min, x_max = plt.xlim(); y_min, y_max = plt.ylim()
    overall_min = min(x_min, y_min); overall_max = max(x_max, y_max)
    range_buffer = (overall_max - overall_min) * 0.05
    final_lims = [overall_min - range_buffer, overall_max + range_buffer]
    plt.plot(final_lims, final_lims, 'r--', alpha=0.8, zorder=0, label='Ideal (y=x)')
    plt.xlim(final_lims); plt.ylim(final_lims)
    plt.xlabel('Reference Formation Energy (Hartree)'); plt.ylabel('Predicted Formation Energy (Hartree)')
    plt.title('Model Predictions vs. Target Values on the Test Set')
    plt.grid(True, linestyle='--', alpha=0.6); plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(title="Molecules", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(filename); plt.close()
    logging.info(f"Test prediction plot saved to '{filename}'")


# =============================================================================
# --- 3. Main Script (main) ---
# =============================================================================

def main(cv_iteration_index: int, repulsive_class: Type[Feed], method_name: str):
    logging.info(f"--- A. DATA PREPARATION for {method_name}, Iteration {cv_iteration_index} ---")
    train_names, train_species, train_coords, train_energies_list = load_data_from_cv_iteration(
        DATASET_PATH, cv_iteration_index, 'training')
    if not train_names: return
    test_names, test_species, test_coords, test_energies_list = load_data_from_cv_iteration(
        DATASET_PATH, cv_iteration_index, 'test')
    if not test_names: return

    cache_file = os.path.join(RESULTS_DIR, f'dftb_cache_iter_{cv_iteration_index}.pkl')
    train_elec_energies_flat, train_geometries_flat, train_targets_filtered = load_or_calculate_dftb_energies(
        train_names, train_species, train_coords, train_energies_list, cache_file, cv_iteration_index)

    logging.info("--- Preparing reference molecules for chemical potentials ---")
    if not os.path.exists('c60_coords.pt'):
        logging.error("ERROR: 'c60_coords.pt' not found."); return
        
    ref_geometries = {
        'H': Geometry(torch.tensor([1, 1]), torch.tensor([[0.1288, 0., 0.], [0.8712, 0., 0.]], dtype=torch.float64), units='a'),
        'O': Geometry(torch.tensor([8, 8]), torch.tensor([
            [-0.10164345, 0.00000000, 0.00000000],
            [1.10164345, -0.00000000, -0.00000000]], dtype=torch.float64), units='a'),
        'N': Geometry(torch.tensor([7, 7]), torch.tensor([[-0.0510, 0., 0.], [1.0510, 0., 0.]], dtype=torch.float64), units='a'),
        'C': Geometry(torch.full((60,), 6), torch.load('c60_coords.pt').to(torch.float64), units='a')}
    ref_orbs = {
        'H': OrbitalInfo(ref_geometries['H'].atomic_numbers, ORBITAL_BASIS),
        'O': OrbitalInfo(ref_geometries['O'].atomic_numbers, ORBITAL_BASIS),
        'N': OrbitalInfo(ref_geometries['N'].atomic_numbers, ORBITAL_BASIS),
        'C': OrbitalInfo(ref_geometries['C'].atomic_numbers, ORBITAL_BASIS)
    }

    h_feed=SkFeed.from_database(SKF_FILE,SPECIES,'hamiltonian');s_feed=SkFeed.from_database(SKF_FILE,SPECIES,'overlap')
    o_feed=SkfOccupationFeed.from_database(SKF_FILE,SPECIES);u_feed=HubbardFeed.from_database(SKF_FILE,SPECIES)
    ref_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, filling_scheme='fermi', filling_temp=0.001)
    
    ref_data = {}
    for symbol, geom, orbs in zip(ref_geometries.keys(), ref_geometries.values(), ref_orbs.values()):
        try:
            ref_calculator(geom, orbs)
            ref_data[symbol] = {
                'geom': geom,
                'elec_energy': ref_calculator.total_energy.detach(),
                'n_atoms': len(geom.atomic_numbers)
            }
        except ConvergenceError:
            logging.critical(f"FATAL ERROR: DFTB calculation for reference molecule '{symbol}' failed to converge.")
            logging.critical("The script cannot continue without all chemical potentials. Aborting this run.")
            return

    all_pairs = combinations_with_replacement(SPECIES, 2)
    cutoff_dict = {str(tuple(sorted(pair))): Tensor([5.0]) for pair in all_pairs}

    logging.info(f"--- B. MODEL TRAINING ({method_name}, Iteration {cv_iteration_index}) ---")
    
    if method_name == 'Gamma':
        logging.info("-> Using specialized starting parameters and learning rate for Gamma method.")
        alpha = {spec: Parameter(Tensor([0.26]), requires_grad=True) for spec in SPECIES}
        Z = {spec: Parameter(Tensor([0.5]), requires_grad=True) for spec in SPECIES}
        current_lr = 0.005
    else:
        logging.info("-> Using default starting parameters.")
        alpha = {spec: Parameter(Tensor([1.0]), requires_grad=True) for spec in SPECIES}
        Z = {spec: Parameter(Tensor([float(spec)]), requires_grad=True) for spec in SPECIES}
        current_lr = DEFAULT_LEARNING_RATE

    trainable_params = list(alpha.values()) + list(Z.values())
    optimizer = torch.optim.Adam(trainable_params, lr=current_lr)
    
    train_targets_flat = torch.tensor(train_targets_filtered, dtype=torch.float64)
    loss_history = []

    for epoch in range(NUMBER_OF_EPOCHS):
        optimizer.zero_grad()
        chem_pots = get_chemical_potentials(ref_data, alpha, Z, repulsive_class, cutoff_dict)
        pred_energies = calculate_formation_energies(
            train_geometries_flat, train_elec_energies_flat,
            chem_pots, alpha, Z, repulsive_class, cutoff_dict)
        loss = mse_loss(pred_energies, train_targets_flat)
        loss.backward(); optimizer.step(); loss_history.append(loss.item())
        
        logging.info(f"Epoch {epoch+1}/{NUMBER_OF_EPOCHS} | Loss: {loss.item():.8f}")

    plot_loss_vs_epochs(loss_history, filename=os.path.join(
        RESULTS_DIR, f'loss_history_{method_name}_iter_{cv_iteration_index}.png'))

    logging.info(f"--- C. MODEL EVALUATION ({method_name}, Iteration {cv_iteration_index}) ---")
    with torch.no_grad():
        test_geometries, test_elec_energies, successful_indices = [], [], []
        dftb_test = calculator = Dftb2(
            h_feed, s_feed, o_feed, u_feed,
            filling_scheme='fermi', filling_temp=0.001
        )
        for i in range(len(test_names)):
            succ_indices_mol = []
            for j, coords in enumerate(test_coords[i]):
                geom = Geometry(torch.tensor(test_species[i]), torch.tensor(coords,dtype=torch.float64), units='a')
                orbs = OrbitalInfo(geom.atomic_numbers, ORBITAL_BASIS)
                try:
                    dftb_test(geom, orbs)
                    test_geometries.append(geom); test_elec_energies.append(dftb_test.total_energy)
                    succ_indices_mol.append(j)
                except ConvergenceError:
                    log_convergence_error(test_names[i], j, method_name, cv_iteration_index, geom)
                    logging.warning(f"    -> WARNING: Convergence error for {test_names[i]}, conformation {j+1}. Skipping.")
            successful_indices.append(succ_indices_mol)
        
        if not test_geometries:
             logging.warning("No successful calculations in the test set. Aborting."); return

        final_chem_pots = get_chemical_potentials(ref_data, alpha, Z, repulsive_class, cutoff_dict)
        test_predicted = calculate_formation_energies(
            test_geometries, test_elec_energies, final_chem_pots, alpha, Z, repulsive_class, cutoff_dict)
        test_target = torch.tensor([test_energies_list[i][j] for i, idxs in enumerate(successful_indices) for j in idxs])
        test_labels = [test_names[i] for i, idxs in enumerate(successful_indices) for _ in idxs]
        
        rmse = torch.sqrt(mse_loss(test_predicted, test_target))
        logging.info(f"RMSE on Test Set: {rmse.item():.6f} Hartree ({rmse.item() * 627.5:.2f} kcal/mol)")
        
        plot_test_results(test_target, test_predicted, test_labels,
                          filename=os.path.join(RESULTS_DIR, f'predictions_{method_name}_iter_{cv_iteration_index}.png'))

# =============================================================================
# --- 4. Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    
    repulsive_methods = {
        'xTB': xTBRepulsive,
        'PTBP': PTBPRepulsive,
        'Gamma': DFTBGammaRepulsive
    }
    
    for name, method_class in repulsive_methods.items():
        for i in range(5):
            logging.info(f"\n{'='*70}\n   STARTING: Method={name}, CV Iteration={i}\n{'='*70}")
            main(cv_iteration_index=i, repulsive_class=method_class, method_name=name)
            logging.info(f"\n{'='*70}\n   FINISHED: Method={name}, CV Iteration={i}\n{'='*70}")