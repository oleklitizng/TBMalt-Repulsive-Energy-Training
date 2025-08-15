# -*- coding: utf-8 -*-
"""
Adapted script for evaluating a DFTB2 model with a
fixed repulsive spline potential loaded from a file.

The script performs 5-fold cross-validation to assess the
prediction accuracy for formation energies on a test dataset.
No training of parameters is performed.
"""
import os
import logging
from collections import Counter
from typing import List, Dict

import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np

from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import (
    SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
)
from tbmalt.ml.loss_function import mse_loss
from tbmalt.common.exceptions import ConvergenceError

# =============================================================================
# --- 1. Global Settings and Constants ---
# =============================================================================
torch.set_default_dtype(torch.float64)
Tensor = torch.Tensor
DEVICE = torch.device('cpu')

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dftb2_evaluation_out.log", mode='w'),
        logging.StreamHandler()
    ]
)

# --- File Paths ---
DATASET_PATH = 'ani_kfold_dataset_with_formation_energies.h5'
SKF_FILE = 'mio.h5'
RESULTS_DIR = 'results_dftb2_spline'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Model Parameters ---
ORBITAL_BASIS = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
SPECIES = [1, 6, 7, 8]
ATOM_MAP = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
ATOM_MAP_REV = {v: k for k, v in ATOM_MAP.items()}


# =============================================================================
# --- 2. Helper and Calculation Functions ---
# =============================================================================

def log_convergence_error(molecule: str, conf_index: int, cv_iter: int, geometry: Geometry):
    """Logs a convergence error, including the geometry."""
    logging.warning(f"Convergence Error: CV_Iter={cv_iter}, Molecule={molecule}, Conformation_Index={conf_index}")
    coords_str = str(geometry.positions.tolist())
    logging.warning(f"  Geometry (Angstrom):\n  {coords_str}")


def load_data_from_cv_iteration(file_path: str, iteration_num: int, dataset_type: str) -> tuple:
    """Loads a dataset (training or test) for a specific CV iteration."""
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


def get_chemical_potentials(ref_data: Dict, calculator: Dftb2) -> Dict:
    """Calculates the chemical potentials for reference molecules using the given Dftb2 calculator."""
    potentials = {}
    logging.info("Calculating chemical potentials for reference species...")
    for symbol, data in ref_data.items():
        try:
            calculator(data['geom'], data['orbs'])
            total_energy_ref = calculator.total_energy
            potentials[symbol] = total_energy_ref / data['n_atoms']
            # Around line 103
            logging.info(f"  - Potential for {symbol}: {potentials[symbol].item():.6f} Hartree/atom")
        except ConvergenceError:
            logging.critical(f"FATAL ERROR: DFTB calculation for reference molecule '{symbol}' failed to converge.")
            logging.critical("The script cannot continue without all chemical potentials. Aborting.")
            raise  # Exits the program
    return potentials


def calculate_formation_energies(total_energies: List[Tensor], geometries: List[Geometry], chem_potentials: Dict) -> Tensor:
    """Calculates formation energies from total energies and chemical potentials."""
    predicted_energies = []
    for geom, e_total in zip(geometries, total_energies):
        atom_counts = Counter(geom.atomic_numbers.tolist())
        e_atomic_ref = sum(count * chem_potentials[ATOM_MAP_REV[element]] for element, count in atom_counts.items())
        predicted_energies.append(e_total - e_atomic_ref)
    return torch.stack(predicted_energies)


def plot_test_results(target: Tensor, predicted: Tensor, labels: List[str], filename: str):
    """Creates and saves a scatter plot of the test predictions versus the reference values."""
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
    plt.title('DFTB2 Predictions vs. Target Values on the Test Set')
    plt.grid(True, linestyle='--', alpha=0.6); plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(title="Molecules", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(filename); plt.close()
    logging.info(f"Test prediction plot saved to '{filename}'")


# =============================================================================
# --- 3. Main Script (main) ---
# =============================================================================

def main(cv_iteration_index: int):
    logging.info(f"--- A. DATA PREPARATION for Iteration {cv_iteration_index} ---")
    test_names, test_species, test_coords, test_energies_list = load_data_from_cv_iteration(
        DATASET_PATH, cv_iteration_index, 'test')
    if not test_names: return

    logging.info("--- B. INITIALIZING DFTB2 CALCULATOR ---")
    # Load all necessary feeds from the SKF file
    h_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'hamiltonian', device=DEVICE)
    s_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'overlap', device=DEVICE)
    o_feed = SkfOccupationFeed.from_database(SKF_FILE, SPECIES, device=DEVICE)
    u_feed = HubbardFeed.from_database(SKF_FILE, SPECIES, device=DEVICE)
    # **Here is the key change: Load the repulsive potential directly**
    r_feed = RepulsiveSplineFeed.from_database(SKF_FILE, SPECIES, device=DEVICE)
    
    # Initialize the Dftb2 calculator with all feeds, including the repulsive one
    calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed,
                       filling_scheme='fermi', filling_temp=0.001)
    logging.info("DFTB2 calculator with spline repulsion successfully created.")

    logging.info("--- C. CALCULATING CHEMICAL POTENTIALS ---")
    if not os.path.exists('c60_coords.pt'):
        logging.error("ERROR: 'c60_coords.pt' not found."); return

    ref_geometries = {
        'H': Geometry(torch.tensor([1, 1]), torch.tensor([[0.1288, 0., 0.], [0.8712, 0., 0.]], dtype=torch.float64), units='a'),
        'O': Geometry(torch.tensor([8, 8]), torch.tensor([
            [-0.10164345, 0.00000000, 0.00000000],
            [1.10164345, -0.00000000, -0.00000000]], dtype=torch.float64), units='a'),
        'N': Geometry(torch.tensor([7, 7]), torch.tensor([[-0.0510, 0., 0.], [1.0510, 0., 0.]], dtype=torch.float64), units='a'),
        'C': Geometry(torch.full((60,), 6), torch.load('c60_coords.pt').to(torch.float64), units='a')}
    
    ref_data = {symbol: {
        'geom': geom,
        'orbs': OrbitalInfo(geom.atomic_numbers, ORBITAL_BASIS),
        'n_atoms': len(geom.atomic_numbers)
    } for symbol, geom in ref_geometries.items()}
    
    try:
        final_chem_pots = get_chemical_potentials(ref_data, calculator)
    except ConvergenceError:
        return # Terminate if reference calculation fails

    logging.info(f"--- D. MODEL EVALUATION (Iteration {cv_iteration_index}) ---")
    with torch.no_grad():
        test_geometries, test_total_energies, successful_indices = [], [], []
        
        for i in range(len(test_names)):
            succ_indices_mol = []
            for j, coords in enumerate(test_coords[i]):
                geom = Geometry(torch.tensor(test_species[i]), torch.tensor(coords, dtype=torch.float64), units='a')
                orbs = OrbitalInfo(geom.atomic_numbers, ORBITAL_BASIS)
                try:
                    # Calculate the *total* energy (electronic + repulsive) in one step
                    calculator(geom, orbs)
                    test_geometries.append(geom)
                    test_total_energies.append(calculator.total_energy)
                    succ_indices_mol.append(j)
                except ConvergenceError:
                    log_convergence_error(test_names[i], j, cv_iteration_index, geom)
                    logging.warning(f"    -> WARNING: Convergence error for {test_names[i]}, conformation {j+1}. Skipping.")
            successful_indices.append(succ_indices_mol)
        
        if not test_geometries:
             logging.warning("No successful calculations in the test set. Aborting."); return

        # Calculate predicted formation energies
        test_predicted = calculate_formation_energies(test_total_energies, test_geometries, final_chem_pots)
        
        # Collect the reference formation energies for the successfully calculated conformations
        test_target = torch.tensor([test_energies_list[i][j] for i, idxs in enumerate(successful_indices) for j in idxs])
        test_labels = [test_names[i] for i, idxs in enumerate(successful_indices) for _ in idxs]
        
        # Calculate and print the RMSE error
        rmse = torch.sqrt(mse_loss(test_predicted, test_target))
        logging.info(f"RMSE on Test Set: {rmse.item():.6f} Hartree ({rmse.item() * 627.5:.2f} kcal/mol)")
        
        # Create the plot
        plot_test_results(test_target, test_predicted, test_labels,
                          filename=os.path.join(RESULTS_DIR, f'predictions_iter_{cv_iteration_index}.png'))

# =============================================================================
# --- 4. Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    # Since we are only evaluating one method, the loop over different methods is removed.
    # We still perform the 5-fold cross-validation.
    for i in range(5):
        logging.info(f"\n{'='*70}\n   STARTING: DFTB2 with Spline Repulsion, CV Iteration={i}\n{'='*70}")
        main(cv_iteration_index=i)
        logging.info(f"\n{'='*70}\n   FINISHED: DFTB2 with Spline Repulsion, CV Iteration={i}\n{'='*70}")