# -*- coding: utf-8 -*-
"""
Kombiniertes Skript zum Trainieren und Evaluieren eines xTB-ähnlichen
Repulsionspotenzials. Angepasst an eine HDF5-Datei mit
'iteration_X/training' und 'iteration_X/test' Struktur.
FÜR MEHRERE CV-ITERATIONEN IN EINER SCHLEIFE.
"""
import os
import re
import pickle
from collections import Counter
from typing import List, Dict
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

from new_feeds_simon import xTBRepulsive, pairwise_repulsive


# =============================================================================
# --- 1. Globale Einstellungen und Konstanten ---
# =============================================================================
torch.set_default_dtype(torch.float64)
Tensor = torch.Tensor
DEVICE = torch.device('cpu')

# --- Dateipfade ---
DATASET_PATH = 'ani_kfold_dataset_with_formation_energies.h5'
SKF_FILE = 'mio.h5'

# --- Modellparameter ---
ORBITAL_BASIS = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
SPECIES = [1, 6, 7, 8]
ATOM_MAP = {'H': 1, 'C': 6, 'N': 7, 'O': 8}

# --- Trainings-Konfiguration ---
NUMBER_OF_EPOCHS = 100
LEARNING_RATE = 0.01


# =============================================================================
# --- 2. Hilfs- und Berechnungsfunktionen ---
# =============================================================================

def load_data_from_cv_iteration(file_path: str, iteration_num: int, dataset_type: str) -> tuple:
    """
    Lädt Daten aus einem spezifischen Trainings- oder Test-Split einer Iteration.
    dataset_type muss 'training' oder 'test' sein.
    """
    path_in_h5 = f'iteration_{iteration_num}/{dataset_type}'
    print(f"\n--- Lade Datensatz '{dataset_type}' aus Pfad: {path_in_h5} ---")

    all_molecule_names, all_species, all_coordinates, all_formation_energies = [], [], [], []

    try:
        with h5py.File(file_path, 'r') as f:
            if path_in_h5 not in f:
                print(f"Fehler: Pfad '{path_in_h5}' nicht in HDF5-Datei gefunden.")
                return [], [], [], []

            data_group = f[path_in_h5]
            for molecule_name, molecule_group in data_group.items():
                if molecule_name == 'O2':
                    print(f"Hinweis: Molekül '{molecule_name}' wird explizit übersprungen.")
                    continue
                
                if all(k in molecule_group for k in ['species', 'coordinates', 'formation_energies']):
                    all_molecule_names.append(molecule_name)
                    species_symbols = [s.decode('utf-8') for s in molecule_group['species'][()]]
                    all_species.append([ATOM_MAP[symbol] for symbol in species_symbols])
                    all_coordinates.append(molecule_group['coordinates'][()])
                    all_formation_energies.append(molecule_group['formation_energies'][()])
                else:
                    print(f"Warnung: Gruppe '{molecule_name}' in '{path_in_h5}' übersprungen (fehlende Datasets).")
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {file_path}")
        return [], [], [], []
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return [], [], [], []

    print(f"Erfolgreich {len(all_molecule_names)} Moleküle geladen.")
    return all_molecule_names, all_species, all_coordinates, all_formation_energies


def load_or_calculate_dftb_energies(molecule_names, atomic_numbers_list, coordinates_list, cache_file):
    """Lädt DFTB-Energien für das Trainings-Set aus dem Cache oder berechnet sie."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f: data = pickle.load(f)
        if data.get('status') == 'complete':
            print(f"\nLade finale DFTB-Ergebnisse für Trainings-Set aus '{cache_file}'.")
            return data['total_energies'], data['geometries']

    print("\nBeginne rechenintensive DFTB-Energie-Berechnung für das Trainings-Set.")
    all_total_energies, all_geometries, start_index = [], [], 0
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f: data = pickle.load(f)
        if data.get('status') != 'complete':
            print("Setze unterbrochene Berechnung fort...")
            all_total_energies, all_geometries = data['total_energies'], data['geometries']
            start_index = data.get('last_index', -1) + 1

    h_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'hamiltonian', device=DEVICE)
    s_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'overlap', device=DEVICE)
    o_feed = SkfOccupationFeed.from_database(SKF_FILE, SPECIES, device=DEVICE)
    u_feed = HubbardFeed.from_database(SKF_FILE, SPECIES, device=DEVICE)
    calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, suppress_scc_error=True)

    for i in range(start_index, len(molecule_names)):
        print(f"\n--- Verarbeite Trainingsmolekül {i+1}/{len(molecule_names)}: {molecule_names[i]} ---")
        for single_conformation_coords in coordinates_list[i]:
            geometry = Geometry(
                torch.tensor(atomic_numbers_list[i], device=DEVICE, dtype=torch.long),
                torch.tensor(single_conformation_coords, device=DEVICE, dtype=torch.float64),
                units='a')
            all_geometries.append(geometry)
            orbs = OrbitalInfo(geometry.atomic_numbers, ORBITAL_BASIS)
            calculator(geometry, orbs)
            all_total_energies.append(calculator.total_energy)

        with open(cache_file, 'wb') as f:
            pickle.dump({'status': 'in_progress', 'last_index': i,
                         'total_energies': all_total_energies, 'geometries': all_geometries}, f)
        print(f"Checkpoint nach Molekül {i+1} gespeichert.")

    with open(cache_file, 'wb') as f:
        pickle.dump({'status': 'complete', 'total_energies': all_total_energies, 'geometries': all_geometries}, f)
    print("\nAlle DFTB-Berechnungen für Trainings-Set erfolgreich abgeschlossen.")
    return all_total_energies, all_geometries


def get_chemical_potentials(calculator, alpha, Z, ref_data, cutoff_dict):
    """Berechnet die chemischen Potentiale basierend auf den aktuellen Modellparametern."""
    potentials = {}
    for symbol, data in ref_data.items():
        pair_repulsive = pairwise_repulsive(data['geom'], alpha, Z, xTBRepulsive, cutoff_dict)
        repulsive_energy = PairwiseRepulsiveEnergyFeed(pair_repulsive)(data['geom'])
        total_energy_ref = data['elec_energy'] + repulsive_energy
        potentials[symbol] = total_energy_ref / data['n_atoms']
    return potentials


def calculate_formation_energies(geometries, electronic_energies, chemical_potentials, alpha, Z, cutoff_dict):
    """Berechnet die Bildungsenergien für eine gegebene Liste von Molekülen."""
    predicted_energies = []
    for geom, e_elec in zip(geometries, electronic_energies):
        pair_repulsive = pairwise_repulsive(geom, alpha, Z, xTBRepulsive, cutoff_dict)
        e_repulsive = PairwiseRepulsiveEnergyFeed(pair_repulsive)(geom)
        e_total = e_elec + e_repulsive

        atom_counts = Counter(geom.atomic_numbers.tolist())
        e_atomic_ref = sum(count * chemical_potentials[ATOM_MAP_REV[element]] for element, count in atom_counts.items())
        
        predicted_energies.append(e_total - e_atomic_ref)
    return torch.stack(predicted_energies)


def plot_loss_vs_epochs(loss_history: List[float], filename: str):
    """Erstellt und speichert einen Plot des Trainings-Loss über die Epochen."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='MSE Loss', marker='o')
    plt.xlabel('Epoche'); plt.ylabel('Loss (log-Skala)')
    plt.title('Trainings-Loss über die Epochen'); plt.grid(True, which="both", ls="--")
    plt.legend(); plt.yscale('log'); plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"\nLoss-Verlauf gespeichert in '{filename}'")


def plot_test_results(target_flat, predicted_flat, labels_flat, fold_num: int, filename: str):
    """Erstellt einen Scatter-Plot der Testergebnisse (Vorhersage vs. Wahrheit)."""
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(list(set(labels_flat)))
    colors = plt.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        indices = [idx for idx, l in enumerate(labels_flat) if l == label]
        plt.scatter(target_flat[indices].numpy(), predicted_flat[indices].numpy(), 
                    color=colors(i), alpha=0.5, s=20, label=label)

    lims = [min(target_flat.min(), predicted_flat.min()).item(), max(target_flat.max(), predicted_flat.max()).item()]
    plt.plot(lims, lims, 'r--', alpha=0.8, zorder=0, label='Idealfall (y=x)')
    plt.xlim(lims); plt.ylim(lims)
    
    plt.xlabel('Tatsächliche Bildungsenergie (Hartree)')
    plt.ylabel('Vorhergesagte Bildungsenergie (Hartree)')
    plt.title(f'Modell-Vorhersagen vs. Zielwerte auf dem Test-Set (iteration_{fold_num})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(title="Moleküle", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()
    print(f"Test-Vorhersage-Plot gespeichert in '{filename}'")


# =============================================================================
# --- 3. Hauptskript (main) ---
# =============================================================================

def main(cv_iteration_index: int):
    """Hauptfunktion zur Ausführung des gesamten Train- und Test-Workflows für eine CV-Iteration."""
    
    # -------------------------------------------------------------------------
    # --- A. DATENVORBEREITUNG ---
    # -------------------------------------------------------------------------
    
    # --- Trainings- und Testdaten laden ---
    train_names, train_species, train_coords, train_energies_list = load_data_from_cv_iteration(
        DATASET_PATH, cv_iteration_index, 'training')
    if not train_names: return

    test_names, test_species, test_coords, test_energies_list = load_data_from_cv_iteration(
        DATASET_PATH, cv_iteration_index, 'test')
    if not test_names: return

    # --- DFTB-Baseline für Trainingsdaten berechnen/laden ---
    cache_file_for_iteration = f'dftb_cache_iter_{cv_iteration_index}.pkl'
    train_elec_energies_flat, train_geometries_flat = load_or_calculate_dftb_energies(
        train_names, train_species, train_coords, cache_file_for_iteration)

    # --- Referenzmoleküle für chemische Potentiale vorbereiten ---
    print("\n--- Bereite Referenzmoleküle für chemische Potentiale vor ---")
    if not os.path.exists('c60_coords.pt'):
        print("FEHLER: 'c60_coords.pt' nicht gefunden. Bitte erstelle diese Datei.")
        return
        
    ref_geometries = {
        'H': Geometry(torch.tensor([1, 1]), torch.tensor([[0.1288, 0., 0.], [0.8712, 0., 0.]], dtype=torch.float64), units='a'),
        'O': Geometry(torch.tensor([8, 8]), torch.tensor([
            [-0.10164345, 0.00000000, 0.00000000],
            [1.10164345, -0.00000000, -0.00000000]], dtype=torch.float64), units='a'),
        'N': Geometry(torch.tensor([7, 7]), torch.tensor([[-0.0510, 0., 0.], [1.0510, 0., 0.]], dtype=torch.float64), units='a'),
        'C': Geometry(torch.full((60,), 6), torch.load('c60_coords.pt').to(torch.float64), units='a')
    }
    ref_orbs = {
        'H': OrbitalInfo(ref_geometries['H'].atomic_numbers, {1: [0]}),
        'O': OrbitalInfo(ref_geometries['O'].atomic_numbers, {8: [0, 1]}),
        'N': OrbitalInfo(ref_geometries['N'].atomic_numbers, {7: [0, 1]}),
        'C': OrbitalInfo(ref_geometries['C'].atomic_numbers, {6: [0, 1]})
    }
    global ATOM_MAP_REV
    ATOM_MAP_REV = {v: k for k, v in ATOM_MAP.items()}

    h_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'hamiltonian')
    s_feed = SkFeed.from_database(SKF_FILE, SPECIES, 'overlap')
    o_feed = SkfOccupationFeed.from_database(SKF_FILE, SPECIES)
    u_feed = HubbardFeed.from_database(SKF_FILE, SPECIES)
    ref_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed)

    ref_data = {}
    for symbol, geom in ref_geometries.items():
        ref_calculator(geom, ref_orbs[symbol])
        ref_data[symbol] = {
            'geom': geom,
            'elec_energy': ref_calculator.total_energy.detach(),
            'n_atoms': len(geom.atomic_numbers)
        }
        
    all_species_pairs = combinations_with_replacement(SPECIES, 2)
    cutoff_dict = {str(pair): Tensor([10.0]) for pair in all_species_pairs}

    # -------------------------------------------------------------------------
    # --- B. MODELLTRAINING ---
    # -------------------------------------------------------------------------
    print("\n" + "="*50 + "\n--- B. MODELLTRAINING WIRD GESTARTET ---\n" + "="*50)

    alpha = {spec: Parameter(Tensor([1.0]), requires_grad=True) for spec in SPECIES}
    Z = {spec: Parameter(Tensor([float(spec)]), requires_grad=True) for spec in SPECIES}
    trainable_params = list(alpha.values()) + list(Z.values())
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    train_targets_flat = torch.tensor([item for sublist in train_energies_list for item in sublist], dtype=torch.float64)
    loss_history = []

    for epoch in range(NUMBER_OF_EPOCHS):
        optimizer.zero_grad()
        current_chemical_potentials = get_chemical_potentials(ref_calculator, alpha, Z, ref_data, cutoff_dict)
        predicted_energies = calculate_formation_energies(
            train_geometries_flat, train_elec_energies_flat,
            current_chemical_potentials, alpha, Z, cutoff_dict
        )
        loss = mse_loss(predicted_energies, train_targets_flat)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        print(f"Epoche {epoch+1}/{NUMBER_OF_EPOCHS} | Loss: {loss.item():.8f}")

    print("\n--- Training abgeschlossen ---")
    final_params_str = ", ".join([f"{v.item():.6f}" for v in trainable_params])
    print(f"Finale trainierte Parameter: [{final_params_str}]")
    plot_loss_vs_epochs(loss_history, filename=f'loss_history_iter_{cv_iteration_index}.png')

    # -------------------------------------------------------------------------
    # --- C. MODELLEVALUATION ---
    # -------------------------------------------------------------------------
    print("\n" + "="*50 + f"\n--- C. MODELLEVALUATION AUF TEST-SET (ITERATION {cv_iteration_index}) ---\n" + "="*50)
    
    with torch.no_grad():
        print("\nBerechne elektronische Energien für das Test-Set...")
        test_geometries_flat, test_elec_energies_flat, successful_indices_map = [], [], []
        
        dftb_calculator_test = Dftb2(h_feed, s_feed, o_feed, u_feed, suppress_scc_error=True)

        for i in range(len(test_names)):
            successful_indices_mol = []
            for j, coords in enumerate(test_coords[i]):
                geom = Geometry(torch.tensor(test_species[i]), torch.tensor(coords, dtype=torch.float64), units='a')
                orbs = OrbitalInfo(geom.atomic_numbers, ORBITAL_BASIS)
                try:
                    dftb_calculator_test(geom, orbs)
                    test_geometries_flat.append(geom)
                    test_elec_energies_flat.append(dftb_calculator_test.total_energy)
                    successful_indices_mol.append(j)
                except ConvergenceError:
                    print(f"    -> WARNUNG: Konvergenzfehler bei {test_names[i]}, Konformation {j+1}. Übersprungen.")
            successful_indices_map.append(successful_indices_mol)
        
        if not test_geometries_flat:
             print("\nKeine erfolgreichen Berechnungen im Test-Set. Evaluation wird abgebrochen.")
             return

        final_chemical_potentials = get_chemical_potentials(ref_calculator, alpha, Z, ref_data, cutoff_dict)
        test_predicted_energies = calculate_formation_energies(
            test_geometries_flat, test_elec_energies_flat,
            final_chemical_potentials, alpha, Z, cutoff_dict
        )

        test_target_energies_flat = torch.tensor([
            test_energies_list[i][j] for i, indices in enumerate(successful_indices_map) for j in indices
        ], dtype=torch.float64)
        test_labels_flat = [
            test_names[i] for i, indices in enumerate(successful_indices_map) for _ in indices
        ]
        
        mse = mse_loss(test_predicted_energies, test_target_energies_flat)
        rmse = torch.sqrt(mse)
        
        print("\n" + "="*25 + "\n      FINALE TESTERGEBNISSE\n" + "="*25)
        print(f"Anzahl erfolgreicher Test-Konformationen: {test_predicted_energies.numel()}")
        print(f"MSE auf Test-Set (iteration_{cv_iteration_index}): {mse.item():.8f}")
        print(f"RMSE auf Test-Set (iteration_{cv_iteration_index}): {rmse.item():.6f} Hartree")
        print(f"RMSE auf Test-Set (iteration_{cv_iteration_index}): {rmse.item() * 627.5:.2f} kcal/mol")
        print("="*25)
        
        plot_test_results(
            test_target_energies_flat,
            test_predicted_energies,
            test_labels_flat,
            cv_iteration_index,
            filename=f'test_predictions_vs_targets_iter_{cv_iteration_index}.png'
        )

# =============================================================================
# --- 4. Haupt-Ausführungsblock ---
# =============================================================================

if __name__ == "__main__":
    
    for i in range(5):
        print(f"\n{'='*25}\n   STARTING CV ITERATION {i}\n{'='*25}\n")
        main(cv_iteration_index=i)
        print(f"\n{'='*25}\n   FINISHED CV ITERATION {i}\n{'='*25}\n")