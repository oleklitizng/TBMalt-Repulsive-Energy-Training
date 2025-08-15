import h5py
import numpy as np
import os
import torch
import re
from itertools import combinations, combinations_with_replacement
from scipy.spatial.distance import pdist
from typing import Dict, Tuple, List, Optional
from collections import Counter

# =============================================================================
# TEIL 1: Importe und Klassen-Definitionen
# =============================================================================

from tbmalt.ml import Feed
from tbmalt.physics.dftb.feeds import PairwiseRepulsiveEnergyFeed
from new_feeds_simon import (
    xTBRepulsive, PTBPRepulsive, DFTBGammaRepulsive, pairwise_repulsive)
from tbmalt import Geometry

symbol_to_z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


# =============================================================================
# TEIL 2: Parser und neue Hilfsfunktionen
# =============================================================================

def parse_all_parameters(filepath: str) -> Dict:
    """
    Liest alle Parametersätze aus der 'extracted_parameters.txt' und
    speichert sie in einem verschachtelten Wörterbuch.
    """
    if not os.path.exists(filepath):
        print(f"Warnung: Parameterdatei '{filepath}' nicht gefunden. Analytische Modelle werden übersprungen.")
        return {}
        
    with open(filepath, 'r') as f: content = f.read()

    all_params = {}
    model_names = ['XTB', 'PTBP', 'GAMMA']

    for model_name in model_names:
        model_block_match = re.search(rf'^{model_name}([\s\S]*?)(?=\n^[A-Z]|\Z)', content, re.MULTILINE)
        if not model_block_match: continue

        model_block = model_block_match.group(1)
        param_sets_str = model_block.split('--')
        model_param_list = []
        for set_str in param_sets_str:
            if not set_str.strip(): continue
            alpha_match = re.search(r'Alpha:\s*({[^}]+})', set_str)
            z_match = re.search(r'Z:\s*({[^}]+})', set_str)
            
            if alpha_match and z_match:
                alpha_dict = eval(alpha_match.group(1).replace('\n', ''))
                z_dict = eval(z_match.group(1).replace('\n', ''))
                model_param_list.append({'Alpha': alpha_dict, 'Z': z_dict})
        
        all_params[model_name] = model_param_list
        
    return all_params

def formula_from_species(species_list: List[str]) -> str:
    """Erzeugt eine standardisierte Summenformel aus einer Atomliste (C, H, dann alphabetisch)."""
    counts = Counter(species_list)
    order = ['C', 'H']
    formula = ""
    for element in order:
        if element in counts:
            count = counts.pop(element)
            formula += f"{element}{count if count > 1 else ''}"
    
    for element in sorted(counts.keys()):
        count = counts[element]
        formula += f"{element}{count if count > 1 else ''}"
    return formula

def get_molecule_to_formula_map(h5_filepath: str) -> Dict[str, str]:
    """
    Durchsucht die HDF5-Datei und erstellt ein Mapping von jedem
    Molekülnamen zu seiner standardisierten Summenformel.
    """
    mapping = {}
    print("Erstelle Mapping von Molekül-Namen zu Summenformeln aus der HDF5-Datei...")
    try:
        with h5py.File(h5_filepath, 'r') as hf:
            for iteration_name in hf.keys():
                if not iteration_name.startswith('iteration'): continue
                for set_type in ['training', 'test']:
                    group_path = f"{iteration_name}/{set_type}"
                    if group_path in hf:
                        for molecule_name, molecule_group in hf[group_path].items():
                            if molecule_name not in mapping and 'species' in molecule_group:
                                species = [s.decode('utf-8') for s in molecule_group['species'][:]]
                                mapping[molecule_name] = formula_from_species(species)
    except Exception as e:
        print(f"Fehler beim Erstellen des Mappings: {e}")
    print(f"Mapping erstellt. {len(mapping)} einzigartige Molekülnamen gefunden.")
    return mapping


# =============================================================================
# TEIL 3: Hauptskript zur Analyse und Berechnung
# =============================================================================
def find_and_sort_molecule(h5_filepath: str, molecule_name: str, all_params: Dict, spline_feed: Optional[PairwiseRepulsiveEnergyFeed]):
    """
    Sucht nach einem spezifischen Molekülnamen und erstellt für jede
    gefundene Test-Gruppe eine separate Ausgabedatei.
    """
    conformations_by_group = {}

    print(f"Suche nach '{molecule_name}' in allen 'iteration_.../test' Gruppen...")

    try:
        with h5py.File(h5_filepath, 'r') as hf:
            for iteration_name in hf.keys():
                if not iteration_name.startswith('iteration'): continue
                
                try:
                    iteration_index = int(iteration_name.split('_')[-1])
                except (ValueError, IndexError):
                    continue
                
                group_path = f"{iteration_name}/test"
                if group_path in hf and molecule_name in hf[group_path]:
                    molecule_group = hf[group_path][molecule_name]
                    if 'coordinates' in molecule_group and 'formation_energies' in molecule_group:
                        coords_data = molecule_group['coordinates'][:]
                        energies_data = molecule_group['formation_energies'][:]
                        species_list = [s.decode('utf-8') for s in molecule_group['species'][:]]
                        
                        if group_path not in conformations_by_group:
                            conformations_by_group[group_path] = []

                        for i in range(len(energies_data)):
                            conformations_by_group[group_path].append(
                                (energies_data[i], coords_data[i], species_list, iteration_index)
                            )
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten beim Suchen von {molecule_name}: {e}")
        return

    if not conformations_by_group:
        print(f"Keine Daten für '{molecule_name}' in den Test-Gruppen gefunden.")
        return
    
    for group_path, conformations in conformations_by_group.items():
        safe_group_name = group_path.replace('/', '_')
        
        # *** HIER IST DIE GEÄNDERTE ZEILE FÜR DEN DATEINAMEN ***
        output_filename = f"{safe_group_name}_{molecule_name}_data.txt"
        
        print(f"Schreibe Daten für '{molecule_name}' aus '{group_path}' in '{output_filename}'...")
        
        conformations.sort(key=lambda x: x[0], reverse=True)

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            ANGSTROM_TO_BOHR = 1.8897259886
            for i, (energy, coords, species, iter_idx) in enumerate(conformations):
                outfile.write(f"--- Konformation Rang {i+1} (aus Gruppe {group_path}) ---\n")
                outfile.write(f"Bildungsenthalpie: {energy}\n")
                outfile.write("Koordinaten (in Ångström):\n")
                
                num_atoms = len(coords)
                for atom_index in range(num_atoms):
                    outfile.write(f"    {species[atom_index]:<4}  {coords[atom_index][0]:12.6f} {coords[atom_index][1]:12.6f} {coords[atom_index][2]:12.6f}\n")
                
                outfile.write("\nPaarweise Abstände und Repulsive Energies (in Ångström & Hartree):\n")
                
                if num_atoms > 1:
                    atomic_numbers = [symbol_to_z[s] for s in species]
                    geometry = Geometry(
                        torch.tensor(atomic_numbers, dtype=torch.long),
                        torch.tensor(coords, dtype=torch.float64),
                        units='angstrom'
                    )
                    unique_z = set(atomic_numbers)
                    cutoff_dict = {str(tuple(sorted(p))): torch.tensor(20.0) for p in combinations_with_replacement(unique_z, 2)}
                    xtb_feed_dict, ptbp_feed_dict, gamma_feed_dict = None, None, None
                    try:
                        if 'XTB' in all_params and iter_idx < len(all_params['XTB']):
                            params = all_params['XTB'][iter_idx]
                            xtb_feed_dict = pairwise_repulsive(geometry, params['Alpha'], params['Z'], xTBRepulsive, cutoff_dict)
                        if 'PTBP' in all_params and iter_idx < len(all_params['PTBP']):
                            params = all_params['PTBP'][iter_idx]
                            ptbp_feed_dict = pairwise_repulsive(geometry, params['Alpha'], params['Z'], PTBPRepulsive, cutoff_dict)
                        if 'GAMMA' in all_params and iter_idx < len(all_params['GAMMA']):
                            params = all_params['GAMMA'][iter_idx]
                            gamma_feed_dict = pairwise_repulsive(geometry, params['Alpha'], params['Z'], DFTBGammaRepulsive, cutoff_dict)
                    except (KeyError, IndexError):
                        outfile.write(f"        - Fehler: Parameter für Iteration {iter_idx} nicht gefunden.\n")

                    atom_index_pairs = list(combinations(range(num_atoms), 2))
                    distances_angstrom = pdist(coords)
                    total_xtb_e, total_ptbp_e, total_gamma_e, total_spline_e = 0.0, 0.0, 0.0, 0.0
                    for pair, dist_angstrom in zip(atom_index_pairs, distances_angstrom):
                        idx1, idx2 = pair
                        z1, z2 = atomic_numbers[idx1], atomic_numbers[idx2]
                        outfile.write(f"    Distanz {species[idx1]}({idx1+1}) - {species[idx2]}({idx2+1}): {dist_angstrom:.6f} Å\n")
                        dist_bohr = dist_angstrom * ANGSTROM_TO_BOHR
                        dist_tensor_bohr = torch.tensor([dist_bohr], dtype=torch.float64)
                        pair_key = str(tuple(sorted((z1, z2))))

                        if xtb_feed_dict and pair_key in xtb_feed_dict:
                            xtb_e = xtb_feed_dict[pair_key].forward(dist_tensor_bohr).item()
                            total_xtb_e += xtb_e
                            outfile.write(f"        - xTB Repulsive Energy  : {xtb_e:15.8f}\n")
                        if ptbp_feed_dict and pair_key in ptbp_feed_dict:
                            ptbp_e = ptbp_feed_dict[pair_key].forward(dist_tensor_bohr).item()
                            total_ptbp_e += ptbp_e
                            outfile.write(f"        - PTBP Repulsive Energy : {ptbp_e:15.8f}\n")
                        if gamma_feed_dict and pair_key in gamma_feed_dict:
                            gamma_e = gamma_feed_dict[pair_key].forward(dist_tensor_bohr).item()
                            total_gamma_e += gamma_e
                            outfile.write(f"        - GAMMA Repulsive Energy: {gamma_e:15.8f}\n")
                        if spline_feed and pair_key in spline_feed.repulsive_feeds:
                            spline_e = spline_feed.repulsive_feeds[pair_key].forward(dist_tensor_bohr).item()
                            total_spline_e += spline_e
                            outfile.write(f"        - Spline Repulsive Energy: {spline_e:15.8f}\n")
                    
                    outfile.write("\n    Total Repulsive Energies:\n")
                    if xtb_feed_dict: outfile.write(f"        - xTB Total Repulsive Energy   : {total_xtb_e:15.8f}\n")
                    if ptbp_feed_dict: outfile.write(f"        - PTBP Total Repulsive Energy  : {total_ptbp_e:15.8f}\n")
                    if gamma_feed_dict: outfile.write(f"        - GAMMA Total Repulsive Energy : {total_gamma_e:15.8f}\n")
                    if spline_feed: outfile.write(f"        - Spline Total Repulsive Energy: {total_spline_e:15.8f}\n")
                outfile.write("\n\n")
        print(f"Erfolgreich {len(conformations)} Konformationen für '{molecule_name}' geschrieben.")

if __name__ == '__main__':
    all_params = parse_all_parameters('extracted_parameters.txt')
    
    spline_feed = None
    if os.path.exists('mio.h5'):
        try:
            print("Lade Spline-Daten mit PairwiseRepulsiveEnergyFeed...")
            spline_feed = PairwiseRepulsiveEnergyFeed.from_database('mio.h5', species=[1, 6, 7, 8])
            print("Spline-Feed erfolgreich geladen.")
        except Exception as e:
            print(f"Warnung: Spline-Feed konnte nicht geladen werden. Fehler: {e}")

    h5_file = 'ani_kfold_dataset_with_formation_energies.h5'
    
    # =============================================================================
    # --- ANGEPASSTER ABSCHNITT: Automatische Analyse aller Moleküle ---
    # =============================================================================
    
    # *** HIER IST DIE KORRIGIERTE LISTE DER SUMMENFORMELN ***
    TARGET_FORMULAS = [
        'C2H6', 'CH5N', 'N2', 'H3NO', 'HNO', 'C2H4', 'CH2O', 'O2', 'H2O2', 
        'C2H2', 'H4N2', 'H2N2', 'C3H8', 'C2H7N', 'CH3NO', 'CH2O2', 
        'CH4N2', 'C3H4', 'C3H6', 'C2H5N', 'C2H4O', 'HNO2', 'H2O3', 
        'C2H6O', 'CO2', 'C2H3N'
    ]
    
    unique_formulas = sorted(list(set(TARGET_FORMULAS)))
    
    molecule_to_formula = get_molecule_to_formula_map(h5_file)
    
    formula_to_molecules = {}
    for name, formula in molecule_to_formula.items():
        if formula not in formula_to_molecules:
            formula_to_molecules[formula] = []
        formula_to_molecules[formula].append(name)

    print(f"\nStarte die automatische Analyse für {len(unique_formulas)} einzigartige Summenformeln...")
    
    for formula in unique_formulas:
        print(f"\n{'='*70}\nVerarbeite Summenformel: {formula}\n{'='*70}")
        
        molecules_to_process = formula_to_molecules.get(formula, [])
        
        if not molecules_to_process:
            print(f"Keine Moleküle für die Formel '{formula}' im Datensatz gefunden.")
            continue
        
        print(f"Gefundene Isomere für {formula}: {', '.join(molecules_to_process)}")
        
        for molecule_name in molecules_to_process:
            find_and_sort_molecule(h5_file, molecule_name, all_params, spline_feed)

    print(f"\n{'='*70}\nAnalyse für alle Moleküle abgeschlossen.\n{'='*70}")