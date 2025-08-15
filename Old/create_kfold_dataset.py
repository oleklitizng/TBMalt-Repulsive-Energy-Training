import pyanitools as pya
import h5py
import numpy as np
import os
import random
import re
from collections import Counter

# =============================================================================
# --- Konfiguration ---
# =============================================================================

# Quelldateien und Zieldatei anpassen
SOURCE_FILES = ['/ani_gdb_s02.h5', '/ani_gdb_s03.h5']
OUTPUT_FILE = 'ani_kfold_dataset_with_formation_energies.h5'
K_FOLDS = 5
MAX_CONFORMATIONS = 200

# Feste Liste der zu berücksichtigenden Molekülformeln
ALLOWED_MOLECULES = [
    'C2H6', 'CH5N', 'N2', 'NH3O', 'NHO', 'C2H4', 'CH2O', 'O2', 'H2O2', 'C2H2', 'N2H4', 'N2H2',
    'C3H8', 'C2H7N', 'CH3NO', 'CH2O2', 'CH4N2', 'CH3NO', 'C3H4', 'C3H6', 'C2H5N', 'C2H4O',
    'HNO2', 'H2O3', 'C2H6O', 'C2H7N', 'C2H6O', 'C3H6', 'C2H4O', 'CO2', 'C2H3N', 'CH4N2'
]

# Chemische Potenziale der reinen Elemente (Referenzenergien pro Atom in Hartree)
CHEMICAL_POTENTIALS = {
    "H": -1.139861178799 / 2,
    "O": -150.109707300186 / 2,
    "N": -109.498428859608 / 2,
    "C": -2285.64357853 / 60,
}


# =============================================================================
# --- Hilfsfunktionen ---
# =============================================================================

def parse_formula(formula: str) -> Counter:
    """
    Parst eine chemische Summenformel und gibt die Anzahl der Atome als Counter-Objekt zurück.
    """
    pattern = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    atom_counts = Counter()
    for atom, count in pattern:
        atom_counts[atom] += int(count) if count else 1
    return atom_counts


def find_molecule_by_symbols(symbols: list[str], molecule_list: list[str]) -> str | None:
    """
    Findet die passende Summenformel aus der `molecule_list` für eine Liste von Atom-Symbolen.
    """
    target_counts = Counter(symbols)
    for formula in molecule_list:
        candidate_counts = parse_formula(formula)
        if candidate_counts == target_counts:
            return formula
    return None


def calculate_reference_energy(formula: str, potentials: dict) -> float:
    """
    Berechnet die Summe der chemischen Potenziale (Sum_i n_i * µ_i) für eine gegebene chemische Formel.
    """
    atom_counts = parse_formula(formula)
    total_energy = 0.0
    for element, count in atom_counts.items():
        if element in potentials:
            total_energy += count * potentials[element]
        else:
            print(f"WARNUNG: Chemisches Potenzial für Element '{element}' nicht gefunden.")
    return total_energy


def write_molecule_to_group(h5_group: h5py.Group, molecule_data: dict, used_names_set: set):
    """
    Schreibt die Daten eines Moleküls in die HDF5-Gruppe und stellt
    sicher, dass der Gruppenname (die Formel) einzigartig ist.
    """
    base_name = molecule_data['formula']
    group_name = base_name
    counter = 1
    while group_name in used_names_set:
        group_name = f"{base_name}_{counter}"
        counter += 1
    used_names_set.add(group_name)

    molecule_group = h5_group.create_group(group_name)

    molecule_group.create_dataset('coordinates', data=molecule_data['coordinates'])
    molecule_group.create_dataset('formation_energies', data=molecule_data['formation_energies'])
    
    species_as_bytes = np.array([s.encode('utf-8') for s in molecule_data['species']])
    molecule_group.create_dataset('species', data=species_as_bytes)
    
    if 'smiles' in molecule_data and molecule_data['smiles']:
        smiles_utf8 = "".join(molecule_data['smiles']).encode('utf-8')
        molecule_group.create_dataset('smiles', data=smiles_utf8)


# =============================================================================
# --- Hauptskript ---
# =============================================================================

print("--- Start: Erstellung des k-Fold Datensatzes mit Bildungsenergien ---")

# --- Schritt 1: Daten laden, filtern und Bildungsenergie berechnen ---
print(f"\nSchritt 1: Lese Daten, filtere, berechne Bildungsenergien und wähle max. {MAX_CONFORMATIONS} Konformationen...")
all_data = {}
for hdf5file in SOURCE_FILES:
    if not os.path.exists(hdf5file):
        print(f"WARNUNG: Datei nicht gefunden: {hdf5file}. Überspringe.")
        continue
    
    print(f"   - Lade aus: {hdf5file}")
    adl = pya.anidataloader(hdf5file)
    for data in adl:
        path = data['path']
        if path not in all_data:
            if 'coordinates' in data and 'energies' in data and 'species' in data:
                species_list = data['species']
                formula = find_molecule_by_symbols(species_list, ALLOWED_MOLECULES)
                
                if formula is None:
                    continue

                raw_energies = data['energies']
                reference_energy = calculate_reference_energy(formula, CHEMICAL_POTENTIALS)
                formation_energies = raw_energies - reference_energy

                coords = data['coordinates']
                if len(coords) > MAX_CONFORMATIONS:
                    indices = np.random.choice(len(coords), MAX_CONFORMATIONS, replace=False)
                    selected_coords = coords[indices]
                    selected_energies = formation_energies[indices]
                else:
                    selected_coords = coords
                    selected_energies = formation_energies
                
                all_data[path] = {
                    'coordinates': selected_coords,
                    'formation_energies': selected_energies,
                    'species': species_list,
                    'smiles': data.get('smiles', ''),
                    'formula': formula
                }
    adl.cleanup()
print(f"Laden und Verarbeiten abgeschlossen. {len(all_data)} einzigartige Moleküle gefunden.\n")

# --- Schritt 2: Molekülpfade mischen ---
print("Schritt 2: Mische die Molekülpfade...")
molecule_paths = list(all_data.keys())
random.shuffle(molecule_paths)
print("Mischen abgeschlossen.\n")

# --- Schritt 3: Pfade in k Folds aufteilen ---
print(f"Schritt 3: Teile die Pfade in {K_FOLDS} Folds auf...")
folds = np.array_split(np.array(molecule_paths), K_FOLDS)
print("Aufteilung abgeschlossen.\n")

# --- Schritt 4: Neue HDF5-Datei mit Trainings-/Test-Splits schreiben ---
print(f"Schritt 4: Schreibe die Kreuzvalidierungs-Splits in die Datei: {OUTPUT_FILE}...")
with h5py.File(OUTPUT_FILE, 'w') as hdf5_out:
    hdf5_out.attrs['source_files'] = str(SOURCE_FILES)
    hdf5_out.attrs['k_folds'] = K_FOLDS
    
    for i in range(K_FOLDS):
        print(f" - Erstelle Iteration k = {i+1}...")
        iteration_group = hdf5_out.create_group(f'iteration_{i}')
        training_group = iteration_group.create_group('training')
        test_group = iteration_group.create_group('test')

        used_train_names, used_test_names = set(), set()

        # Test-Set (Fold i)
        for path in folds[i]:
            write_molecule_to_group(test_group, all_data[path], used_test_names)
            
        # Trainings-Set (alle anderen Folds)
        for j in range(K_FOLDS):
            if i == j: continue
            for path in folds[j]:
                write_molecule_to_group(training_group, all_data[path], used_train_names)

print("\n--- Prozess abgeschlossen! Die finale Datei wurde erfolgreich erstellt. ---")