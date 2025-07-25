import h5py
import numpy as np
import os
from itertools import combinations
from scipy.spatial.distance import pdist

def find_and_sort_molecule(h5_filepath: str, molecule_name: str):
    """
    Sucht nach einem Molekül, extrahiert für jede Konformation die Energie, 
    Koordinaten und Atomsymbole. Berechnet zusätzlich die paarweisen Abstände.
    Sortiert das Ergebnis nach Energie und schreibt es in eine Textdatei.
    """
    if not os.path.exists(h5_filepath):
        print(f"Fehler: Die Datei '{h5_filepath}' wurde nicht gefunden.")
        return

    output_filename = f"{molecule_name}_data_with_distances.txt"
    # Die Liste sammelt jetzt auch die Atomsymbole (species)
    all_found_conformations = [] 

    print(f"Suche nach dem Molekül '{molecule_name}' in allen 'iteration_.../test' Gruppen...")

    try:
        with h5py.File(h5_filepath, 'r') as hf:
            for iteration_name in hf.keys():
                if not iteration_name.startswith('iteration'):
                    continue
                iteration_group = hf[iteration_name]
                set_type = 'test'
                if set_type in iteration_group:
                    set_group = iteration_group[set_type]
                    if molecule_name in set_group:
                        print(f"  --> Treffer gefunden in: {iteration_name}/{set_type}/{molecule_name}")
                        molecule_group = set_group[molecule_name]

                        if 'coordinates' in molecule_group and 'formation_energies' in molecule_group:
                            coords_data = molecule_group['coordinates'][:]
                            energies_data = molecule_group['formation_energies'][:]
                            
                            # NEU: Lese auch die Atomsymbole (species)
                            species_list = None
                            if 'species' in molecule_group:
                                # Dekodieren der Byte-Strings zu normalen Strings
                                species_list = [s.decode('utf-8') for s in molecule_group['species'][:]]
                            else:
                                print(f"    --> Warnung: 'species' Dataset nicht gefunden für {molecule_name}. Verwende generische Atomnamen.")

                            if len(coords_data) == len(energies_data):
                                for i in range(len(energies_data)):
                                    all_found_conformations.append(
                                        (energies_data[i], coords_data[i], f"{iteration_name}/{set_type}", species_list)
                                    )
                            else:
                                print(f"    --> Warnung: Anzahl der Koordinaten und Energien stimmt nicht überein. Überspringe.")
                        else:
                            print(f"    --> Warnung: Gruppe für {molecule_name} enthält keine 'coordinates' oder 'formation_energies'. Überspringe.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return

    if not all_found_conformations:
        print(f"\nSuche abgeschlossen. Keine Daten für '{molecule_name}' gefunden.")
        return
    
    print("\nSortiere alle gefundenen Konformationen nach ihrer Bildungsenthalpie...")
    all_found_conformations.sort(key=lambda x: x[0], reverse=True)

    print(f"Schreibe die sortierten Daten in '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Sortierte Konformationen für das Molekül: {molecule_name}\n")
        outfile.write("Sortiert nach Bildungsenthalpie (höchste zuerst).\n")
        outfile.write("="*80 + "\n\n")

        for i, (energy, coords, location, species) in enumerate(all_found_conformations):
            outfile.write(f"--- Konformation Rang {i+1} (gefunden in {location}) ---\n")
            outfile.write(f"Bildungsenthalpie: {energy}\n")
            outfile.write("Koordinaten:\n")
            
            num_atoms = len(coords)
            for atom_index in range(num_atoms):
                # Verwende das Atomsymbol, falls vorhanden, sonst "Atom"
                atom_symbol = species[atom_index] if species else f"Atom {atom_index+1}"
                x, y, z = coords[atom_index]
                outfile.write(f"    {atom_symbol:<4}  {x:12.6f} {y:12.6f} {z:12.6f}\n")
            
            outfile.write("\n")
            
            # NEU: Berechne und schreibe die Distanzen
            if num_atoms > 1:
                outfile.write("Paarweise Abstände (in Ångström):\n")
                # Erzeuge alle einzigartigen Paare von Atom-Indizes, z.B. (0, 1), (0, 2), ...
                atom_index_pairs = list(combinations(range(num_atoms), 2))
                # Berechne die Distanzen für genau diese Paare
                distances = pdist(coords)
                
                for pair, dist in zip(atom_index_pairs, distances):
                    idx1, idx2 = pair
                    # Hole die Atomsymbole für die Ausgabe
                    symbol1 = species[idx1] if species else "Atom"
                    symbol2 = species[idx2] if species else "Atom"
                    outfile.write(f"    Distanz {symbol1}({idx1+1}) - {symbol2}({idx2+1}): {dist:.6f}\n")

            outfile.write("\n\n")

    print(f"\nErfolgreich {len(all_found_conformations)} Konformationen in '{output_filename}' geschrieben.")

if __name__ == '__main__':
    h5_file = 'ani_kfold_dataset_with_formation_energies.h5' 
    molecule_name = input("Geben Sie den exakten Namen der Molekülgruppe ein (z.B. O2, H2O2): ")
    
    if molecule_name:
        find_and_sort_molecule(h5_file, molecule_name)
    else:
        print("Kein Molekülname eingegeben. Das Programm wird beendet.")