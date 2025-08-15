import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

# Ein Dictionary mit typischen kovalenten Bindungslängen in Ångström.
# Dies wird verwendet, um eine hilfreiche Referenzlinie in jedem Plot zu zeichnen.
AVERAGE_BOND_LENGTHS = {
    'C-H': 1.09, 'C-C': 1.54, 'C-N': 1.47, 'C-O': 1.43,
    'H-H': 0.74, 'H-N': 1.01, 'H-O': 0.96,
    'N-N': 1.45, 'N-O': 1.36, 'O-O': 1.48
}

def collect_all_bond_lengths(directory=".") -> Dict[str, List[float]]:
    """
    Sammelt alle Bindungslängen aus den *_data.txt Dateien und gruppiert sie nach Bindungstyp.
    """
    # Allgemeiner Regex, der jeden Atom-Typ (z.B. C, H, N, O) erkennt
    distance_pattern = re.compile(
        r"^\s*Distanz\s+([A-Z][a-z]?)\(\d+\)\s+-\s+([A-Z][a-z]?)\(\d+\):\s+([\d.]+)"
    )
    
    all_distances = {}
    files_to_check = [f for f in os.listdir(directory) if f.endswith('_data.txt')]

    if not files_to_check:
        print("Warning: No '_data.txt' files found in the current directory.")
        return {}

    print(f"Collecting bond distances from {len(files_to_check)} files...")

    for filename in sorted(files_to_check):
        try:
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    match = distance_pattern.search(line)
                    if match:
                        symbol1, symbol2, distance_str = match.groups()
                        
                        # Standardisierter Schlüssel (z.B. C-H statt H-C)
                        bond_type = '-'.join(sorted([symbol1, symbol2]))
                        
                        if bond_type not in all_distances:
                            all_distances[bond_type] = []
                        
                        all_distances[bond_type].append(float(distance_str))
        except Exception as e:
            print(f"Error while reading file {filename}: {e}")
            
    return all_distances

def create_distribution_plot(bond_type: str, distances: List[float]):
    """
    Erstellt ein Histogramm mit einer KDE-Kurve für einen spezifischen Bindungstyp.
    """
    if not distances:
        print(f"No distances found for bond type {bond_type}. Skipping plot.")
        return

    output_filename = f"bond_distribution_{bond_type}.png"
    print(f"\nCreating plot for {bond_type} from {len(distances)} data points...")

    plt.figure(figsize=(12, 7))
    
    sns.histplot(distances, kde=True, binwidth=0.05, stat="density")
    
    plt.title(f'Distribution of {bond_type} Bond Lengths', fontsize=16)
    plt.xlabel('Bond Distance (Å)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Füge eine Referenzlinie hinzu, wenn eine typische Länge bekannt ist
    if bond_type in AVERAGE_BOND_LENGTHS:
        avg_len = AVERAGE_BOND_LENGTHS[bond_type]
        plt.axvline(avg_len, color='red', linestyle='--', linewidth=1.5, 
                    label=f'Typical Covalent Bond (~{avg_len} Å)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Plot successfully saved as '{output_filename}'")
    
    # Statistische Zusammenfassung für diesen Bindungstyp
    distances_np = np.array(distances)
    print("--- Statistical Summary ---")
    print(f"Shortest bond: {np.min(distances_np):.4f} Å")
    print(f"Longest bond:  {np.max(distances_np):.4f} Å")
    print(f"Average length: {np.mean(distances_np):.4f} Å")


if __name__ == "__main__":
    # Schritt 1: Sammle alle Bindungslängen, gruppiert nach Typ.
    all_bond_data = collect_all_bond_lengths()
    
    # Schritt 2: Erstelle für jeden gefundenen Bindungstyp einen eigenen Plot.
    if not all_bond_data:
        print("Analysis complete. No bond data was found.")
    else:
        print(f"\nFound data for {len(all_bond_data)} unique bond types: {', '.join(sorted(all_bond_data.keys()))}")
        for bond_type, distances in sorted(all_bond_data.items()):
            create_distribution_plot(bond_type, distances)
        print("\nAll plots have been generated.")