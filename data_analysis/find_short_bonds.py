import os
import re

def analyze_bond_distances(directory="."):
    """
    Searches all *_data.txt files in a directory for O-H/H-O
    bonds and counts how many are shorter than 0.9 Å and how many
    are greater than or equal to 0.9 Å.
    """
    
    # Regular expressions to extract the relevant information
    # Recognizes e.g., "--- Conformation Rank 123 ..."
    conformation_pattern = re.compile(r"--- Konformation Rang (\d+)") # "Rang" is left as is, assuming it's a key part of the format
    
    # Recognizes e.g., "Distanz O(1) - H(3): 0.987654 Å"
    # It ensures that a bond between O and H (in any order) is present.
    distance_pattern = re.compile(
        r"^\s*Distanz\s+(O\(\d+\)|H\(\d+\))\s+-\s+(O\(\d+\)|H\(\d+\)):\s+([\d.]+)" # "Distanz" is also left as is
    )

    short_bonds_found = []
    # Counter for longer bonds
    long_bonds_count = 0

    print("Starting the analysis of bond distances...")

    # Find all relevant files in the specified directory
    files_to_check = [f for f in os.listdir(directory) if f.endswith('_data.txt')]

    if not files_to_check:
        print("No '_data.txt' files found in the current directory.")
        return

    # Process each found file
    for filename in sorted(files_to_check):
        found_in_this_file = []
        current_conformation = "Unknown"
        
        try:
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    # Check if a new conformation section begins
                    conformation_match = conformation_pattern.search(line)
                    if conformation_match:
                        current_conformation = f"Rank {conformation_match.group(1)}"
                        continue

                    # Check if the line describes a distance
                    distance_match = distance_pattern.search(line)
                    if distance_match:
                        atom1, atom2, distance_str = distance_match.groups()
                        
                        # Ensure that it is an O-H bond
                        is_oh_bond = (atom1.startswith('O') and atom2.startswith('H')) or \
                                     (atom1.startswith('H') and atom2.startswith('O'))
                        
                        if is_oh_bond:
                            distance = float(distance_str)
                            
                            # --- Extended logic ---
                            if distance < 0.9:
                                result = {
                                    "file": filename,
                                    "conformation": current_conformation,
                                    "line": line.strip()
                                }
                                found_in_this_file.append(result)
                            else:
                                # Increment the counter for longer bonds
                                long_bonds_count += 1

        except Exception as e:
            print(f"Error while reading file {filename}: {e}")

        if found_in_this_file:
            short_bonds_found.extend(found_in_this_file)
            print(f"\n--- Hits (< 0.9 Å) in file: {filename} ---")
            for hit in found_in_this_file:
                print(f"  - {hit['conformation']}: {hit['line']}")
    
    # --- Final Summary (ADJUSTED) ---
    print("\n" + "="*70)
    print("Analysis complete.")
    print(f"- Found a total of {len(short_bonds_found)} O-H/H-O bonds < 0.9 Å.")
    print(f"- Found a total of {long_bonds_count} O-H/H-O bonds >= 0.9 Å.")
    print("="*70)


if __name__ == "__main__":
    analyze_bond_distances()