import re
import matplotlib.pyplot as plt
import pandas as pd

def plot_repulsive_energies(file_path):
    """
    Parses a file containing conformational data and plots the pairwise
    distances versus repulsive energies for O-O, O-H, and H-H interactions.
    This version uses a more robust parsing method.

    Args:
        file_path (str): The path to the input text file.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # --- MODIFIED PARSING LOGIC ---
    # Split the content into blocks, with each block starting with "Distanz"
    distance_blocks = content.split('Distanz ')[1:] # Skip the initial header
    data = []

    # Define simpler, more robust regex patterns for each piece of info
    pair_pattern = re.compile(r"([OH])\(\d\)\s+-\s+([OH])\(\d\):\s+([\d.]+)")
    xtb_pattern = re.compile(r"xTB Repulsive Energy\s*:\s*([\d.-]+)")
    ptbp_pattern = re.compile(r"PTBP Repulsive Energy\s*:\s*([\d.-]+)")
    gamma_pattern = re.compile(r"GAMMA Repulsive Energy\s*:\s*([\d.-]+)")
    spline_pattern = re.compile(r"Spline Repulsive Energy:\s*([\d.-]+)")

    for block in distance_blocks:
        pair_match = pair_pattern.search(block)
        xtb_match = xtb_pattern.search(block)
        ptbp_match = ptbp_pattern.search(block)
        gamma_match = gamma_pattern.search(block)
        spline_match = spline_pattern.search(block)

        # Only add the data if all parts were successfully found
        if all((pair_match, xtb_match, ptbp_match, gamma_match, spline_match)):
            atom1, atom2, distance = pair_match.groups()
            pair = '-'.join(sorted((atom1, atom2)))
            
            data.append([
                pair,
                float(distance),
                float(xtb_match.group(1)),
                float(ptbp_match.group(1)),
                float(gamma_match.group(1)),
                float(spline_match.group(1))
            ])

    if not data:
        print("Warning: No data was successfully parsed from the file. No plots will be generated.")
        return

    df = pd.DataFrame(data, columns=['pair', 'distance', 'xtb', 'ptbp', 'gamma', 'spline'])

    # --- PLOTTING LOGIC (Unchanged) ---
    # Create separate plots for each interaction type
    for pair_type in ['O-O', 'H-O', 'H-H']:
        subset = df[df['pair'] == pair_type]

        if not subset.empty:
            plt.figure(figsize=(10, 6))
            plt.scatter(subset['distance'], subset['xtb'], label='xTB', alpha=0.7)
            plt.scatter(subset['distance'], subset['ptbp'], label='PTBP', alpha=0.7)
            plt.scatter(subset['distance'], subset['gamma'], label='GAMMA', alpha=0.7)
            plt.scatter(subset['distance'], subset['spline'], label='Spline', alpha=0.7)

            plt.xlabel("Paarweise Abstände (Å)")
            plt.ylabel("Repulsive Energies (Hartree)")
            plt.title(f"Repulsive Energies vs. Pairwise Distance for {pair_type} Interactions")
            plt.legend()
            plt.grid(True)
            
            # Adjust y-axis for better visualization, especially for H-H and H-O
            if pair_type == 'H-H':
                plt.ylim(-0.01, 0.05)
            elif pair_type == 'H-O':
                 plt.ylim(-0.1, 1.5)
            
            output_filename = f"{pair_type}_interactions_plot.png"
            plt.savefig(output_filename)
            plt.close()
            print(f"Plot saved as {output_filename}")

# Calling the function with the provided file name
plot_repulsive_energies('H2O2_data_with_repulsive_energies.txt')