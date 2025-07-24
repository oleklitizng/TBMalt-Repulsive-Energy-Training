import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def parse_rmse_from_file_hartree(filepath):
    """
    Parses a file to extract the RMSE values in Hartree.
    """
    results = {}
    current_method = None
    # Modified regex to extract the Hartree value
    rmse_regex = re.compile(r'RMSE on Test Set: (\d+\.\d+) Hartree')

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rmse_match = rmse_regex.search(line)
            # If the line contains an RMSE value
            if rmse_match:
                if current_method and current_method in results:
                    results[current_method].append(float(rmse_match.group(1)))
            # Otherwise, it's a new method name
            else:
                current_method = line
                results[current_method] = []
    return results

def visualize_grouped_barchart_hartree(data, output_filename='RMSE_Final_Comparison_Hartree.png'):
    """
    Creates a grouped bar chart from the RMSE data in Hartree.
    """
    # Convert data to a "long" format
    df = pd.DataFrame(data)
    df = df.reset_index().rename(columns={'index': 'Test'})
    df_long = pd.melt(df, id_vars=['Test'], var_name='Method', value_name='RMSE')

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(data=df_long, x='Test', y='RMSE', hue='Method', ax=ax, palette='viridis')

    # Modified titles and axis labels
    ax.set_title('Comparison of Methods per Test Run (5-Fold CV)', fontsize=16, pad=20)
    ax.set_ylabel('RMSE (Hartree)', fontsize=12)
    ax.set_xlabel('Test Index (CV-Fold)', fontsize=12)
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Modified formatting for the values above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.5f', fontsize=8, padding=3, rotation=45)

    plt.ylim(top=ax.get_ylim()[1] * 1.15)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(output_filename, dpi=300)
    print(f"Grouped bar chart saved to: {output_filename}")


if __name__ == '__main__':
    rmse_file = 'RMSE.txt'
    # The file is parsed with the new function
    parsed_data = parse_rmse_from_file_hartree(rmse_file)

    if parsed_data:
        print("Successfully parsed Hartree values:")
        for method, values in parsed_data.items():
            print(f"- {method}: {values}")

        # The final visualization is created
        visualize_grouped_barchart_hartree(parsed_data)
    else:
        print("Could not extract data from the file.")