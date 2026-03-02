import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings('ignore')

# Set the desired Seaborn style and color palette
sns.set_theme(style='darkgrid')
sns.set_palette('viridis')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings('ignore')

# Set the desired Seaborn style and color palette
sns.set_theme(style='darkgrid')
sns.set_palette('viridis')

def parse_filename(filename):
    """
    Parse filename to extract image, alpha, and inpainting.
    Expected format: {image}_{alpha}_{inpainting}_{omega}.csv
    """
    # Try the 4-part pattern first
    # Note: Assuming the third value is 'inpainting' based on the user's request.
    pattern_4part = r'(\d+)_(\d+\.?\d*)_(\d+\.?\d*)_(\d+\.?\d*)\.csv'
    match = re.match(pattern_4part, filename)
    if match:
        image = match.group(1)
        alpha = float(match.group(2))
        inpainting = float(match.group(3))
        omega = float(match.group(4))
        return image, alpha, inpainting

    # Try 3-part pattern as fallback
    pattern_3part = r'(\d+)_(\d+\.?\d*)_(\d+\.?\d*)\.csv'
    match = re.match(pattern_3part, filename)
    if match:
        image = match.group(1)
        alpha = float(match.group(2))
        inpainting = float(match.group(3)) # Assuming this is inpainting
        return image, alpha, inpainting

    # Try original 2-part pattern as fallback
    pattern_2part = r'(\d+)_(\d+\.?\d*)\.csv'
    match = re.match(pattern_2part, filename)
    if match:
        image = match.group(1)
        alpha = float(match.group(2))
        return image, alpha, None # No inpainting value found

    return None, None, None

def load_all_data(directory_path):
    """
    Load all CSV files from a directory, filtering for inpainting=0.25.
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")

    csv_files = list(directory.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory_path}.")

    all_dfs = []
    print(f"Found {len(csv_files)} CSV files. Processing...")

    for csv_file in csv_files:
        filename = csv_file.name
        image, alpha, inpainting = parse_filename(filename)

        # Skip files that don't match the required inpainting value
        if inpainting is None or not np.isclose(inpainting, 0.25, atol=1e-5):
            print(f"Skipping file {filename}: inpainting value not found or not 0.25.")
            continue

        try:
            df = pd.read_csv(csv_file)
            df['image'] = image
            df['alpha'] = alpha
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No valid data could be loaded after filtering.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Filter alpha to only include those between 1 and 4 (inclusive)
    initial_count = len(combined_df)
    combined_df = combined_df[(combined_df['alpha'] >= 1) & (combined_df['alpha'] <= 4)]
    filtered_count = len(combined_df)

    print(f"Filtered data: kept {filtered_count} rows out of {initial_count} (alpha between 1 and 4)")

    if len(combined_df) == 0:
        raise ValueError("No data remains after filtering alpha between 1 and 4.")

    combined_df['alpha'] = combined_df['alpha'].astype('category')
    return combined_df

def create_final_summary_plot(combined_data, out_dir):
    """
    Create a 1x3 plot showing final Train PSNR, Test PSNR, and L2 Spectral Norm Product vs. Alpha.
    """
    # Find the maximum epoch for each image and select that row
    final_data = combined_data.loc[combined_data.groupby(['image', 'alpha'])['epoch'].idxmax()]
    
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics for plotting
    alpha_values = sorted(final_data['alpha'].unique())
    
    # Get colors
    default_palette = sns.color_palette("tab10")
    blue_color = default_palette[0]
    orange_color = default_palette[1]
    viridis_palette = sns.color_palette("viridis", n_colors=len(combined_data['alpha'].unique()))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Final Train PSNR vs. Alpha
    train_means = final_data.groupby('alpha')['train_psnr'].mean()
    train_stds = final_data.groupby('alpha')['train_psnr'].std()
    axes[0].plot(alpha_values, train_means, 'o-', color=blue_color, linewidth=2, markersize=5)
    axes[0].fill_between(alpha_values, 
                         train_means - train_stds, 
                         train_means + train_stds, 
                         alpha=0.3, color=blue_color)
    axes[0].set_xlabel('Alpha', fontsize=12)
    axes[0].set_ylabel('PSNR', fontsize=12)
    axes[0].set_title('Final Train PSNR', fontsize=12, fontweight='bold')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plot 2: Final Test PSNR vs. Alpha
    test_means = final_data.groupby('alpha')['test_psnr'].mean()
    test_stds = final_data.groupby('alpha')['test_psnr'].std()
    axes[1].plot(alpha_values, test_means, 'o-', color=orange_color, linewidth=2, markersize=5)
    axes[1].fill_between(alpha_values, 
                         test_means - test_stds, 
                         test_means + test_stds, 
                         alpha=0.3, color=orange_color)
    axes[1].set_xlabel('Alpha', fontsize=12)
    axes[1].set_ylabel('PSNR', fontsize=12)
    axes[1].set_title('Final Test PSNR', fontsize=12, fontweight='bold')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plot 3: Spectral Norm Product over epochs
    if 'combined_spectral_norm_product' in combined_data.columns:
        sns.lineplot(
            data=combined_data,
            x='epoch',
            y='combined_spectral_norm_product',
            hue='alpha',
            errorbar=None,
            marker='o',
            ax=axes[2],
            palette=viridis_palette
        )
        axes[2].set_title('Network L2-Spectral Norm', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('L2 Spectral Norm Product', fontsize=12)
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        # Position legend outside the plot area
        axes[2].legend(title='alpha', bbox_to_anchor=(1.05, 0.95), loc='upper left')
    else:
        axes[2].set_title('Product Norm Not Found', fontsize=14)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('', fontsize=12)

    # Use plt.tight_layout() to fit everything in the figure
    plt.tight_layout()
    # Remove the problematic plt.subplots_adjust() call
    # The bbox_inches='tight' below will handle the spacing for saving.
    
    # Save the figure with a tight bounding box
    plt.savefig(Path(out_dir) / 'final_summary_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_spectral_norm_plots(combined_data, out_dir):
    """
    Create the 1x4 plot of spectral norms over epochs with improved spacing.
    """
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Find all layer spectral norm columns
    layer_spectral_norm_cols = [col for col in combined_data.columns if 'layer_' in col and 'linear_spectral_norm' in col]
    layer_spectral_norm_cols.sort()

    print(f"Found spectral norm columns for layers: {layer_spectral_norm_cols}")

    # Define the 'viridis' palette
    viridis_palette = sns.color_palette("viridis", n_colors=len(combined_data['alpha'].unique()))

    # Create the combined 1x4 plot with better spacing
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))  # Increased width and height
    
    # Check if there are at least 3 layers to plot
    if len(layer_spectral_norm_cols) < 3:
        print("WARNING: Less than 3 spectral norm layers found. The 1x4 layout may be incomplete.")

    # Plot the first three spectral norm layers
    for idx, col in enumerate(layer_spectral_norm_cols):
        if idx >= 3:
            break

        sns.lineplot(
            data=combined_data,
            x='epoch',
            y=col,
            hue='alpha',
            errorbar=None,
            marker='o',
            ax=axes[idx],
            palette=viridis_palette
        )
        axes[idx].set_title(f'Linear Layer {idx}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel('L2 Spectral Norm', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].get_legend().remove()

    # Plot the product of all combined spectral norms on the fourth subplot
    if 'combined_spectral_norm_product' in combined_data.columns:
        sns.lineplot(
            data=combined_data,
            x='epoch',
            y='combined_spectral_norm_product',
            hue='alpha',
            errorbar=None,
            marker='o',
            ax=axes[3],
            palette=viridis_palette
        )
        axes[3].set_title('Network', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Epoch', fontsize=12)
        axes[3].set_ylabel('Product of L2 Spectral Norms', fontsize=12)
        axes[3].set_yscale('log')
        axes[3].grid(True, alpha=0.3)
        # Position legend outside the plot area
        axes[3].legend(title='alpha', bbox_to_anchor=(1.05, 0.95), loc='upper left')
    else:
        axes[3].set_title('Product Norm Not Found', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Epoch', fontsize=12)
        axes[3].set_ylabel('', fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(
        left=0.06,      # Left margin
        right=0.85,     # Right margin (leave space for legend)
        bottom=0.12,    # Bottom margin
        top=0.88,       # Top margin
        wspace=0.35,    # Width spacing between subplots (key parameter!)
        hspace=0.2      # Height spacing (not needed for 1 row, but good to set)
    )
    
    plt.savefig(Path(out_dir) / 'all_spectral_norm_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_stable_rank_plots(combined_data, out_dir):
    """
    Create the 1x3 plot of stable ranks over epochs for the first 3 linear layers.
    """
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Find all layer stable rank columns - looking for layer_0_stable_rank, layer_1_stable_rank, etc.
    layer_stable_rank_cols = [col for col in combined_data.columns if re.match(r'layer_\d+_stable_rank', col)]
    layer_stable_rank_cols.sort(key=lambda x: int(re.search(r'layer_(\d+)_stable_rank', x).group(1)))

    print(f"Found stable rank columns for layers: {layer_stable_rank_cols}")

    # Define the 'viridis' palette
    viridis_palette = sns.color_palette("viridis", n_colors=len(combined_data['alpha'].unique()))

    # Create the combined 1x3 plot with better spacing
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjusted for 3 subplots
    
    # Check if there are at least 3 layers to plot
    if len(layer_stable_rank_cols) < 3:
        print(f"WARNING: Only {len(layer_stable_rank_cols)} stable rank layers found. The 1x3 layout may be incomplete.")

    # Plot the stable rank for the first 3 layers
    for idx in range(3):
        if idx < len(layer_stable_rank_cols):
            col = layer_stable_rank_cols[idx]
            
            sns.lineplot(
                data=combined_data,
                x='epoch',
                y=col,
                hue='alpha',
                errorbar=None,
                marker='o',
                ax=axes[idx],
                palette=viridis_palette
            )
            axes[idx].set_title(f'Linear Layer {idx}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel('Stable Rank', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            
            # Remove legend from first 2 plots, keep it only on the last plot
            if idx < 2:
                axes[idx].get_legend().remove()
            else:
                # Position legend outside the plot area for the last plot (same as spectral norm plot)
                axes[idx].legend(title='alpha', bbox_to_anchor=(1.05, 0.95), loc='upper left')
        else:
            # If we don't have enough layers, create an empty plot
            axes[idx].set_title(f'Layer {idx} Not Found', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel('', fontsize=12)

    # Adjust spacing between subplots (same as spectral norm plot)
    plt.subplots_adjust(
        left=0.06,      # Left margin
        right=0.85,     # Right margin (leave space for legend)
        bottom=0.12,    # Bottom margin
        top=0.88,       # Top margin
        wspace=0.35,    # Width spacing between subplots
        hspace=0.2      # Height spacing (not needed for 1 row, but good to set)
    )
    
    plt.savefig(Path(out_dir) / 'all_stable_rank_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to parse arguments and run the analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Analyze and plot data from CSV files.')
    parser.add_argument('--directory', default='../logs/scaling', help='Path to the directory containing CSV files.')
    parser.add_argument('--out_dir', default='../logs/scaling', help='Directory to save the plots. Defaults to ../logs/scaling')

    args = parser.parse_args()

    try:
        combined_data = load_all_data(args.directory)

        # Identify all 'combined_spectral_norm' columns and calculate the product for summary stats.
        spectral_norm_cols = [col for col in combined_data.columns if 'combined_spectral_norm' in col]
        print(spectral_norm_cols)
        if spectral_norm_cols:
            combined_data['combined_spectral_norm_product'] = combined_data[spectral_norm_cols].prod(axis=1)

        unique_alpha = combined_data['alpha'].unique()
        print(f"\nFound data for {len(unique_alpha)} different alpha: {sorted(unique_alpha)}")

        create_final_summary_plot(combined_data, args.out_dir)
        create_spectral_norm_plots(combined_data, args.out_dir)
        create_stable_rank_plots(combined_data, args.out_dir)


    except Exception as e:
        print(f"Error during analysis: {e}")
        exit(1)

if __name__ == "__main__":
    main()
