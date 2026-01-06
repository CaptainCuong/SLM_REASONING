#!/usr/bin/env python3
"""
Visualize gradient dot products from .npy file.
Plots dot products against their index.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def visualize_dot_products(npy_file, output_file=None):
    """
    Visualize dot products from numpy array.

    Args:
        npy_file: Path to .npy file containing dot products
        output_file: Optional path to save the plot
    """
    # Load dot products
    # first_file = np.load("data/grad_dot_products_algebra_self_generated_vs_mean_math500.npy")
    dot_products = np.load(npy_file)
    # dot_products = np.concatenate([first_file, dot_products])

    print(f"Loaded {len(dot_products)} dot products from {npy_file}")
    print(f"Shape: {dot_products.shape}")
    print(f"Min: {np.min(dot_products):.6e}")
    print(f"Max: {np.max(dot_products):.6e}")
    print(f"Mean: {np.mean(dot_products):.6e}")
    print(f"Std: {np.std(dot_products):.6e}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Dot products vs index
    ax1 = axes[0]
    indices = np.arange(len(dot_products))
    ax1.plot(indices, dot_products, linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Example Index')
    ax1.set_ylabel('Dot Product')
    ax1.set_title('Gradient Dot Products vs Example Index')
    ax1.grid(True, alpha=0.3)

    # Add horizontal line for mean
    mean_val = np.mean(dot_products)
    ax1.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2e}')
    ax1.legend()

    # Plot 2: Histogram
    ax2 = axes[1]
    ax2.hist(dot_products, bins=100, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Dot Product Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Dot Products')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add vertical line for mean
    ax2.axvline(x=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2e}')
    ax2.legend()

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()

    # Print top and bottom indices
    print("\n" + "="*80)
    print("Top 10 highest dot products:")
    print("="*80)
    top_indices = np.argsort(dot_products)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Index {idx}: {dot_products[idx]:.6e}")

    print("\n" + "="*80)
    print("Top 10 lowest dot products:")
    print("="*80)
    bottom_indices = np.argsort(dot_products)[:10]
    for rank, idx in enumerate(bottom_indices, 1):
        print(f"  {rank}. Index {idx}: {dot_products[idx]:.6e}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize gradient dot products from .npy file'
    )

    parser.add_argument(
        'npy_file',
        type=str,
        help='Path to .npy file containing dot products'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default="temp_str/grad_dot_products_plot.png",
        help='Path to save plot (e.g., plot.png). If not provided, will display interactively.'
    )

    args = parser.parse_args()

    visualize_dot_products(args.npy_file, args.output)


if __name__ == "__main__":
    main()
