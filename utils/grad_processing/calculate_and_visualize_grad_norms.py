#!/usr/bin/env python3
"""
Calculate gradient norms from projection files and visualize them across indices.

This script loads gradient projection files from a directory, calculates the
L2 norm (magnitude) of each gradient, and creates visualizations similar to
the dot product visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm


def load_gradient_projection(proj_file):
    """
    Load gradient projection from a .pt file.

    Args:
        proj_file: Path to .pt file containing gradient projection

    Returns:
        Gradient projection as numpy array
    """
    data = torch.load(proj_file, map_location='cpu')

    # The file contains a dict with 'proj' key containing the gradient tensor
    # Format: {'proj': tensor, 'iter': int, 'batch_size': int, 'batch_idx': list}
    grad = data['proj']

    # Convert to numpy (convert BFloat16 to float32 first for numpy compatibility)
    if isinstance(grad, torch.Tensor):
        grad = grad.float().cpu().numpy()

    return grad


def calculate_gradient_norms(grad_dir, output_file=None):
    """
    Calculate L2 norms of gradient projections from a directory.

    Args:
        grad_dir: Directory containing gradient projection .pt files
        output_file: Optional path to save the norms as .npy file

    Returns:
        Array of gradient norms
    """
    grad_path = Path(grad_dir)

    # Get all .pt files sorted by iteration number
    proj_files = sorted(
        grad_path.glob("proj_iter_*.pt"),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    if len(proj_files) == 0:
        raise ValueError(f"No proj_iter_*.pt files found in {grad_dir}")

    print(f"Found {len(proj_files)} gradient projection files")
    print(f"Processing files from {proj_files[0].name} to {proj_files[-1].name}")

    # Calculate norms for each gradient
    norms = []
    for proj_file in tqdm(proj_files, desc="Calculating gradient norms"):
        grad = load_gradient_projection(proj_file)

        # Calculate L2 norm (Euclidean norm)
        norm = np.linalg.norm(grad.flatten())
        norms.append(norm)

    norms = np.array(norms)

    print(f"\nCalculated {len(norms)} gradient norms")
    print(f"Shape: {norms.shape}")
    print(f"Min: {np.min(norms):.6e}")
    print(f"Max: {np.max(norms):.6e}")
    print(f"Mean: {np.mean(norms):.6e}")
    print(f"Std: {np.std(norms):.6e}")

    # Save norms if output file specified
    if output_file:
        np.save(output_file, norms)
        print(f"\nSaved gradient norms to: {output_file}")

    return norms


def visualize_gradient_norms(norms, output_plot=None):
    """
    Visualize gradient norms across indices.

    Args:
        norms: Array of gradient norms
        output_plot: Optional path to save the plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Gradient norms vs index
    ax1 = axes[0]
    indices = np.arange(len(norms))
    ax1.plot(indices, norms, linewidth=0.5, alpha=0.7, color='blue')
    ax1.set_xlabel('Example Index')
    ax1.set_ylabel('Gradient Norm (L2)')
    ax1.set_title('Gradient Norms vs Example Index')
    ax1.grid(True, alpha=0.3)

    # Add horizontal line for mean
    mean_val = np.mean(norms)
    ax1.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2e}')
    ax1.legend()

    # Plot 2: Histogram
    ax2 = axes[1]
    ax2.hist(norms, bins=100, alpha=0.7, edgecolor='black', color='blue')
    ax2.set_xlabel('Gradient Norm Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Gradient Norms')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add vertical line for mean
    ax2.axvline(x=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2e}')
    ax2.legend()

    plt.tight_layout()

    # Save or show
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_plot}")
    else:
        plt.show()

    # Print top and bottom indices
    print("\n" + "="*80)
    print("Top 10 highest gradient norms:")
    print("="*80)
    top_indices = np.argsort(norms)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Index {idx}: {norms[idx]:.6e}")

    print("\n" + "="*80)
    print("Top 10 lowest gradient norms:")
    print("="*80)
    bottom_indices = np.argsort(norms)[:10]
    for rank, idx in enumerate(bottom_indices, 1):
        print(f"  {rank}. Index {idx}: {norms[idx]:.6e}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and visualize gradient norms from projection files'
    )

    parser.add_argument(
        'grad_dir',
        type=str,
        help='Directory containing gradient projection files (proj_iter_*.pt)'
    )
    parser.add_argument(
        '--output_npy',
        type=str,
        default=None,
        help='Path to save gradient norms as .npy file (e.g., grad_norms.npy)'
    )
    parser.add_argument(
        '--output_plot',
        '-o',
        type=str,
        default="./temp_str/grad_norms_plot.png",
        help='Path to save visualization plot (e.g., grad_norms_plot.png)'
    )
    parser.add_argument(
        '--load_npy',
        type=str,
        default=None,
        help='Load pre-calculated norms from .npy file instead of calculating'
    )

    args = parser.parse_args()

    # Either load pre-calculated norms or calculate them
    if args.load_npy:
        print(f"Loading pre-calculated norms from {args.load_npy}")
        norms = np.load(args.load_npy)
        print(f"Loaded {len(norms)} gradient norms")
        print(f"Min: {np.min(norms):.6e}, Max: {np.max(norms):.6e}, Mean: {np.mean(norms):.6e}")
    else:
        # Calculate norms from gradient directory
        norms = calculate_gradient_norms(args.grad_dir, args.output_npy)

    # Visualize the norms
    visualize_gradient_norms(norms, args.output_plot)


if __name__ == "__main__":
    main()
