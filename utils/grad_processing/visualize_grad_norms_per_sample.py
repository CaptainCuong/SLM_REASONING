#!/usr/bin/env python3
"""
Calculate and visualize gradient norms per sample from projection files.

This script loads gradient projection files, calculates L2 norm for each sample
(handling batch dimensions), and creates comprehensive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm
import json


def load_projection_files(grad_dir):
    """
    Load all gradient projection files from a directory.

    Args:
        grad_dir: Directory containing gradient projection .pt files

    Returns:
        all_norms: Array of gradient norms per sample
        metadata: Dictionary with file information
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

    all_norms = []
    file_info = []
    total_samples = 0

    for proj_file in tqdm(proj_files, desc="Loading gradient projections"):
        # Load the file
        data = torch.load(proj_file, map_location='cpu')

        # Extract projection tensor: [B, D] where B is batch size, D is projection dim
        proj = data['proj']

        # Convert to float32 if needed (handle BFloat16)
        if proj.dtype == torch.bfloat16:
            proj = proj.float()

        # Calculate L2 norm for each sample in the batch
        # Flatten each sample's projection and compute norm
        batch_size = proj.shape[0]
        proj_flat = proj.reshape(batch_size, -1)  # [B, D]

        # Compute L2 norm along dimension 1 (across features)
        norms = torch.norm(proj_flat, p=2, dim=1).cpu().numpy()  # [B]

        all_norms.append(norms)

        # Store metadata
        file_info.append({
            'filename': proj_file.name,
            'iter': data.get('iter', -1),
            'batch_size': batch_size,
            'start_idx': total_samples,
            'end_idx': total_samples + batch_size,
            'mean_norm': float(norms.mean()),
            'std_norm': float(norms.std()),
        })

        total_samples += batch_size

    # Concatenate all norms
    all_norms = np.concatenate(all_norms)

    metadata = {
        'total_samples': total_samples,
        'total_files': len(proj_files),
        'projection_dim': proj.shape[1],
        'files': file_info,
    }

    print(f"\nLoaded {total_samples} samples from {len(proj_files)} files")
    print(f"Projection dimension: {proj.shape[1]}")

    return all_norms, metadata


def compute_statistics(norms):
    """Compute comprehensive statistics on gradient norms."""
    stats = {
        'count': len(norms),
        'mean': float(np.mean(norms)),
        'std': float(np.std(norms)),
        'median': float(np.median(norms)),
        'min': float(np.min(norms)),
        'max': float(np.max(norms)),
        'percentiles': {
            '1': float(np.percentile(norms, 1)),
            '5': float(np.percentile(norms, 5)),
            '10': float(np.percentile(norms, 10)),
            '25': float(np.percentile(norms, 25)),
            '75': float(np.percentile(norms, 75)),
            '90': float(np.percentile(norms, 90)),
            '95': float(np.percentile(norms, 95)),
            '99': float(np.percentile(norms, 99)),
        },
        'coefficient_of_variation': float(np.std(norms) / np.mean(norms)) if np.mean(norms) != 0 else 0,
    }

    return stats


def visualize_gradient_norms(norms, metadata, output_dir, max_norm=20):
    """
    Create comprehensive visualizations for gradient norms.

    Args:
        norms: Array of gradient norms per sample
        metadata: Dictionary with file and sample information
        output_dir: Directory to save visualizations
        max_norm: Maximum norm value to display (default: 20)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter norms for visualization
    mask = norms < max_norm
    filtered_norms = norms[mask]
    filtered_indices = np.where(mask)[0]

    print(f"Filtering: {len(filtered_norms)} / {len(norms)} samples with norm < {max_norm}")
    print(f"Excluded: {len(norms) - len(filtered_norms)} samples with norm >= {max_norm}")

    # Compute statistics on ALL norms
    stats = compute_statistics(norms)

    # Compute statistics on FILTERED norms
    stats_filtered = compute_statistics(filtered_norms)

    # Save statistics
    stats_file = output_dir / 'gradient_norm_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'statistics_all': stats,
            'statistics_filtered': stats_filtered,
            'filter_threshold': max_norm,
            'n_samples_all': len(norms),
            'n_samples_filtered': len(filtered_norms),
            'metadata': metadata
        }, f, indent=2)
    print(f"Statistics saved to {stats_file}")

    # 1. Line plot of gradient norms across samples (FILTERED)
    print("Creating line plot...")
    plt.figure(figsize=(16, 6))
    plt.plot(filtered_indices, filtered_norms, linewidth=0.5, alpha=0.7, color='steelblue')
    plt.axhline(y=stats_filtered['mean'], color='red', linestyle='--', linewidth=1.5,
                label=f"Mean (filtered): {stats_filtered['mean']:.4f}")
    plt.axhline(y=stats_filtered['median'], color='orange', linestyle='--', linewidth=1.5,
                label=f"Median (filtered): {stats_filtered['median']:.4f}")
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Gradient Norm (L2)', fontsize=12)
    plt.title(f'Gradient Norms (< {max_norm}) Across Samples\n{len(filtered_norms)} / {len(norms)} samples shown',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max_norm)

    # Increase x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norms_line_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_norms_line_plot.png")

    # 2. Histogram with KDE (FILTERED)
    print("Creating histogram with KDE...")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(filtered_norms, bins=100, alpha=0.7, edgecolor='black', color='steelblue', density=True)

    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(filtered_norms)
    x_range = np.linspace(filtered_norms.min(), filtered_norms.max(), 1000)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    ax.axvline(x=stats_filtered['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {stats_filtered['mean']:.4f}")
    ax.axvline(x=stats_filtered['median'], color='orange', linestyle='--', linewidth=2,
               label=f"Median: {stats_filtered['median']:.4f}")
    ax.set_xlabel('Gradient Norm', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of Gradient Norms (< {max_norm})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, max_norm)
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norms_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_norms_histogram.png")

    # 3. Box plot and violin plot (FILTERED)
    print("Creating box and violin plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    axes[0].boxplot(filtered_norms, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[0].set_ylabel('Gradient Norm', fontsize=12)
    axes[0].set_ylim(0, max_norm)
    axes[0].set_title('Box Plot of Gradient Norms', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Violin plot
    parts = axes[1].violinplot([filtered_norms], vert=True, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    axes[1].set_ylabel('Gradient Norm', fontsize=12)
    axes[1].set_ylim(0, max_norm)
    axes[1].set_title('Violin Plot of Gradient Norms', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norms_box_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_norms_box_violin.png")

    # 4. Percentile plot
    print("Creating percentile plot...")
    percentiles = np.arange(0, 101, 1)
    percentile_values = np.percentile(norms, percentiles)

    plt.figure(figsize=(12, 6))
    plt.plot(percentiles, percentile_values, linewidth=2, color='steelblue')
    plt.fill_between(percentiles, percentile_values, alpha=0.3, color='steelblue')
    plt.xlabel('Percentile', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.title('Gradient Norm Percentile Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Mark key percentiles
    key_percentiles = [25, 50, 75, 90, 95, 99]
    for p in key_percentiles:
        val = np.percentile(norms, p)
        plt.axhline(y=val, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        plt.text(2, val, f'P{p}: {val:.4f}', fontsize=9, va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norms_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_norms_percentiles.png")

    # 5. Log-scale histogram
    print("Creating log-scale histogram...")
    plt.figure(figsize=(12, 6))
    plt.hist(norms, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    plt.yscale('log')
    plt.axvline(x=stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {stats['mean']:.4f}")
    plt.axvline(x=stats['median'], color='orange', linestyle='--', linewidth=2,
                label=f"Median: {stats['median']:.4f}")
    plt.xlabel('Gradient Norm', fontsize=12)
    plt.ylabel('Frequency (log scale)', fontsize=12)
    plt.title('Distribution of Gradient Norms (Log Scale)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norms_histogram_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_norms_histogram_log.png")

    # 6. Moving average plot (FILTERED)
    print("Creating moving average plot...")
    window_size = min(100, len(filtered_norms) // 10)
    if window_size > 1:
        moving_avg = np.convolve(filtered_norms, np.ones(window_size)/window_size, mode='valid')

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Original with moving average
        axes[0].plot(filtered_indices, filtered_norms, linewidth=0.3, alpha=0.3, color='gray', label='Original')
        axes[0].plot(filtered_indices[:len(moving_avg)], moving_avg, linewidth=1.5,
                     color='steelblue', label=f'Moving Avg (window={window_size})')
        axes[0].axhline(y=stats_filtered['mean'], color='red', linestyle='--', linewidth=1,
                        label=f"Mean: {stats_filtered['mean']:.4f}")
        axes[0].set_xlabel('Sample Index', fontsize=12)
        axes[0].set_ylabel('Gradient Norm', fontsize=12)
        axes[0].set_title(f'Gradient Norms (< {max_norm}) with Moving Average', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, max_norm)

        # Increase x-axis ticks
        axes[0].xaxis.set_major_locator(plt.MaxNLocator(nbins=20))

        # Deviation from moving average
        deviation = filtered_norms[window_size-1:len(moving_avg)+window_size-1] - moving_avg
        axes[1].plot(filtered_indices[window_size-1:len(moving_avg)+window_size-1], deviation,
                    linewidth=0.5, color='coral')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_xlabel('Sample Index', fontsize=12)
        axes[1].set_ylabel('Deviation from Moving Avg', fontsize=12)
        axes[1].set_title('Deviation from Moving Average', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(nbins=20))

        plt.tight_layout()
        plt.savefig(output_dir / 'gradient_norms_moving_average.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: gradient_norms_moving_average.png")

    # Print summary
    print("\n" + "="*70)
    print("GRADIENT NORM STATISTICS (ALL DATA)")
    print("="*70)
    print(f"Total samples: {stats['count']:,}")
    print(f"\nBasic statistics:")
    print(f"  Mean:   {stats['mean']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Std:    {stats['std']:.6f}")
    print(f"  Min:    {stats['min']:.6f}")
    print(f"  Max:    {stats['max']:.6f}")
    print(f"  CV:     {stats['coefficient_of_variation']:.6f}")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:3d}th: {stats['percentiles'][str(p)]:.6f}")

    print("\n" + "="*70)
    print(f"GRADIENT NORM STATISTICS (FILTERED < {max_norm})")
    print("="*70)
    print(f"Filtered samples: {stats_filtered['count']:,} / {stats['count']:,}")
    print(f"Excluded samples: {stats['count'] - stats_filtered['count']:,}")
    print(f"\nBasic statistics:")
    print(f"  Mean:   {stats_filtered['mean']:.6f}")
    print(f"  Median: {stats_filtered['median']:.6f}")
    print(f"  Std:    {stats_filtered['std']:.6f}")
    print(f"  Min:    {stats_filtered['min']:.6f}")
    print(f"  Max:    {stats_filtered['max']:.6f}")

    # Top and bottom samples (from ALL data)
    print("\n" + "="*70)
    print(f"Top 10 highest gradient norms (ALL DATA):")
    print("="*70)
    top_indices = np.argsort(norms)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. Index {idx:6d}: {norms[idx]:.6f}")

    print(f"\nTop 10 lowest gradient norms (ALL DATA):")
    bottom_indices = np.argsort(norms)[:10]
    for rank, idx in enumerate(bottom_indices, 1):
        print(f"  {rank:2d}. Index {idx:6d}: {norms[idx]:.6f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and visualize gradient norms per sample from projection files'
    )

    parser.add_argument(
        'grad_dir',
        type=str,
        help='Directory containing gradient projection files (proj_iter_*.pt)'
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        type=str,
        default="./temp_str/gradient_norm_analysis",
        help='Directory to save visualizations and statistics (default: ./temp_str/gradient_norm_analysis)'
    )
    parser.add_argument(
        '--save_norms',
        action='store_true',
        help='Save gradient norms array as .npy file'
    )
    parser.add_argument(
        '--max_norm',
        type=float,
        default=20.0,
        help='Maximum norm value to display in visualizations (default: 20.0)'
    )

    args = parser.parse_args()

    # Load and process
    print(f"Loading gradient projections from: {args.grad_dir}")
    norms, metadata = load_projection_files(args.grad_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save norms if requested
    if args.save_norms:
        norms_file = output_dir / 'gradient_norms.npy'
        np.save(norms_file, norms)
        print(f"Gradient norms saved to {norms_file}")

    # Visualize
    print("\nCreating visualizations...")
    visualize_gradient_norms(norms, metadata, output_dir, max_norm=args.max_norm)

    print(f"\n✓ All results saved to {output_dir}")


if __name__ == "__main__":
    main()
