"""
Calculate pairwise cosine similarity between gradient projections from two folders.

This script computes cosine similarity between each sample in folder1 and each sample
in folder2, then creates visualizations including heatmaps, histograms, and statistics.
"""

import torch
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


def load_projection_file(file_path):
    """Load a single projection file."""
    try:
        data = torch.load(file_path, map_location='cpu')
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_projection_files(folder_path):
    """Get all projection .pt files from a folder, sorted by iteration."""
    folder = Path(folder_path)
    proj_files = sorted(folder.glob("proj_iter_*.pt"))
    return proj_files


def load_all_projections(folder_path):
    """
    Load all projection files from a folder and concatenate into a single tensor.

    Returns:
        projections: Tensor of shape [N, D] where N is total samples, D is projection dimension
        file_info: List of tuples (filename, batch_size, start_idx, end_idx)
    """
    files = get_projection_files(folder_path)
    if not files:
        raise ValueError(f"No projection files found in {folder_path}")

    all_projections = []
    file_info = []
    current_idx = 0

    print(f"Loading projections from {folder_path}...")
    for file_path in tqdm(files):
        data = load_projection_file(file_path)
        if data is None:
            continue

        # Extract projection tensor
        proj = data['proj']  # Shape: [B, D]
        batch_size = proj.shape[0]

        all_projections.append(proj)
        file_info.append({
            'filename': file_path.name,
            'iter': data.get('iter', -1),
            'batch_size': batch_size,
            'start_idx': current_idx,
            'end_idx': current_idx + batch_size
        })

        current_idx += batch_size

    # Concatenate all projections
    projections = torch.cat(all_projections, dim=0)  # [N, D]

    print(f"Loaded {projections.shape[0]} samples with dimension {projections.shape[1]}")

    return projections, file_info


def compute_pairwise_cosine_similarity(proj1, proj2):
    """
    Compute pairwise cosine similarity between all samples in proj1 and proj2.

    Args:
        proj1: Tensor of shape [N1, D]
        proj2: Tensor of shape [N2, D]

    Returns:
        similarity_matrix: Tensor of shape [N1, N2] with cosine similarities
    """
    # Normalize to unit vectors
    proj1_norm = F.normalize(proj1, p=2, dim=1)  # [N1, D]
    proj2_norm = F.normalize(proj2, p=2, dim=1)  # [N2, D]

    # Compute cosine similarity: proj1_norm @ proj2_norm.T
    similarity_matrix = torch.mm(proj1_norm, proj2_norm.T)  # [N1, N2]

    return similarity_matrix


def visualize_similarity_matrix(similarity_matrix, output_dir, max_display=100):
    """
    Create visualizations for the similarity matrix.

    Args:
        similarity_matrix: Numpy array of shape [N1, N2]
        output_dir: Directory to save visualizations
        max_display: Maximum number of samples to display in heatmap
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N1, N2 = similarity_matrix.shape

    # 1. Full heatmap (downsampled if too large)
    print("Creating heatmap...")
    plt.figure(figsize=(12, 10))

    # Downsample if necessary
    if N1 > max_display or N2 > max_display:
        step1 = max(1, N1 // max_display)
        step2 = max(1, N2 // max_display)
        display_matrix = similarity_matrix[::step1, ::step2]
        title = f'Cosine Similarity Heatmap (sampled {display_matrix.shape[0]}x{display_matrix.shape[1]} from {N1}x{N2})'
    else:
        display_matrix = similarity_matrix
        title = f'Cosine Similarity Heatmap ({N1}x{N2})'

    sns.heatmap(display_matrix, cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, cbar_kws={'label': 'Cosine Similarity'})
    plt.title(title)
    plt.xlabel('Samples in Folder 2')
    plt.ylabel('Samples in Folder 1')
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_dir / 'similarity_heatmap.png'}")

    # 2. Histogram of all similarity values
    print("Creating histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_matrix.flatten(), bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Pairwise Cosine Similarities\n({N1} × {N2} = {N1*N2:,} pairs)')
    plt.axvline(similarity_matrix.mean(), color='red', linestyle='--',
                label=f'Mean: {similarity_matrix.mean():.4f}')
    plt.axvline(np.median(similarity_matrix), color='blue', linestyle='--',
                label=f'Median: {np.median(similarity_matrix):.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_dir / 'similarity_histogram.png'}")

    # 3. Max similarity per sample in folder1
    print("Creating max similarity plot...")
    max_sim_per_sample1 = similarity_matrix.max(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Line plot
    axes[0].plot(max_sim_per_sample1, linewidth=0.5)
    axes[0].set_xlabel('Sample Index (Folder 1)')
    axes[0].set_ylabel('Max Cosine Similarity')
    axes[0].set_title('Maximum Cosine Similarity for Each Sample in Folder 1')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(max_sim_per_sample1.mean(), color='red', linestyle='--',
                    label=f'Mean: {max_sim_per_sample1.mean():.4f}')
    axes[0].legend()

    # Histogram
    axes[1].hist(max_sim_per_sample1, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Max Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Maximum Similarities')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'max_similarity_per_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved max similarity plot to {output_dir / 'max_similarity_per_sample.png'}")

    # 4. Mean similarity per sample in folder1
    print("Creating mean similarity plot...")
    mean_sim_per_sample1 = similarity_matrix.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Line plot
    axes[0].plot(mean_sim_per_sample1, linewidth=0.5)
    axes[0].set_xlabel('Sample Index (Folder 1)')
    axes[0].set_ylabel('Mean Cosine Similarity')
    axes[0].set_title('Mean Cosine Similarity for Each Sample in Folder 1')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(mean_sim_per_sample1.mean(), color='red', linestyle='--',
                    label=f'Overall Mean: {mean_sim_per_sample1.mean():.4f}')
    axes[0].legend()

    # Histogram
    axes[1].hist(mean_sim_per_sample1, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Mean Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Mean Similarities')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'mean_similarity_per_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mean similarity plot to {output_dir / 'mean_similarity_per_sample.png'}")

    # 5. Diagonal similarity (if same number of samples)
    if N1 == N2:
        print("Creating diagonal similarity plot...")
        diagonal_sim = np.diag(similarity_matrix)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Line plot
        axes[0].plot(diagonal_sim, linewidth=0.5)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].set_title('Diagonal Cosine Similarity (Same Index Pairs)')
        axes[0].grid(alpha=0.3)
        axes[0].axhline(diagonal_sim.mean(), color='red', linestyle='--',
                        label=f'Mean: {diagonal_sim.mean():.4f}')
        axes[0].legend()

        # Histogram
        axes[1].hist(diagonal_sim, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Diagonal Cosine Similarity')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Diagonal Similarities')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'diagonal_similarity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved diagonal similarity plot to {output_dir / 'diagonal_similarity.png'}")


def compute_and_save_statistics(similarity_matrix, output_dir):
    """Compute and save statistics about the similarity matrix."""
    output_dir = Path(output_dir)

    stats = {
        'shape': similarity_matrix.shape,
        'total_pairs': int(similarity_matrix.size),
        'mean': float(similarity_matrix.mean()),
        'std': float(similarity_matrix.std()),
        'median': float(np.median(similarity_matrix)),
        'min': float(similarity_matrix.min()),
        'max': float(similarity_matrix.max()),
        'percentiles': {
            '5': float(np.percentile(similarity_matrix, 5)),
            '25': float(np.percentile(similarity_matrix, 25)),
            '75': float(np.percentile(similarity_matrix, 75)),
            '95': float(np.percentile(similarity_matrix, 95)),
        },
        'max_per_sample1': {
            'mean': float(similarity_matrix.max(axis=1).mean()),
            'std': float(similarity_matrix.max(axis=1).std()),
            'min': float(similarity_matrix.max(axis=1).min()),
            'max': float(similarity_matrix.max(axis=1).max()),
        },
        'mean_per_sample1': {
            'mean': float(similarity_matrix.mean(axis=1).mean()),
            'std': float(similarity_matrix.mean(axis=1).std()),
            'min': float(similarity_matrix.mean(axis=1).min()),
            'max': float(similarity_matrix.mean(axis=1).max()),
        }
    }

    # Add diagonal stats if square matrix
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        diagonal = np.diag(similarity_matrix)
        stats['diagonal'] = {
            'mean': float(diagonal.mean()),
            'std': float(diagonal.std()),
            'min': float(diagonal.min()),
            'max': float(diagonal.max()),
        }

    # Save to JSON
    stats_file = output_dir / 'similarity_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to {stats_file}")

    # Print summary
    print("\n" + "="*60)
    print("SIMILARITY STATISTICS SUMMARY")
    print("="*60)
    print(f"Matrix shape: {stats['shape'][0]} × {stats['shape'][1]}")
    print(f"Total pairs: {stats['total_pairs']:,}")
    print(f"\nOverall similarity:")
    print(f"  Mean:   {stats['mean']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Std:    {stats['std']:.6f}")
    print(f"  Min:    {stats['min']:.6f}")
    print(f"  Max:    {stats['max']:.6f}")
    print(f"\nPercentiles:")
    print(f"  5th:  {stats['percentiles']['5']:.6f}")
    print(f"  25th: {stats['percentiles']['25']:.6f}")
    print(f"  75th: {stats['percentiles']['75']:.6f}")
    print(f"  95th: {stats['percentiles']['95']:.6f}")
    print(f"\nMax similarity per sample (Folder 1):")
    print(f"  Mean: {stats['max_per_sample1']['mean']:.6f}")
    print(f"  Std:  {stats['max_per_sample1']['std']:.6f}")
    print(f"  Range: [{stats['max_per_sample1']['min']:.6f}, {stats['max_per_sample1']['max']:.6f}]")

    if 'diagonal' in stats:
        print(f"\nDiagonal similarity (same index pairs):")
        print(f"  Mean: {stats['diagonal']['mean']:.6f}")
        print(f"  Std:  {stats['diagonal']['std']:.6f}")
        print(f"  Range: [{stats['diagonal']['min']:.6f}, {stats['diagonal']['max']:.6f}]")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pairwise cosine similarity between gradient projections"
    )
    parser.add_argument(
        '--folder1',
        type=str,
        required=True,
        help='Path to first gradient projection folder'
    )
    parser.add_argument(
        '--folder2',
        type=str,
        required=True,
        help='Path to second gradient projection folder'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./temp_str/cosine_similarity_analysis',
        help='Directory to save outputs (default: ./temp_str/cosine_similarity_analysis)'
    )
    parser.add_argument(
        '--save_matrix',
        action='store_true',
        help='Save the full similarity matrix as .npy file (may be large)'
    )
    parser.add_argument(
        '--max_display',
        type=int,
        default=100,
        help='Maximum number of samples to display in heatmap (default: 100)'
    )

    args = parser.parse_args()

    # Load projections
    print("Loading projections from folder 1...")
    proj1, info1 = load_all_projections(args.folder1)

    print("\nLoading projections from folder 2...")
    proj2, info2 = load_all_projections(args.folder2)

    # Verify dimensions match
    if proj1.shape[1] != proj2.shape[1]:
        raise ValueError(
            f"Projection dimensions don't match: {proj1.shape[1]} vs {proj2.shape[1]}"
        )

    # Compute pairwise cosine similarity
    print(f"\nComputing pairwise cosine similarity...")
    print(f"This will create a {proj1.shape[0]} × {proj2.shape[0]} matrix...")

    similarity_matrix = compute_pairwise_cosine_similarity(proj1, proj2)
    similarity_np = similarity_matrix.float().cpu().numpy()

    print(f"Similarity matrix computed: shape {similarity_np.shape}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        'folder1': str(args.folder1),
        'folder2': str(args.folder2),
        'n_samples_folder1': proj1.shape[0],
        'n_samples_folder2': proj2.shape[0],
        'projection_dim': proj1.shape[1],
        'folder1_files': info1,
        'folder2_files': info2,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_dir / 'metadata.json'}")

    # Compute and save statistics
    compute_and_save_statistics(similarity_np, output_dir)

    # Create visualizations
    visualize_similarity_matrix(similarity_np, output_dir, max_display=args.max_display)

    # Save similarity matrix if requested
    if args.save_matrix:
        matrix_file = output_dir / 'similarity_matrix.npy'
        np.save(matrix_file, similarity_np)
        print(f"\nSimilarity matrix saved to {matrix_file}")
        print(f"File size: {matrix_file.stat().st_size / 1024**2:.2f} MB")

    print(f"\n✓ All results saved to {output_dir}")


if __name__ == '__main__':
    main()
