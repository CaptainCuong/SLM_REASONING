#!/usr/bin/env python3
"""
Calculate dot product similarity between gradient projections from two folders.
"""

import torch
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
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
    """Get all projection .pt files from a folder."""
    folder = Path(folder_path)
    proj_files = sorted(folder.glob("proj_iter_*.pt"))
    return proj_files


def flatten_gradients(proj_data):
    """
    Flatten all gradient tensors from a projection file into a single vector.

    Args:
        proj_data: Dictionary of layer_name -> tensor or nested structure

    Returns:
        flattened: 1D tensor with all gradients concatenated
    """
    all_grads = []

    if isinstance(proj_data, dict):
        # Sort keys for consistent ordering
        for key in sorted(proj_data.keys()):
            value = proj_data[key]

            if isinstance(value, torch.Tensor):
                all_grads.append(value.flatten())
            elif isinstance(value, dict):
                # Nested structure
                for sub_key in sorted(value.keys()):
                    sub_val = value[sub_key]
                    if isinstance(sub_val, torch.Tensor):
                        all_grads.append(sub_val.flatten())
    elif isinstance(proj_data, torch.Tensor):
        all_grads.append(proj_data.flatten())

    if len(all_grads) == 0:
        return torch.tensor([])

    return torch.cat(all_grads)


def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1, vec2: 1D tensors

    Returns:
        similarity: scalar in [-1, 1]
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return similarity.item()


def compute_dot_product(vec1, vec2):
    """Compute raw dot product between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    return torch.dot(vec1, vec2).item()


def compute_mean_projection(folder_path):
    """
    Compute the mean projection across all files in a folder.
    Averages across both files AND batch dimension, resulting in shape with first dim = 1.

    Args:
        folder_path: Path to folder containing projection files

    Returns:
        mean_proj: Dictionary with same structure as input projections,
                  containing mean values across all files and batches (first dim = 1)
    """
    print(f"\nComputing mean projection for: {folder_path}")
    print("="*80)

    proj_files = get_projection_files(folder_path)
    print(f"Found {len(proj_files)} projection files")

    if len(proj_files) == 0:
        raise ValueError(f"No projection files found in {folder_path}")

    # Initialize accumulator
    mean_proj = None
    total_examples = 0

    # Accumulate all projections across files AND batches
    for file_path in tqdm(proj_files, desc="Loading projections"):
        proj = load_projection_file(file_path)
        if proj is None:
            raise ValueError(f"Failed to load projection from {file_path}")
        
        # Get batch size from first tensor
        batch_size = proj["proj"].shape[0]

        if mean_proj is None:
            # Initialize with zeros matching the structure but with first dim = 1
            mean_proj = {}
            new_shape = (1,) + proj["proj"].shape[1:]
            mean_proj["proj"] = torch.zeros(new_shape, dtype=torch.float32)

        # Accumulate sum across batch dimension
        mean_proj["proj"] += proj["proj"].float().sum(dim=0, keepdim=True)
        
        # Sum across batch dimension (dim 0) for this file
        total_examples += batch_size

    # Divide by total number of examples to get mean
    mean_proj["proj"] /= total_examples

    print(f"✓ Mean projection computed from {total_examples} examples across {len(proj_files)} files")
    print(f"✓ Result has first dimension = 1 (averaged across all batches)")
    
    return mean_proj


def compute_structured_dot_product(proj1, proj2_mean):
    """
    Compute dot product between a batch of projections and a mean projection.
    Handles batch dimension in proj1 and broadcasts with proj2_mean (first dim = 1).

    Args:
        proj1: Projection dictionary with batch dimension (first dim = batch_size)
        proj2_mean: Mean projection dictionary (first dim = 1)

    Returns:
        dot_products: List of dot products, one per example in batch
    """
    # Get batch size from first tensor
    batch_size = None
    batch_size = proj1["proj"].shape[0]
    if batch_size is None:
        raise ValueError("Could not determine batch size from proj1")

    # Initialize results
    dot_products = [0.0] * batch_size

    val1 = proj1["proj"]
    val2 = proj2_mean["proj"]

    # Element-wise multiply: (batch_size, ...) * (1, ...) -> (batch_size, ...)
    # Sum over all dims except batch: -> (batch_size,)
    layer_dot_batch = torch.sum(val1 * val2, dim=tuple(range(1, len(val1.shape))))

    # Add to total
    for i in range(batch_size):
        dot_products[i] += layer_dot_batch[i].item()

    return dot_products


def compute_structured_cosine_similarity(proj1, proj2_mean):
    """
    Compute cosine similarity between a batch of projections and a mean projection.
    Handles batch dimension in proj1 and broadcasts with proj2_mean (first dim = 1).

    Args:
        proj1: Projection dictionary with batch dimension (first dim = batch_size)
        proj2_mean: Mean projection dictionary (first dim = 1)

    Returns:
        cosine_similarities: List of cosine similarities, one per example in batch
    """
    # Get batch size from first tensor
    batch_size = proj1["proj"].shape[0]

    val1 = proj1["proj"]
    val2 = proj2_mean["proj"]

    # Flatten last dimensions for cosine similarity calculation
    # Reshape to (batch_size, -1) and (1, -1)
    val1_flat = val1.reshape(batch_size, -1)
    val2_flat = val2.reshape(1, -1)

    # Compute cosine similarity: (batch_size, d) vs (1, d)
    # F.cosine_similarity expects both inputs to have the same first dimension
    # so we'll manually compute it
    cosine_sims = F.cosine_similarity(val1_flat, val2_flat.expand(batch_size, -1), dim=1)

    # Convert to list
    cosine_similarities = cosine_sims.tolist()

    return cosine_similarities


def calculate_dots_with_mean(folder1, folder2, save_path=None):
    """
    Calculate dot product between each example in folder1 and mean of folder2.
    Handles batch dimension: each file in folder1 contains multiple examples.
    Does not flatten - maintains projection structure.

    Args:
        folder1: Path to first folder (projections with batch dimension)
        folder2: Path to second folder (compute mean with first dim = 1)
        save_path: Optional path to save results

    Returns:
        results: Dictionary with dot products and statistics for each individual example
    """
    from collections import defaultdict

    print("="*80)
    print("Calculating dot products with mean projection (no flattening)")
    print("="*80)
    print(f"\nFolder 1 (individual examples): {folder1}")
    print(f"Folder 2 (mean): {folder2}")

    # Get projection files from folder1
    files1 = get_projection_files(folder1)
    print(f"\nFound {len(files1)} files in folder 1")

    if len(files1) == 0:
        raise ValueError("No projection files found in folder 1")

    # Compute mean projection from folder2 (first dim = 1)
    mean_proj2 = compute_mean_projection(folder2)

    # Calculate dot products
    print("\n" + "="*80)
    print("Calculating dot products with mean...")
    print("="*80)

    all_dot_products = []

    for file1 in tqdm(files1, desc="Processing folder 1"):
        proj1 = load_projection_file(file1)
        if proj1 is None:
            continue

        # Compute dot products for all examples in batch
        batch_dot_products = compute_structured_dot_product(proj1, mean_proj2)

        # batch_dot_products is a list of length batch_size
        # Add to results
        all_dot_products.extend(batch_dot_products)

    # Convert to numpy array
    all_dot_products = np.array(all_dot_products)

    # Save to file if requested
    if save_path:
        np.save(save_path, all_dot_products)
        print(f"\n✓ Dot products saved to: {save_path}")

    return all_dot_products


def calculate_cosine_with_mean(folder1, folder2, save_path=None):
    """
    Calculate cosine similarity between each example in folder1 and mean of folder2.
    Handles batch dimension: each file in folder1 contains multiple examples.
    Does not flatten - maintains projection structure.

    Args:
        folder1: Path to first folder (projections with batch dimension)
        folder2: Path to second folder (compute mean with first dim = 1)
        save_path: Optional path to save results

    Returns:
        results: Array of cosine similarities for each individual example
    """
    print("="*80)
    print("Calculating cosine similarities with mean projection (no flattening)")
    print("="*80)
    print(f"\nFolder 1 (individual examples): {folder1}")
    print(f"Folder 2 (mean): {folder2}")

    # Get projection files from folder1
    files1 = get_projection_files(folder1)
    print(f"\nFound {len(files1)} files in folder 1")

    if len(files1) == 0:
        raise ValueError("No projection files found in folder 1")

    # Compute mean projection from folder2 (first dim = 1)
    mean_proj2 = compute_mean_projection(folder2)

    # Calculate cosine similarities
    print("\n" + "="*80)
    print("Calculating cosine similarities with mean...")
    print("="*80)

    all_cosine_sims = []

    for file1 in tqdm(files1, desc="Processing folder 1"):
        proj1 = load_projection_file(file1)
        if proj1 is None:
            continue

        # Compute cosine similarities for all examples in batch
        batch_cosine_sims = compute_structured_cosine_similarity(proj1, mean_proj2)

        # batch_cosine_sims is a list of length batch_size
        # Add to results
        all_cosine_sims.extend(batch_cosine_sims)

    # Convert to numpy array
    all_cosine_sims = np.array(all_cosine_sims)

    # Save to file if requested
    if save_path:
        np.save(save_path, all_cosine_sims)
        print(f"\n✓ Cosine similarities saved to: {save_path}")

    return all_cosine_sims


def print_mean_dot_results(dot_products):
    """Print basic statistics for dot products array."""
    print("\n" + "="*80)
    print("DOT PRODUCT WITH MEAN RESULTS")
    print("="*80)

    print(f"\nTotal number of examples: {len(dot_products)}")

    print("\n" + "-"*80)
    print("Dot Product Statistics:")
    print("-"*80)
    print(f"  Mean:   {np.mean(dot_products):.6e}")
    print(f"  Median: {np.median(dot_products):.6e}")
    print(f"  Std:    {np.std(dot_products):.6e}")
    print(f"  Min:    {np.min(dot_products):.6e}")
    print(f"  Max:    {np.max(dot_products):.6e}")

    print("\n" + "="*80)

def print_similarity_statistics(similarity_matrix, file_info):
    """Print statistics about the similarity matrix."""
    print("\n" + "="*80)
    print("SIMILARITY STATISTICS")
    print("="*80)

    print(f"\nMetric: {file_info['metric']}")
    print(f"Matrix shape: {similarity_matrix.shape}")
    print(f"  ({len(file_info['files1'])} files from folder 1 × {len(file_info['files2'])} files from folder 2)")

    print(f"\nStatistics:")
    print(f"  Mean:   {np.nanmean(similarity_matrix):.6f}")
    print(f"  Median: {np.nanmedian(similarity_matrix):.6f}")
    print(f"  Std:    {np.nanstd(similarity_matrix):.6f}")
    print(f"  Min:    {np.nanmin(similarity_matrix):.6f}")
    print(f"  Max:    {np.nanmax(similarity_matrix):.6f}")

    # Find most similar pairs
    print("\nTop 10 most similar pairs:")
    flat_indices = np.argsort(similarity_matrix.ravel())[::-1][:10]
    for rank, idx in enumerate(flat_indices, 1):
        i, j = np.unravel_index(idx, similarity_matrix.shape)
        sim = similarity_matrix[i, j]
        if not np.isnan(sim):
            print(f"  {rank}. {file_info['files1'][i]} ↔ {file_info['files2'][j]}: {sim:.6f}")

    # Diagonal similarities (if square matrix)
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        print("\nDiagonal (same index) similarities:")
        diag_sims = np.diag(similarity_matrix)
        print(f"  Mean: {np.nanmean(diag_sims):.6f}")
        print(f"  Min:  {np.nanmin(diag_sims):.6f}")
        print(f"  Max:  {np.nanmax(diag_sims):.6f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate gradient similarity between two projection folders'
    )

    parser.add_argument(
        '--folder1',
        type=str,
        required=True,
        help='Path to first projection folder'
    )
    parser.add_argument(
        '--folder2',
        type=str,
        required=True,
        help='Path to second projection folder'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='dot',
        choices=['cosine', 'dot'],
        help='Similarity metric (default: dot)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON file'
    )

    args = parser.parse_args()

    # Calculate dot products with mean (no flattening)
    results = calculate_dots_with_mean(
        args.folder1,
        args.folder2,
        save_path=args.output
    )
    print_mean_dot_results(results)


if __name__ == "__main__":
    main()
