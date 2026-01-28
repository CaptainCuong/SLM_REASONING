#!/usr/bin/env python3
"""
Calculate average loss for specific checkpoints from results JSON.

This script loads loss results and computes average loss for each checkpoint.
"""

import json
import re
import argparse
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def load_results(results_path: str) -> Dict[str, Any]:
    """Load loss results JSON."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_type_string(type_str: str) -> tuple:
    """
    Parse type string to extract question_id, source, and correctness.

    Args:
        type_str: Type string like "id_4_base_correct" or "id_4_cp555_incorrect"

    Returns:
        Tuple of (question_id, source, correctness)
        source is "base" or checkpoint step number as string
    """
    match = re.match(r'id_(\d+)_(base|cp(\d+))_(correct|incorrect)', type_str)
    if match:
        question_id = int(match.group(1))
        if match.group(2) == 'base':
            source = 'base'
        else:
            source = f"cp{match.group(3)}"
        correctness = match.group(4)
        return question_id, source, correctness
    return None, None, None


def calculate_avg_loss_by_checkpoint(
    results: Dict[str, Any],
    checkpoints: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate average loss for each checkpoint.

    Args:
        results: Results dictionary loaded from JSON
        checkpoints: List of checkpoint names to include (e.g., ['base', 'cp555', 'cp1110'])
                    If None, includes all checkpoints found in the data.

    Returns:
        Dictionary mapping checkpoint name to loss statistics
    """
    summary_by_type = results.get('summary_by_type', {})

    # Group losses by checkpoint
    checkpoint_losses = defaultdict(list)

    for type_str, stats in summary_by_type.items():
        question_id, source, correctness = parse_type_string(type_str)
        if question_id is None:
            continue

        # Filter by checkpoints if specified
        if checkpoints is not None and source not in checkpoints:
            continue

        checkpoint_losses[source].append(stats['avg_loss'])

    # Calculate statistics for each checkpoint
    checkpoint_stats = {}
    for checkpoint, losses in sorted(checkpoint_losses.items(), key=lambda x: (x[0] != 'base', x[0])):
        checkpoint_stats[checkpoint] = {
            'count': len(losses),
            'avg_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses)
        }

    return checkpoint_stats


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average loss for specific checkpoints"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to loss results JSON file"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="cp555,cp1110,cp1665,cp2220",
        help="Comma-separated list of checkpoints to include, or 'all' for all checkpoints (default: cp555,cp1110,cp1665,cp2220)"
    )
    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Include base model in the output"
    )

    args = parser.parse_args()

    # Parse checkpoints
    if args.checkpoints.lower() == 'all':
        checkpoints = None  # None means include all checkpoints
    else:
        checkpoints = [cp.strip() for cp in args.checkpoints.split(',')]
        if args.include_base:
            checkpoints = ['base'] + checkpoints

    # Load results
    print(f"Loading results from {args.results_path}...")
    results = load_results(args.results_path)

    print(f"Model: {results.get('model_path', 'N/A')}")
    print(f"Pool: {results.get('pool_path', 'N/A')}")
    print(f"Total samples: {results.get('total_samples', 'N/A')}")

    # Calculate average loss by checkpoint
    checkpoint_stats = calculate_avg_loss_by_checkpoint(results, checkpoints)

    # Print results
    print("\n" + "=" * 60)
    print("Average Loss by Checkpoint")
    print("=" * 60)

    for checkpoint, stats in checkpoint_stats.items():
        print(f"\n{checkpoint}:")
        print(f"  Count:    {stats['count']}")
        print(f"  Avg Loss: {stats['avg_loss']:.6f}")
        print(f"  Std Loss: {stats['std_loss']:.6f}")
        print(f"  Min Loss: {stats['min_loss']:.6f}")
        print(f"  Max Loss: {stats['max_loss']:.6f}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Checkpoint':<12} {'Count':<8} {'Avg Loss':<12} {'Std Loss':<12}")
    print("-" * 44)
    for checkpoint, stats in checkpoint_stats.items():
        print(f"{checkpoint:<12} {stats['count']:<8} {stats['avg_loss']:<12.6f} {stats['std_loss']:<12.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
