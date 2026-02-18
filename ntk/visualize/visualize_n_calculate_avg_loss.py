#!/usr/bin/env python3
"""
Calculate average loss for specific checkpoints from results JSON.

This script loads loss results and computes average loss for each checkpoint.
"""

import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
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


def visualize_loss_dynamics(
    checkpoint_stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Average Loss Dynamics Through Checkpoints"
) -> None:
    """
    Visualize the average loss dynamics through checkpoints.

    Args:
        checkpoint_stats: Dictionary mapping checkpoint name to loss statistics
        output_path: Path to save the figure (if None, displays the plot)
        title: Title for the plot
    """
    # Extract checkpoint names and sort them
    checkpoints = list(checkpoint_stats.keys())

    # Sort checkpoints: base first, then by checkpoint number
    def sort_key(cp):
        if cp == 'base':
            return (0, 0)
        match = re.match(r'cp(\d+)', cp)
        if match:
            return (1, int(match.group(1)))
        return (2, 0)

    checkpoints = sorted(checkpoints, key=sort_key)

    # Extract data for plotting
    avg_losses = [checkpoint_stats[cp]['avg_loss'] for cp in checkpoints]
    std_losses = [checkpoint_stats[cp]['std_loss'] for cp in checkpoints]

    # Create x-axis values (checkpoint steps)
    x_values = []
    x_labels = []
    for cp in checkpoints:
        if cp == 'base':
            x_values.append(0)
            x_labels.append('base')
        else:
            match = re.match(r'cp(\d+)', cp)
            if match:
                step = int(match.group(1))
                x_values.append(step)
                x_labels.append(str(step))

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot average loss with error bars (std)
    ax.errorbar(x_values, avg_losses, yerr=std_losses,
                fmt='o-', capsize=5, capthick=1.5,
                linewidth=2, markersize=8,
                color='#2E86AB', ecolor='#A23B72',
                label='Avg Loss ± Std')

    # Fill the area between mean ± std
    lower = np.array(avg_losses) - np.array(std_losses)
    upper = np.array(avg_losses) + np.array(std_losses)
    ax.fill_between(x_values, lower, upper, alpha=0.2, color='#2E86AB')

    # Customize the plot
    ax.set_xlabel('Checkpoint Step', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set x-ticks
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # Add value annotations
    for i, (x, y) in enumerate(zip(x_values, avg_losses)):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of loss dynamics"
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="ntk/image/avg_loss_dynamics.png",
        help="Path to save the visualization plot (if not specified, displays the plot)"
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default="Average Loss Dynamics Through Checkpoints",
        help="Title for the visualization plot"
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

    # Visualize if requested
    if args.visualize:
        visualize_loss_dynamics(
            checkpoint_stats,
            output_path=args.output_plot,
            title=args.plot_title
        )


if __name__ == "__main__":
    main()
