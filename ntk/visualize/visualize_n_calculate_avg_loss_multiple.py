#!/usr/bin/env python3
"""
Calculate and visualize average loss dynamics from multiple results JSON files.

Usage:
    python ntk/visualize/visualize_n_calculate_avg_loss_multiple.py \
        --results_paths ntk/results/train_set_results/losses_math12k_high_1.5B.json \
                        ntk/results/train_set_results/losses_math12k_high_3B.json \
                        ntk/results/train_set_results/losses_math12k_high_7B.json \
        --labels "1.5B" "3B" "7B" \
        --visualize --output_plot ntk/image/avg_loss_dynamics_multiple.png
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

    Returns:
        Tuple of (question_id, source, correctness)
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


def sort_checkpoint_key(cp):
    if cp == 'base':
        return (0, 0)
    match = re.match(r'cp(\d+)', cp)
    if match:
        return (1, int(match.group(1)))
    return (2, 0)


def calculate_avg_loss_by_checkpoint(
    results: Dict[str, Any],
    checkpoints: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """Calculate average loss for each checkpoint."""
    summary_by_type = results.get('summary_by_type', {})
    checkpoint_losses = defaultdict(list)

    for type_str, stats in summary_by_type.items():
        question_id, source, correctness = parse_type_string(type_str)
        if question_id is None:
            continue
        if checkpoints is not None and source not in checkpoints:
            continue
        checkpoint_losses[source].append(stats['avg_loss'])

    checkpoint_stats = {}
    for checkpoint, losses in sorted(checkpoint_losses.items(), key=lambda x: sort_checkpoint_key(x[0])):
        checkpoint_stats[checkpoint] = {
            'count': len(losses),
            'avg_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses)
        }

    return checkpoint_stats


def visualize_loss_dynamics_multiple(
    all_checkpoint_stats: List[Dict[str, Dict[str, float]]],
    labels: List[str],
    output_path: Optional[str] = None,
    title: str = "Average Loss Dynamics Through Checkpoints"
) -> None:
    """
    Visualize average loss dynamics for multiple result files on the same plot.

    Args:
        all_checkpoint_stats: List of checkpoint_stats dicts (one per file)
        labels: List of labels for each result file
        output_path: Path to save the figure (if None, displays the plot)
        title: Title for the plot
    """
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#9C27B0', '#FF5722']
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (checkpoint_stats, label) in enumerate(zip(all_checkpoint_stats, labels)):
        checkpoints = sorted(checkpoint_stats.keys(), key=sort_checkpoint_key)

        x_values = []
        x_labels_list = []
        avg_losses = []
        std_losses = []

        for cp in checkpoints:
            if cp == 'base':
                x_values.append(0)
                x_labels_list.append('base')
            else:
                match = re.match(r'cp(\d+)', cp)
                if match:
                    step = int(match.group(1))
                    x_values.append(step)
                    x_labels_list.append(str(step))
            avg_losses.append(checkpoint_stats[cp]['avg_loss'])
            std_losses.append(checkpoint_stats[cp]['std_loss'])

        color = colors[idx % len(colors)]

        ax.errorbar(x_values, avg_losses, yerr=std_losses,
                    fmt='o-', capsize=5, capthick=1.5,
                    linewidth=2, markersize=8,
                    color=color, ecolor=color,
                    label=label, alpha=0.85)

        lower = np.array(avg_losses) - np.array(std_losses)
        upper = np.array(avg_losses) + np.array(std_losses)
        ax.fill_between(x_values, lower, upper, alpha=0.1, color=color)

        for x, y in zip(x_values, avg_losses):
            ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                       xytext=(0, 12 + idx * 10), ha='center', fontsize=7, color=color)

    # Use x-ticks from the first result file (they should share the same checkpoints)
    first_stats = all_checkpoint_stats[0]
    checkpoints = sorted(first_stats.keys(), key=sort_checkpoint_key)
    x_ticks = []
    x_tick_labels = []
    for cp in checkpoints:
        if cp == 'base':
            x_ticks.append(0)
            x_tick_labels.append('base')
        else:
            match = re.match(r'cp(\d+)', cp)
            if match:
                step = int(match.group(1))
                x_ticks.append(step)
                x_tick_labels.append(str(step))

    ax.set_xlabel('Checkpoint Step', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and visualize average loss from multiple results JSON files"
    )
    parser.add_argument(
        "--results_paths",
        type=str,
        nargs='+',
        required=True,
        help="Paths to loss results JSON files"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        help="Labels for each result file (default: derived from file names)"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="all",
        help="Comma-separated list of checkpoints to include, or 'all' (default: all)"
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
        default="ntk/image/avg_loss_dynamics_multiple.png",
        help="Path to save the visualization plot"
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
        checkpoints = None
    else:
        checkpoints = [cp.strip() for cp in args.checkpoints.split(',')]
        if args.include_base:
            checkpoints = ['base'] + checkpoints

    # Generate labels from filenames if not provided
    if args.labels is None:
        labels = []
        for p in args.results_paths:
            name = p.rsplit('/', 1)[-1].replace('.json', '').replace('losses_', '')
            labels.append(name)
    else:
        labels = args.labels

    # Process each results file
    all_checkpoint_stats = []
    for path, label in zip(args.results_paths, labels):
        print(f"\n{'=' * 60}")
        print(f"Loading: {path}")
        results = load_results(path)
        print(f"  Model: {results.get('model_path', 'N/A')}")
        print(f"  Pool:  {results.get('pool_path', 'N/A')}")
        print(f"  Total: {results.get('total_samples', 'N/A')} samples")

        checkpoint_stats = calculate_avg_loss_by_checkpoint(results, checkpoints)
        all_checkpoint_stats.append(checkpoint_stats)

        # Print table for this file
        print(f"\n  {'Checkpoint':<12} {'Count':<8} {'Avg Loss':<12} {'Std Loss':<12}")
        print(f"  {'-' * 44}")
        for checkpoint, stats in checkpoint_stats.items():
            print(f"  {checkpoint:<12} {stats['count']:<8} {stats['avg_loss']:<12.6f} {stats['std_loss']:<12.6f}")

    print(f"\n{'=' * 60}")

    # Visualize
    if args.visualize:
        visualize_loss_dynamics_multiple(
            all_checkpoint_stats,
            labels,
            output_path=args.output_plot,
            title=args.plot_title
        )


if __name__ == "__main__":
    main()
