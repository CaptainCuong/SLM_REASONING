#!/usr/bin/env python3
"""
Visualize average log-likelihood dynamics across training checkpoints.

This script loads the checkpoint summary JSON and creates plots showing
how average log-likelihood evolves for each sample type.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_summary(summary_path: str):
    """Load checkpoint summary JSON."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_avg_log_likelihood(summary: dict, output_path: str = None, show: bool = True):
    """
    Plot average log-likelihood for all sample types.

    Args:
        summary: Checkpoint summary dictionary
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    steps = summary['steps']
    results_by_type = summary['results_by_type']

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot each sample type
    for sample_type, metrics in results_by_type.items():
        avg_llh = metrics['avg_log_likelihood']

        # Filter out None values
        valid_indices = [i for i, val in enumerate(avg_llh) if val is not None]
        valid_steps = [steps[i] for i in valid_indices]
        valid_llh = [avg_llh[i] for i in valid_indices]

        if valid_llh:
            plt.plot(valid_steps, valid_llh, marker='o', label=sample_type, linewidth=2)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Average Log-Likelihood', fontsize=12)
    plt.title('Average Log-Likelihood Dynamics Across Training', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_perplexity(summary: dict, output_path: str = None, show: bool = True):
    """
    Plot average perplexity for all sample types.

    Args:
        summary: Checkpoint summary dictionary
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    steps = summary['steps']
    results_by_type = summary['results_by_type']

    plt.figure(figsize=(12, 8))

    for sample_type, metrics in results_by_type.items():
        avg_ppl = metrics['avg_perplexity']

        # Filter out None values
        valid_indices = [i for i, val in enumerate(avg_ppl) if val is not None]
        valid_steps = [steps[i] for i in valid_indices]
        valid_ppl = [avg_ppl[i] for i in valid_indices]

        if valid_ppl:
            plt.plot(valid_steps, valid_ppl, marker='o', label=sample_type, linewidth=2)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Average Perplexity', fontsize=12)
    plt.title('Average Perplexity Dynamics Across Training', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def create_summary_stats(summary: dict):
    """
    Print summary statistics for log-likelihood dynamics.

    Args:
        summary: Checkpoint summary dictionary
    """
    steps = summary['steps']
    results_by_type = summary['results_by_type']

    print("\n" + "="*80)
    print("Log-Likelihood Dynamics Summary")
    print("="*80)

    for sample_type, metrics in results_by_type.items():
        avg_llh = metrics['avg_log_likelihood']

        # Filter out None values
        valid_llh = [x for x in avg_llh if x is not None]

        if valid_llh:
            initial = valid_llh[0]
            final = valid_llh[-1]
            improvement = final - initial
            percent_change = (improvement / abs(initial)) * 100 if initial != 0 else 0

            print(f"\n{sample_type}:")
            print(f"  Initial LLH: {initial:.4f}")
            print(f"  Final LLH:   {final:.4f}")
            print(f"  Change:      {improvement:+.4f} ({percent_change:+.2f}%)")
            print(f"  Min LLH:     {min(valid_llh):.4f} (step {steps[avg_llh.index(min(valid_llh))]})")
            print(f"  Max LLH:     {max(valid_llh):.4f} (step {steps[avg_llh.index(max(valid_llh))]})")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize average log-likelihood dynamics from checkpoint summary"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="prob_tracking/results/Qwen_Math_high_all_checkpoints_summary.json",
        help="Path to checkpoint summary JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prob_tracking/image",
        help="Directory to save plots (default: prob_tracking/image)"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plots (only save)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=['all', 'llh', 'perplexity'],
        default='all',
        help="Type of plot to generate (default: all)"
    )

    args = parser.parse_args()

    # Load summary
    print(f"Loading summary from {args.summary_path}...")
    summary = load_summary(args.summary_path)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Print summary statistics
    create_summary_stats(summary)

    show = not args.no_show

    # Generate plots
    if args.plot_type in ['all', 'llh']:
        print("\nGenerating log-likelihood plot...")
        plot_avg_log_likelihood(
            summary,
            output_path=f"{args.output_dir}/avg_llh.png",
            show=show
        )

    if args.plot_type in ['all', 'perplexity']:
        print("\nGenerating perplexity plot...")
        plot_perplexity(
            summary,
            output_path=f"{args.output_dir}/avg_perplexity.png",
            show=show
        )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
