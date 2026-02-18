#!/usr/bin/env python3
"""
Visualize the avg_loss distribution of samples matching given patterns.

Patterns use fnmatch-style wildcards on the type keys in summary_by_type.
For example:
    "id_11_base_*"      -> matches id_11_base_correct, id_11_base_incorrect
    "id_*_base_*"       -> all base model entries
    "id_*_cp1110_*"     -> all entries at checkpoint 1110
    "id_11_*"           -> all entries for question 11 across checkpoints

Usage:
    # Single pattern, single file
    python ntk/visualize/visualize_loss_distribution.py \
        --results_path ntk/results/train_set_results/losses_math12k_high_1.5B.json \
        --patterns "id_11_base_*"

    # Multiple patterns overlaid (compare distributions)
    python ntk/visualize/visualize_loss_distribution.py \
        --results_path ntk/results/train_set_results/losses_math12k_high_1.5B.json \
        --patterns "id_*_base_incorrect" "id_*_cp1110_incorrect" "id_*_cp22000_incorrect" \
        --labels "base" "cp1110" "cp22000"

    # Multiple files, one pattern each
    python ntk/visualize/visualize_loss_distribution.py \
        --results_path ntk/results/train_set_results/losses_math12k_high_1.5B.json \
                       ntk/results/train_set_results/losses_math12k_high_3B.json \
        --patterns "id_*_base_incorrect" \
        --labels "1.5B" "3B" \
        --plot_type box

    # Use --plot_type to choose: histogram (default), box, violin
"""

import json
import argparse
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


def load_results(results_path: str) -> Dict[str, Any]:
    """Load loss results JSON."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_losses_by_pattern(
    results: Dict[str, Any],
    pattern: str
) -> List[float]:
    """
    Collect avg_loss values from summary_by_type entries matching the pattern.

    Args:
        results: Loaded JSON results dict.
        pattern: fnmatch-style pattern (e.g. "id_11_base_*", "id_*_cp1110_*").

    Returns:
        List of avg_loss values for all matching type keys.
    """
    summary_by_type = results.get('summary_by_type', {})
    losses = []
    for type_str, stats in summary_by_type.items():
        if fnmatch.fnmatch(type_str, pattern):
            losses.append(stats['avg_loss'])
    return losses


def visualize_distribution(
    all_losses: List[List[float]],
    labels: List[str],
    plot_type: str = "histogram",
    bins: int = 50,
    log_scale: bool = False,
    output_path: Optional[str] = None,
    title: str = "Loss Distribution",
) -> None:
    """
    Visualize the distribution of avg_loss values.

    Args:
        all_losses: List of loss arrays (one per pattern/file).
        labels: Display label for each loss array.
        plot_type: "histogram", "box", or "violin".
        bins: Number of bins for histogram.
        log_scale: Use log scale on x-axis (histogram only).
        output_path: Path to save the figure (None = show interactively).
        title: Plot title.
    """
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#9C27B0', '#FF5722']
    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == "histogram":
        for idx, (losses, label) in enumerate(zip(all_losses, labels)):
            color = colors[idx % len(colors)]
            ax.hist(losses, bins=bins, alpha=0.5, color=color, edgecolor=color,
                    linewidth=0.8, label=f"{label} (n={len(losses)})")
        ax.set_xlabel('Average Loss', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        if log_scale:
            ax.set_xscale('log')

    elif plot_type == "box":
        bp = ax.boxplot(all_losses, patch_artist=True, tick_labels=labels,
                        showfliers=True, notch=True)
        for idx, patch in enumerate(bp['boxes']):
            color = colors[idx % len(colors)]
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel('Average Loss', fontsize=12)

    elif plot_type == "violin":
        vp = ax.violinplot(all_losses, showmeans=True, showmedians=True)
        for idx, body in enumerate(vp['bodies']):
            color = colors[idx % len(colors)]
            body.set_facecolor(color)
            body.set_alpha(0.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Average Loss', fontsize=12)

    # Print summary stats
    print(f"\n{'Label':<30} {'N':>6} {'Mean':>12} {'Std':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
    print("-" * 96)
    for losses, label in zip(all_losses, labels):
        arr = np.array(losses)
        print(f"{label:<30} {len(arr):>6} {arr.mean():>12.6f} {arr.std():>12.6f} "
              f"{np.median(arr):>12.6f} {arr.min():>12.6f} {arr.max():>12.6f}")

    ax.set_title(title, fontsize=14, fontweight='bold')
    if plot_type == "histogram":
        ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize avg_loss distribution for samples matching given patterns"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to loss results JSON file(s)"
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs='+',
        required=True,
        help="fnmatch-style pattern(s) to filter type keys (e.g. 'id_11_base_*', 'id_*_cp1110_*')"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+',
        default=None,
        help="Display labels (default: derived from patterns/filenames)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["histogram", "box", "violin"],
        default="histogram",
        help="Type of distribution plot (default: histogram)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for histogram (default: 50)"
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use log scale on x-axis (histogram only)"
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="Path to save the plot (default: show interactively)"
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default=None,
        help="Title for the plot"
    )

    args = parser.parse_args()

    # Determine how to pair files and patterns:
    #   - Multiple patterns, single file  -> each pattern applied to that file
    #   - Multiple files, single pattern   -> that pattern applied to each file
    #   - Equal count of files and patterns -> paired 1-to-1
    n_files = len(args.results_path)
    n_patterns = len(args.patterns)

    if n_files == 1 and n_patterns >= 1:
        pairs = [(args.results_path[0], p) for p in args.patterns]
    elif n_files >= 1 and n_patterns == 1:
        pairs = [(f, args.patterns[0]) for f in args.results_path]
    elif n_files == n_patterns:
        pairs = list(zip(args.results_path, args.patterns))
    else:
        parser.error("Provide either 1 file with N patterns, N files with 1 pattern, "
                      "or equal counts of files and patterns.")

    # Generate default labels
    if args.labels is None:
        labels = []
        for fpath, pat in pairs:
            fname = fpath.rsplit('/', 1)[-1].replace('.json', '').replace('losses_', '')
            if n_files == 1:
                labels.append(pat)
            elif n_patterns == 1:
                labels.append(fname)
            else:
                labels.append(f"{fname} | {pat}")
    else:
        labels = args.labels

    # Collect losses
    all_losses = []
    for (fpath, pattern), label in zip(pairs, labels):
        results = load_results(fpath)
        losses = collect_losses_by_pattern(results, pattern)
        if not losses:
            print(f"WARNING: No matches for pattern '{pattern}' in {fpath}")
        all_losses.append(losses)

    # Default title
    if args.plot_title is None:
        if n_files == 1:
            fname = args.results_path[0].rsplit('/', 1)[-1]
            title = f"Loss Distribution — {fname}"
        else:
            title = "Loss Distribution"
    else:
        title = args.plot_title

    visualize_distribution(
        all_losses,
        labels,
        plot_type=args.plot_type,
        bins=args.bins,
        log_scale=args.log_scale,
        output_path=args.output_plot,
        title=title,
    )


if __name__ == "__main__":
    main()
