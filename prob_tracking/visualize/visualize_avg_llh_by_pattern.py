#!/usr/bin/env python3
"""
Visualize average log-likelihood curves across training steps, one curve per
id pattern.  Multiple result JSON files are merged before averaging.

Usage:
    python visualize_avg_llh_by_pattern.py \
        --files  prob_tracking/results/track_push_down_1.5B_low.json \
                 prob_tracking/results/track_push_down_1.5B_high.json \
        --patterns "id_1*" "id_2*" \
        --notations "Group 1" "Group 2" \
        --output  prob_tracking/image/avg_llh_by_pattern.png

Each pattern gets one averaged curve labeled with the corresponding notation.
"""

import json
import fnmatch
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── data helpers ────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_patterns(keys: list, patterns: list) -> list:
    return [k for k in keys if any(fnmatch.fnmatch(k, p) for p in patterns)]


def compute_curve(loaded: list, pattern: str) -> tuple[list, list, list]:
    """
    For a single fnmatch pattern, compute per-step (mean, std) across all
    matching IDs in all loaded files.

    Returns (steps, means, stds).  Steps are taken from the first file.
    """
    reference_steps = loaded[0][1]["steps"]
    n_steps = len(reference_steps)

    values_per_step: list[list] = [[] for _ in range(n_steps)]

    for _, data in loaded:
        results = data.get("results_by_type", {})
        matched = match_patterns(list(results.keys()), [pattern])
        for key in matched:
            series = results[key]["avg_log_likelihood"]
            for i in range(min(n_steps, len(series))):
                if series[i] is not None:
                    values_per_step[i].append(series[i])

    means, stds, valid_steps = [], [], []
    for step, vals in zip(reference_steps, values_per_step):
        if vals:
            arr = np.array(vals)
            means.append(arr.mean())
            stds.append(arr.std())
            valid_steps.append(step)

    return valid_steps, means, stds


# ── plotting ─────────────────────────────────────────────────────────────────

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_curves(curves: list, output_path: str = None, show: bool = True,
                title: str = "Average Log-Likelihood by Pattern",
                shade_std: bool = True):
    """
    curves: list of (notation, steps, means, stds)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (notation, steps, means, stds) in enumerate(curves):
        color = COLORS[idx % len(COLORS)]
        steps_arr  = np.array(steps)
        means_arr  = np.array(means)
        stds_arr   = np.array(stds)

        ax.plot(steps_arr, means_arr, marker="o", linewidth=2,
                markersize=5, color=color, label=notation)

        if shade_std:
            ax.fill_between(steps_arr,
                            means_arr - stds_arr,
                            means_arr + stds_arr,
                            alpha=0.15, color=color)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Average Log-Likelihood", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot average log-likelihood curves, one per id pattern."
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+", required=True, metavar="FILE",
        help="One or more result JSON files.",
    )
    parser.add_argument(
        "--patterns", "-p",
        nargs="+", required=True, metavar="PATTERN",
        help='fnmatch-style id patterns, e.g. "id_1*" "id_2*".',
    )
    parser.add_argument(
        "--notations", "-n",
        nargs="+", required=True, metavar="LABEL",
        help="Display label for each pattern (same order, same count).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None, metavar="PATH",
        help="Path to save the plot image (optional).",
    )
    parser.add_argument(
        "--title", "-t",
        default="Average Log-Likelihood by Pattern",
        help="Plot title.",
    )
    parser.add_argument(
        "--no_shade",
        action="store_true",
        help="Disable ±1 std-dev shading around each curve.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the plot interactively.",
    )
    args = parser.parse_args()

    if len(args.patterns) != len(args.notations):
        parser.error(
            f"--patterns has {len(args.patterns)} entries but "
            f"--notations has {len(args.notations)}.  They must match."
        )

    # Load all files once
    loaded = []
    for filepath in args.files:
        print(f"Loading {filepath} ...")
        loaded.append((filepath, load_json(filepath)))

    # Compute one curve per pattern
    curves = []
    for pattern, notation in zip(args.patterns, args.notations):
        steps, means, stds = compute_curve(loaded, pattern)
        n_ids = sum(
            len(match_patterns(list(d.get("results_by_type", {}).keys()), [pattern]))
            for _, d in loaded
        )
        print(f"  Pattern '{pattern}' ({notation}): {n_ids} matched IDs, "
              f"{len(steps)} steps with data")
        curves.append((notation, steps, means, stds))

    plot_curves(
        curves,
        output_path=args.output,
        show=not args.no_show,
        title=args.title,
        shade_std=not args.no_shade,
    )


if __name__ == "__main__":
    main()
