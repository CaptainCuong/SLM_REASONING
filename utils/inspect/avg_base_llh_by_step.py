#!/usr/bin/env python3
"""
Calculate the average log-likelihood under the base model for two groups of answers
determined at a given checkpoint step:
  - entries classified as *correct* at that step
  - entries classified as *incorrect* at that step

Input: a summary JSON file like
    prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json
and an optional --step argument (e.g. 1110). If --step is omitted, all steps
are processed and results are shown in a compact table.

The script looks up entries whose keys match `*_cp{step}_correct` and
`*_cp{step}_incorrect` (using `_` as a word boundary to avoid e.g. cp1110
matching cp11100), then reads the base-model avg_log_likelihood (index 0 in
the checkpoints list, which is always "base_model") and reports mean and
variance per group.

Usage:
    # Single step
    python utils/inspect/avg_base_llh_by_step.py \\
        prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json \\
        --step 1110

    # All steps (table)
    python utils/inspect/avg_base_llh_by_step.py \\
        prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json

    # Show per-entry values (only meaningful with --step):
    python utils/inspect/avg_base_llh_by_step.py \\
        prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json \\
        --step 1110 --verbose

    # Write output to a file:
    python utils/inspect/avg_base_llh_by_step.py \\
        prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json \\
        --output results.txt
"""

import json
import sys
import argparse
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stats(values: list) -> tuple:
    """Return (mean, variance, std_dev) for a non-empty list of floats."""
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, var, var ** 0.5


def _collect_for_step(results: dict, step: int) -> tuple:
    """
    Scan `results_by_type` dict for a single step.

    Returns:
        correct_items   : list of (key, base_llh)
        incorrect_items : list of (key, base_llh)
    """
    correct_suffix = f"_cp{step}_correct"
    incorrect_suffix = f"_cp{step}_incorrect"

    correct_items = []
    incorrect_items = []

    for key, entry in results.items():
        llh_series = entry.get("avg_log_likelihood", [])
        if not llh_series or llh_series[0] is None:
            continue
        base_llh = llh_series[0]  # index 0 == base_model

        if key.endswith(correct_suffix):
            correct_items.append((key, base_llh))
        elif key.endswith(incorrect_suffix):
            incorrect_items.append((key, base_llh))

    return correct_items, incorrect_items


def collect_base_llh_by_step(summary_path: str, step: int, verbose: bool = False, out=sys.stdout):
    """
    Load the summary JSON, find all keys for the given step, split into
    correct / incorrect, extract base-model avg_log_likelihood, compute stats.

    Returns:
        correct_vals   : list of float
        incorrect_vals : list of float
    """
    data = load_json(summary_path)

    checkpoints = data.get("checkpoints", [])
    if not checkpoints or checkpoints[0] != "base_model":
        raise ValueError(
            f"Expected checkpoints[0] == 'base_model', got {checkpoints[:3]}"
        )

    steps = data.get("steps", [])
    if step not in steps:
        available = ", ".join(str(s) for s in steps)
        raise ValueError(f"Step {step} not found. Available steps: {available}")

    results = data.get("results_by_type", {})
    correct_items, incorrect_items = _collect_for_step(results, step)

    if verbose:
        print(f"\n--- correct entries (step {step}) ---", file=out)
        for key, v in sorted(correct_items):
            print(f"  {key}: {v:.6f}", file=out)
        print(f"\n--- incorrect entries (step {step}) ---", file=out)
        for key, v in sorted(incorrect_items):
            print(f"  {key}: {v:.6f}", file=out)

    return [v for _, v in correct_items], [v for _, v in incorrect_items]


def collect_all_steps(summary_path: str):
    """
    For every step in the summary file, collect correct / incorrect base-model
    log-likelihoods and compute stats.

    Returns:
        steps_list : list of int
        rows       : list of (step, n_correct, mean_c, var_c, n_incorrect, mean_i, var_i, delta)
                     Values are None when a group has no entries for that step.
    """
    data = load_json(summary_path)

    checkpoints = data.get("checkpoints", [])
    if not checkpoints or checkpoints[0] != "base_model":
        raise ValueError(
            f"Expected checkpoints[0] == 'base_model', got {checkpoints[:3]}"
        )

    steps_list = data.get("steps", [])
    results = data.get("results_by_type", {})

    rows = []
    for step in steps_list:
        correct_items, incorrect_items = _collect_for_step(results, step)
        correct_vals = [v for _, v in correct_items]
        incorrect_vals = [v for _, v in incorrect_items]

        mean_c = var_c = mean_i = var_i = delta = None
        if correct_vals:
            mean_c, var_c, _ = stats(correct_vals)
        if incorrect_vals:
            mean_i, var_i, _ = stats(incorrect_vals)
        if mean_c is not None and mean_i is not None:
            delta = mean_c - mean_i

        rows.append((step, len(correct_vals), mean_c, var_c, len(incorrect_vals), mean_i, var_i, delta))

    return steps_list, rows


def print_report(summary_path: str, step: int, correct_vals: list, incorrect_vals: list, out=sys.stdout):
    print("=" * 68, file=out)
    print(f"File : {summary_path}", file=out)
    print(f"Step : {step}  (base model log-likelihood comparison)", file=out)
    print("=" * 68, file=out)

    for label, vals in [("correct", correct_vals), ("incorrect", incorrect_vals)]:
        print(f"\n[{label.upper()}]  n = {len(vals)}", file=out)
        if not vals:
            print("  No entries found.", file=out)
            continue
        mean, var, std = stats(vals)
        print(f"  Mean     : {mean:.6f}", file=out)
        print(f"  Variance : {var:.6f}", file=out)
        print(f"  Std dev  : {std:.6f}", file=out)
        print(f"  Min      : {min(vals):.6f}", file=out)
        print(f"  Max      : {max(vals):.6f}", file=out)

    if correct_vals and incorrect_vals:
        mean_c, _, _ = stats(correct_vals)
        mean_i, _, _ = stats(incorrect_vals)
        print(f"\n[DELTA]  correct_mean - incorrect_mean = {mean_c - mean_i:.6f}", file=out)

    print("=" * 68, file=out)


def print_all_steps_table(summary_path: str, rows: list, out=sys.stdout):
    W = 92
    print("=" * W, file=out)
    print(f"File : {summary_path}", file=out)
    print(f"Base model log-likelihood — correct vs incorrect across all steps", file=out)
    print("=" * W, file=out)
    hdr = (f"{'Step':>8}  {'N_corr':>6}  {'Mean_corr':>12}  {'Var_corr':>12}"
           f"  {'N_incorr':>8}  {'Mean_incorr':>12}  {'Var_incorr':>12}  {'Delta':>10}")
    print(hdr, file=out)
    print("-" * W, file=out)
    for step, nc, mc, vc, ni, mi, vi, delta in rows:
        mc_s  = f"{mc:.4f}"  if mc    is not None else "N/A"
        vc_s  = f"{vc:.4f}"  if vc    is not None else "N/A"
        mi_s  = f"{mi:.4f}"  if mi    is not None else "N/A"
        vi_s  = f"{vi:.4f}"  if vi    is not None else "N/A"
        d_s   = f"{delta:.4f}" if delta is not None else "N/A"
        print(f"{step:>8}  {nc:>6}  {mc_s:>12}  {vc_s:>12}"
              f"  {ni:>8}  {mi_s:>12}  {vi_s:>12}  {d_s:>10}", file=out)
    print("=" * W, file=out)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate avg base-model log-likelihood for correct vs incorrect groups "
            "at a given checkpoint step, from a summary JSON file."
        )
    )
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to the summary JSON file (e.g. prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json)",
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=None,
        help="Checkpoint step number (e.g. 1110). Omit to compute for all steps.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print per-entry log-likelihood values before the summary",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output file (default: stdout)",
    )
    args = parser.parse_args()

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    try:
        if args.step is not None:
            correct_vals, incorrect_vals = collect_base_llh_by_step(
                args.summary_file, args.step, verbose=args.verbose, out=out
            )
            print_report(args.summary_file, args.step, correct_vals, incorrect_vals, out=out)
        else:
            _, rows = collect_all_steps(args.summary_file)
            print_all_steps_table(args.summary_file, rows, out=out)
    finally:
        if args.output:
            out.close()
            print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
