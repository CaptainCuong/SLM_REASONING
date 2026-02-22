#!/usr/bin/env python3
"""
Calculate the average log-likelihood of questions matching specified id patterns
at a specific training checkpoint step (or all steps if --step is omitted).

Usage:
    # Single step
    python avg_llh_at_step.py \
        --files prob_tracking/results/track_push_down_1.5B_low.json \
                prob_tracking/results/track_push_down_1.5B_high.json \
        --patterns "id_1*" "id_2*" \
        --step 1110

    # All steps
    python avg_llh_at_step.py \
        --files prob_tracking/results/track_push_down_1.5B_low.json \
        --patterns "id_1*" "id_2*"

The script prints:
  - Matched IDs from each file and their avg_log_likelihood at the given step(s)
  - Overall average and variance across all matched entries
"""

import json
import fnmatch
import argparse
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_step_index(steps: list, step: int) -> int:
    """Return the index of `step` in `steps`, or raise if not found."""
    for i, s in enumerate(steps):
        if s == step:
            return i
    available = ", ".join(str(s) for s in steps)
    raise ValueError(
        f"Step {step} not found in file. Available steps: {available}"
    )


def match_patterns(keys: list, patterns: list) -> list:
    """Return keys that match any of the given fnmatch patterns."""
    matched = []
    for key in keys:
        if any(fnmatch.fnmatch(key, p) for p in patterns):
            matched.append(key)
    return matched


def stats(values: list) -> tuple:
    """Return (mean, variance, std_dev) for a list of floats."""
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, var, var ** 0.5


def compute_at_step(files: list, patterns: list, step: int):
    """
    Collect avg_log_likelihood values for all matched IDs at a single step.
    Returns (mean, variance, std_dev, all_values).
    """
    all_values = []

    for filepath in files:
        data = load_json(filepath)
        step_idx = find_step_index(data["steps"], step)

        results = data.get("results_by_type", {})
        matched_keys = match_patterns(list(results.keys()), patterns)

        if not matched_keys:
            print(f"  [{Path(filepath).name}] No keys matched patterns {patterns}")
            continue

        file_values = []
        for key in matched_keys:
            value = results[key]["avg_log_likelihood"][step_idx]
            if value is not None:
                file_values.append(value)
                print(f"  [{Path(filepath).name}] {key}: {value:.6f}")
            else:
                print(f"  [{Path(filepath).name}] {key}: None (skipped)")

        if file_values:
            fm, fv, fs = stats(file_values)
            print(f"  [{Path(filepath).name}] file average ({len(file_values)} entries): "
                  f"{fm:.6f}  var={fv:.6f}  std={fs:.6f}")
            all_values.extend(file_values)

    if not all_values:
        raise RuntimeError("No values collected — check your patterns and step.")

    mean, var, std = stats(all_values)
    return mean, var, std, all_values


def compute_all_steps(files: list, patterns: list):
    """
    Collect avg_log_likelihood for all matched IDs across every step.
    Returns a list of (step, mean, variance, std_dev, n_values) tuples,
    one per step (using the union of steps across all files).
    """
    # Gather the step list from the first file; verify consistency across files.
    reference_steps = None
    loaded = []
    for filepath in files:
        data = load_json(filepath)
        loaded.append((filepath, data))
        if reference_steps is None:
            reference_steps = data["steps"]

    results_per_step = []
    for step_idx, step in enumerate(reference_steps):
        all_values = []
        for filepath, data in loaded:
            results = data.get("results_by_type", {})
            matched_keys = match_patterns(list(results.keys()), patterns)
            for key in matched_keys:
                series = results[key]["avg_log_likelihood"]
                if step_idx < len(series) and series[step_idx] is not None:
                    all_values.append(series[step_idx])

        if all_values:
            mean, var, std = stats(all_values)
            results_per_step.append((step, mean, var, std, len(all_values)))
        else:
            results_per_step.append((step, None, None, None, 0))

    return results_per_step


def main():
    parser = argparse.ArgumentParser(
        description="Compute average log-likelihood (and variance) for matched IDs at a given step or all steps."
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more result JSON files.",
    )
    parser.add_argument(
        "--patterns", "-p",
        nargs="+",
        required=True,
        metavar="PATTERN",
        help='One or more fnmatch-style id patterns, e.g. "id_1*" "id_2*".',
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=None,
        metavar="STEP",
        help="Training checkpoint step number (e.g. 1110). Omit to compute for all steps.",
    )
    args = parser.parse_args()

    print(f"\nFiles   : {args.files}")
    print(f"Patterns: {args.patterns}")
    print(f"Step    : {args.step if args.step is not None else 'all'}\n")

    if args.step is not None:
        mean, var, std, all_values = compute_at_step(args.files, args.patterns, args.step)
        print(f"\n{'='*60}")
        print(f"Total matched entries : {len(all_values)}")
        print(f"Overall avg log-likelihood at step {args.step}: {mean:.6f}")
        print(f"Variance                               : {var:.6f}")
        print(f"Std dev                                : {std:.6f}")
        print(f"{'='*60}\n")
    else:
        rows = compute_all_steps(args.files, args.patterns)
        print(f"\n{'='*72}")
        print(f"{'Step':>8}  {'Avg LLH':>14}  {'Variance':>14}  {'Std Dev':>14}  {'N':>5}")
        print(f"{'-'*72}")
        for step, mean, var, std, n in rows:
            if mean is not None:
                print(f"{step:>8}  {mean:>14.6f}  {var:>14.6f}  {std:>14.6f}  {n:>5}")
            else:
                print(f"{step:>8}  {'N/A':>14}  {'N/A':>14}  {'N/A':>14}  {n:>5}")
        print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
