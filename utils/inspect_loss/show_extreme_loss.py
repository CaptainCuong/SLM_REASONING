#!/usr/bin/env python3
"""
Show incorrect answers with lowest loss and correct answers with highest loss.

These are the most "surprising" cases:
- Incorrect answers the model is very confident about (low loss)
- Correct answers the model is least confident about (high loss)

Input: a loss results file like ntk/results/test_set_results/losses_gemma_amc_high_1.5B.json

Usage:
    python utils/inspect/show_extreme_loss.py path/to/loss_results.json --top_k 5 --max_output_len 500

    # Write output to a file:
    python utils/inspect/show_extreme_loss.py path/to/loss_results.json -o output.txt
"""

import json
import sys
import argparse


def show_extreme_loss(loss_file: str, top_k: int = 5, max_output_len: int = 500, output_file: str = None):
    with open(loss_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pool_path = data['pool_path']
    with open(pool_path, 'r', encoding='utf-8') as f:
        pool = json.load(f)

    results = data['individual_results']
    assert len(results) == len(pool), "Mismatch between results and pool length"

    # Pair each result with its pool entry
    paired = list(zip(results, pool))

    incorrect = [(r, p) for r, p in paired if 'incorrect' in r['type']]
    correct = [(r, p) for r, p in paired if 'correct' in r['type'] and 'incorrect' not in r['type']]

    # Sort: incorrect by loss ascending, correct by loss descending
    incorrect_sorted = sorted(incorrect, key=lambda x: x[0]['loss_stats']['loss'])
    correct_sorted = sorted(correct, key=lambda x: x[0]['loss_stats']['loss'], reverse=True)

    out = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout

    def print_entry(rank, result, pool_entry, max_len):
        loss = result['loss_stats']['loss']
        entry_type = result['type']
        question = pool_entry['input']
        answer = pool_entry['output']
        if len(answer) > max_len:
            answer = answer[:max_len] + "..."
        print(f"  [{rank}] {entry_type}  |  loss = {loss:.6f}", file=out)
        print(f"      Q: {question}", file=out)
        print(f"      A: {answer}", file=out)
        print(file=out)

    print("=" * 80, file=out)
    print(f"Loss file : {loss_file}", file=out)
    print(f"Pool file : {pool_path}", file=out)
    print(f"Total: {len(results)} | Correct: {len(correct)} | Incorrect: {len(incorrect)}", file=out)
    print("=" * 80, file=out)

    print(f"\n--- Top {top_k} INCORRECT answers with LOWEST loss (model confident but wrong) ---\n", file=out)
    for i, (r, p) in enumerate(incorrect_sorted[:top_k], 1):
        print_entry(i, r, p, max_output_len)

    print(f"\n--- Top {top_k} CORRECT answers with HIGHEST loss (model unsure but right) ---\n", file=out)
    for i, (r, p) in enumerate(correct_sorted[:top_k], 1):
        print_entry(i, r, p, max_output_len)

    if output_file:
        out.close()
        print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Show incorrect answers with lowest loss and correct answers with highest loss"
    )
    parser.add_argument(
        "loss_file",
        type=str,
        help="Path to loss results JSON file"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of entries to show per category (default: 5)"
    )
    parser.add_argument(
        "--max_output_len", type=int, default=500,
        help="Max characters of answer to display (default: 500)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output file (if not set, prints to stdout)"
    )
    args = parser.parse_args()
    show_extreme_loss(args.loss_file, args.top_k, args.max_output_len, args.output)


if __name__ == "__main__":
    main()
