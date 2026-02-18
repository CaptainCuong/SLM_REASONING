#!/usr/bin/env python3
"""
Mix training data from two files by randomly selecting the answer (output) for
each question based on a given ratio.

Both files must have questions in the same order. Only the `output` field is
selected; `instruction` and `input` are taken from file_a.

Example:
    python mix_training_data.py \
        --file_a data/math12K_highest_likelihood.json \
        --file_b data/math12K_lowest_likelihood.json \
        --ratio_a 0.7 \
        --output data/math12K_mixed_70_30.json \
        --seed 42
"""

import json
import random
import argparse
import os


def mix_training_data(
    file_a: str,
    file_b: str,
    ratio_a: float,
    output_file: str,
    seed: int = 42,
):
    """
    For each question (same index in both files), randomly pick the output
    from file_a with probability `ratio_a`, otherwise from file_b.

    Args:
        file_a: Path to the first training data file.
        file_b: Path to the second training data file.
        ratio_a: Probability [0, 1] of picking the answer from file_a.
        output_file: Path to write the mixed output.
        seed: Random seed for reproducibility.
    """
    if not 0.0 <= ratio_a <= 1.0:
        raise ValueError(f"ratio_a must be between 0 and 1, got {ratio_a}")

    print("=" * 80)
    print("MIX TRAINING DATA")
    print("=" * 80)
    print(f"\n  file_a  : {file_a}")
    print(f"  file_b  : {file_b}")
    print(f"  ratio_a : {ratio_a:.4f}  (file_b: {1 - ratio_a:.4f})")
    print(f"  seed    : {seed}")
    print(f"  output  : {output_file}\n")

    with open(file_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(file_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    print(f"Loaded {len(data_a)} entries from file_a")
    print(f"Loaded {len(data_b)} entries from file_b")

    if len(data_a) != len(data_b):
        raise ValueError(
            f"Files have different lengths: {len(data_a)} vs {len(data_b)}"
        )

    rng = random.Random(seed)
    mixed = []
    count_a = 0
    count_b = 0

    for idx, (entry_a, entry_b) in enumerate(zip(data_a, data_b)):
        if rng.random() < ratio_a:
            chosen_output = entry_a["output"]
            count_a += 1
        else:
            chosen_output = entry_b["output"]
            count_b += 1

        mixed.append({
            "instruction": entry_a["instruction"],
            "input": entry_a["input"],
            "output": chosen_output,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mixed, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_file) / (1024 * 1024)

    print(f"\nResults:")
    print(f"  Total entries : {len(mixed)}")
    print(f"  From file_a   : {count_a} ({count_a / len(mixed) * 100:.1f}%)")
    print(f"  From file_b   : {count_b} ({count_b / len(mixed) * 100:.1f}%)")
    print(f"  Output size   : {file_size:.2f} MB")
    print(f"\nSaved to {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Mix answers from two training data files with a given ratio."
    )
    parser.add_argument(
        "--file_a",
        type=str,
        default="data/math12K_highest_likelihood.json",
        help="Path to the first training file (answers chosen with probability ratio_a).",
    )
    parser.add_argument(
        "--file_b",
        type=str,
        default="data/math12K_lowest_likelihood.json",
        help="Path to the second training file (answers chosen with probability 1 - ratio_a).",
    )
    parser.add_argument(
        "--ratio_a",
        type=float,
        default=0.5,
        help="Probability of selecting the answer from file_a (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output file path. Defaults to data/math12K_mixed_<ratio_a*100>_<ratio_b*100>.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    if args.output is None:
        pct_a = int(round(args.ratio_a * 100))
        pct_b = 100 - pct_a
        args.output = f"data/math12K_mixed_{pct_a}_{pct_b}.json"

    mix_training_data(
        file_a=args.file_a,
        file_b=args.file_b,
        ratio_a=args.ratio_a,
        output_file=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
