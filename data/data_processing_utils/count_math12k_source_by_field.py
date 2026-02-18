#!/usr/bin/env python3
"""
Count how many selected answers come from each source (e.g., "gemma3_72b", "qwen2.5_72b")
when selecting by a given field (lowest/highest value).

Example fields: "Qwen2.5_Math_7B_gradient_norm", "gemma_3_27b_llh", "log_likelihood", etc.
"""

import json
import math
import argparse
from collections import Counter


def count_source_by_field(input_file: str, field_name: str):
    """
    For each question, select the answer with the lowest and highest field value,
    then count how many come from each source.
    """
    print("=" * 80)
    print(f"COUNTING SOURCES BY FIELD: {field_name}")
    print("=" * 80)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions")

    lowest_source_counter = Counter()
    highest_source_counter = Counter()
    skipped = 0

    for question_item in data:
        answers = question_item['answers']

        # Filter valid answers with finite field values
        finite_answers = [
            ans for ans in answers
            if ans.get(field_name) is not None
            and not math.isnan(ans[field_name])
            and not math.isinf(ans[field_name])
        ]

        if not finite_answers:
            skipped += 1
            continue

        sorted_answers = sorted(finite_answers, key=lambda x: x[field_name])

        lowest_answer = sorted_answers[0]
        highest_answer = sorted_answers[-1]

        lowest_source_counter[lowest_answer.get('source', 'unknown')] += 1
        highest_source_counter[highest_answer.get('source', 'unknown')] += 1

    # Print results
    processed = len(data) - skipped
    print(f"\nProcessed: {processed} / {len(data)} questions (skipped {skipped})")

    print(f"\n--- Lowest {field_name} (selected) ---")
    for source, count in lowest_source_counter.most_common():
        pct = count / processed * 100 if processed else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print(f"\n--- Highest {field_name} ---")
    for source, count in highest_source_counter.most_common():
        pct = count / processed * 100 if processed else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Count answer sources when selecting by a given field"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/cuongdc/SLM_REASONING/data/math12K_merged_loglikelihood.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--field",
        type=str,
        required=True,
        help="Field name to sort by (e.g., 'Qwen2.5_Math_7B_loglikelihood', 'log_likelihood')"
    )

    args = parser.parse_args()
    count_source_by_field(args.input_file, args.field)


if __name__ == "__main__":
    main()
