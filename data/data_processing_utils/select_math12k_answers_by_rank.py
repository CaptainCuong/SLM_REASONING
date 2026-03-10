#!/usr/bin/env python3
"""
Select one answer per question from math12K file based on the r-th highest value
of a given field.

For example, r=1 selects the highest value, r=2 selects the second highest, etc.

Output format matches identity.json structure.

Example fields: "Qwen2.5_Math_7B_gradient_norm", "gemma_3_27b_llh", "log_likelihood", etc.
"""

import json
import math
import argparse
from typing import List, Dict
import os


def select_answers_by_rank(
    input_file: str,
    output_file: str,
    field_name: str,
    rank: int = 2,
    instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
):
    """
    Select the answer with the r-th highest field value per question.

    Args:
        input_file: Path to merged answers file
        output_file: Output file path
        field_name: The field name to sort by (e.g., "Qwen2.5_Math_7B_gradient_norm")
        rank: Which rank to select (1 = highest, 2 = second highest, etc.)
        instruction: Instruction text to use for all entries
    """
    print("=" * 80)
    print(f"SELECTING ANSWERS BY RANK {rank} OF FIELD: {field_name}")
    print("=" * 80)

    print(f"\nLoading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} questions")

    output_data = []

    stats = {
        'total_questions': len(data),
        'skipped_no_valid_answers': 0,
        'skipped_all_nan_inf': 0,
        'skipped_insufficient_answers': 0,
        'processed': 0
    }

    print("\nProcessing questions...")

    for q_idx, question_item in enumerate(data):
        question = question_item['question']
        answers = question_item['answers']

        # Filter valid answers (with non-None field value)
        valid_answers = [
            ans for ans in answers
            if ans.get(field_name) is not None
        ]

        if not valid_answers:
            stats['skipped_no_valid_answers'] += 1
            continue

        # Filter out answers with NaN or infinite field values
        finite_answers = [
            ans for ans in valid_answers
            if not math.isnan(ans[field_name])
            and not math.isinf(ans[field_name])
        ]

        if not finite_answers:
            stats['skipped_all_nan_inf'] += 1
            continue

        # Check if we have enough answers for the requested rank
        if len(finite_answers) < rank:
            stats['skipped_insufficient_answers'] += 1
            continue

        # Sort by field value descending (highest first)
        sorted_answers = sorted(finite_answers, key=lambda x: x[field_name], reverse=True)

        # Select the r-th highest (rank is 1-based, index is 0-based)
        selected_answer = sorted_answers[rank - 1]

        output_data.append({
            "instruction": instruction,
            "input": question,
            "output": selected_answer['answer']
        })

        stats['processed'] += 1

        if (q_idx + 1) % 1000 == 0:
            print(f"  Processed {q_idx + 1}/{len(data)} questions...")

    # Save output
    print("\n" + "-" * 80)
    print("Saving output file...")
    print("-" * 80)

    print(f"\nSaving rank-{rank} {field_name} answers to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {len(output_data)} entries ({file_size:.2f} MB)")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Field used: {field_name}")
    print(f"Rank selected: {rank}")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (no valid answers): {stats['skipped_no_valid_answers']}")
    print(f"Skipped (all NaN/Inf): {stats['skipped_all_nan_inf']}")
    print(f"Skipped (fewer than {rank} answers): {stats['skipped_insufficient_answers']}")
    print(f"\nOutput entries: {len(output_data)}")

    # Show sample entry
    if output_data:
        print("\n" + "-" * 80)
        print("SAMPLE ENTRY")
        print("-" * 80)
        sample = output_data[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return stats


def list_available_fields(input_file: str):
    """List all available numeric fields in the answers."""
    print(f"\nScanning {input_file} for available fields...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_fields = set()
    for q_item in data[:100]:  # Check first 100 questions
        for ans in q_item.get('answers', []):
            for key, value in ans.items():
                if isinstance(value, (int, float)) and key != 'token_count':
                    all_fields.add(key)

    print("\nAvailable numeric fields:")
    for field in sorted(all_fields):
        print(f"  - {field}")

    return sorted(all_fields)


def main():
    parser = argparse.ArgumentParser(
        description="Select answers with the r-th highest value of a given field"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/cuongdc/SLM_REASONING/data/math12K_merged_loglikelihood.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/cuongdc/SLM_REASONING/data",
        help="Output directory"
    )
    parser.add_argument(
        "--field",
        type=str,
        required=True,
        help="Field name to sort by (e.g., 'Qwen2.5_Math_7B_loglikelihood', 'gemma_3_27b_llh')"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=2,
        help="Which rank to select (1 = highest, 2 = second highest, etc.)"
    )
    parser.add_argument(
        "--list_fields",
        action="store_true",
        help="List available fields and exit"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Please reason step by step, and put your final answer within \\boxed{}.",
        help="Instruction text for output"
    )

    args = parser.parse_args()

    if args.list_fields:
        list_available_fields(args.input_file)
        return

    if args.rank < 1:
        parser.error("--rank must be >= 1")

    # Generate output filename based on field name and rank
    field_clean = args.field.replace('.', '_').replace(' ', '_')
    output_file = os.path.join(args.output_dir, f"math12K_rank{args.rank}_{field_clean}.json")

    print(f"\nConfiguration:")
    print(f"  Input: {args.input_file}")
    print(f"  Field: {args.field}")
    print(f"  Rank: {args.rank}")
    print(f"  Output: {output_file}")
    print()

    select_answers_by_rank(
        input_file=args.input_file,
        output_file=output_file,
        field_name=args.field,
        rank=args.rank,
        instruction=args.instruction
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
