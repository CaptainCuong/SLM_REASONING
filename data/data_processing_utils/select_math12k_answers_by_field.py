#!/usr/bin/env python3
"""
Select one answer per question from math12K file based on a given field:
- Lowest value (selected)
- Highest value (for comparison)

Output format matches identity.json structure.

Example fields: "Qwen2.5_Math_7B_gradient_norm", "gemma_3_27b_llh", "log_likelihood", etc.
"""

import json
import math
import argparse
from typing import List, Dict
import os


def select_answers_by_field(
    input_file: str,
    output_lowest: str,
    output_highest: str,
    field_name: str,
    instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
):
    """
    Select answers based on a given field and save in identity.json format.

    Args:
        input_file: Path to merged answers file
        output_lowest: Output file for lowest field value answers
        output_highest: Output file for highest field value answers
        field_name: The field name to sort by (e.g., "Qwen2.5_Math_7B_gradient_norm")
        instruction: Instruction text to use for all entries
    """
    print("=" * 80)
    print(f"SELECTING ANSWERS BY FIELD: {field_name}")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} questions")

    # Prepare output lists
    lowest_data = []
    highest_data = []

    # Statistics
    stats = {
        'total_questions': len(data),
        'skipped_no_valid_answers': 0,
        'skipped_all_nan_inf': 0,
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

        # Sort by field value (lower first)
        sorted_answers = sorted(finite_answers, key=lambda x: x[field_name])

        # Select answers
        lowest_answer = sorted_answers[0]   # Lowest field value
        highest_answer = sorted_answers[-1]  # Highest field value

        # Add to output lists
        lowest_data.append({
            "instruction": instruction,
            "input": question,
            "output": lowest_answer['answer'],
            "source": lowest_answer.get('source')
        })

        highest_data.append({
            "instruction": instruction,
            "input": question,
            "output": highest_answer['answer'],
            "source": highest_answer.get('source')
        })

        stats['processed'] += 1

        if (q_idx + 1) % 1000 == 0:
            print(f"  Processed {q_idx + 1}/{len(data)} questions...")

    # Save outputs
    print("\n" + "-" * 80)
    print("Saving output files...")
    print("-" * 80)

    print(f"\nSaving lowest {field_name} answers to {output_lowest}...")
    with open(output_lowest, 'w', encoding='utf-8') as f:
        json.dump(lowest_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_lowest) / (1024 * 1024)
    print(f"  Saved {len(lowest_data)} entries ({file_size:.2f} MB)")

    print(f"\nSaving highest {field_name} answers to {output_highest}...")
    with open(output_highest, 'w', encoding='utf-8') as f:
        json.dump(highest_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_highest) / (1024 * 1024)
    print(f"  Saved {len(highest_data)} entries ({file_size:.2f} MB)")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Field used: {field_name}")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (no valid answers): {stats['skipped_no_valid_answers']}")
    print(f"Skipped (all NaN/Inf): {stats['skipped_all_nan_inf']}")
    print(f"\nOutput entries:")
    print(f"  Lowest {field_name}: {len(lowest_data)}")
    print(f"  Highest {field_name}: {len(highest_data)}")

    # Show sample entries
    if lowest_data:
        print("\n" + "-" * 80)
        print("SAMPLE ENTRIES")
        print("-" * 80)

        print(f"\nLowest {field_name} sample:")
        sample = lowest_data[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")
        print(f"  Source: {sample['source']}")

        print(f"\nHighest {field_name} sample:")
        sample = highest_data[0]
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")
        print(f"  Source: {sample['source']}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return stats


def list_available_fields(input_file: str):
    """List all available numeric fields in the answers."""
    print(f"\nScanning {input_file} for available fields...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect all fields from first few questions
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
        description="Select answers by a given field (lowest/highest)"
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

    # Generate output filenames based on field name
    field_clean = args.field.replace('.', '_').replace(' ', '_')
    output_lowest = os.path.join(args.output_dir, f"math12K_lowest_{field_clean}.json")
    output_highest = os.path.join(args.output_dir, f"math12K_highest_{field_clean}.json")

    print(f"\nConfiguration:")
    print(f"  Input: {args.input_file}")
    print(f"  Field: {args.field}")
    print(f"  Output files:")
    print(f"    - {output_lowest}")
    print(f"    - {output_highest}")
    print()

    # Select answers
    select_answers_by_field(
        input_file=args.input_file,
        output_lowest=output_lowest,
        output_highest=output_highest,
        field_name=args.field,
        instruction=args.instruction
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
