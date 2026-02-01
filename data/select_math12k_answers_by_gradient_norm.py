#!/usr/bin/env python3
"""
Select one answer per question from math12K file based on gradient norm:
- Lowest gradient norm (selected)
- Highest gradient norm (for comparison)

Output format matches identity.json structure.
"""

import json
import math
from typing import List, Dict
import os

def select_answers_by_gradient_norm(
    input_file: str,
    output_lowest: str,
    output_highest: str,
    instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
):
    """
    Select answers based on gradient norm and save in identity.json format.

    Args:
        input_file: Path to merged answers with gradient norms
        output_lowest: Output file for lowest gradient norm answers
        output_highest: Output file for highest gradient norm answers
        instruction: Instruction text to use for all entries
    """
    print("=" * 80)
    print("SELECTING ANSWERS BY GRADIENT NORM (MATH12K)")
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

        # Filter valid answers (with non-None gradient_norm)
        valid_answers = [
            ans for ans in answers
            if ans.get('gradient_norm') is not None
        ]

        if not valid_answers:
            stats['skipped_no_valid_answers'] += 1
            continue

        # Filter out answers with NaN or infinite gradient_norm
        finite_answers = [
            ans for ans in valid_answers
            if not math.isnan(ans['gradient_norm'])
            and not math.isinf(ans['gradient_norm'])
        ]

        if not finite_answers:
            stats['skipped_all_nan_inf'] += 1
            continue

        # Sort by gradient_norm (lower is better for selection)
        sorted_answers = sorted(finite_answers, key=lambda x: x['gradient_norm'])

        # Select answers
        lowest_answer = sorted_answers[0]   # Lowest gradient norm
        highest_answer = sorted_answers[-1]  # Highest gradient norm

        # Add to output lists
        lowest_data.append({
            "instruction": instruction,
            "input": question,
            "output": lowest_answer['answer']
        })

        highest_data.append({
            "instruction": instruction,
            "input": question,
            "output": highest_answer['answer']
        })

        stats['processed'] += 1

        if (q_idx + 1) % 1000 == 0:
            print(f"  Processed {q_idx + 1}/{len(data)} questions...")

    # Save outputs
    print("\n" + "-" * 80)
    print("Saving output files...")
    print("-" * 80)

    print(f"\nSaving lowest gradient norm answers to {output_lowest}...")
    with open(output_lowest, 'w', encoding='utf-8') as f:
        json.dump(lowest_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_lowest) / (1024 * 1024)
    print(f"  Saved {len(lowest_data)} entries ({file_size:.2f} MB)")

    print(f"\nSaving highest gradient norm answers to {output_highest}...")
    with open(output_highest, 'w', encoding='utf-8') as f:
        json.dump(highest_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_highest) / (1024 * 1024)
    print(f"  Saved {len(highest_data)} entries ({file_size:.2f} MB)")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total questions: {stats['total_questions']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (no valid answers): {stats['skipped_no_valid_answers']}")
    print(f"Skipped (all NaN/Inf): {stats['skipped_all_nan_inf']}")
    print(f"\nOutput entries:")
    print(f"  Lowest gradient norm: {len(lowest_data)}")
    print(f"  Highest gradient norm: {len(highest_data)}")

    # Show sample entries
    if lowest_data:
        print("\n" + "-" * 80)
        print("SAMPLE ENTRIES")
        print("-" * 80)

        print("\nLowest gradient norm sample:")
        sample = lowest_data[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")

        print("\nHighest gradient norm sample:")
        sample = highest_data[0]
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return stats


def analyze_selections(
    lowest_file: str,
    highest_file: str,
    original_file: str
):
    """
    Analyze the selected answers to show source distribution.
    """
    print("\n" + "=" * 80)
    print("ANALYZING SOURCE DISTRIBUTION")
    print("=" * 80)

    # Load original data with sources
    print("Loading original data...")
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # Load selected files
    print("Loading selected files...")
    with open(lowest_file, 'r', encoding='utf-8') as f:
        lowest_data = json.load(f)

    with open(highest_file, 'r', encoding='utf-8') as f:
        highest_data = json.load(f)

    # Count sources for each selection type
    def count_sources(selected_data, original_data):
        source_counts = {}
        for selected_item in selected_data:
            question = selected_item['input']
            answer = selected_item['output']

            # Find this question in original data
            for q_item in original_data:
                if q_item['question'] == question:
                    # Find matching answer and get source
                    for ans in q_item['answers']:
                        if ans['answer'] == answer:
                            source = ans.get('source', 'unknown')
                            source_counts[source] = source_counts.get(source, 0) + 1
                            break
                    break
        return source_counts

    print("\nCounting sources...")
    print("\nSource distribution for lowest gradient norm:")
    lowest_sources = count_sources(lowest_data, original_data)
    for source, count in sorted(lowest_sources.items()):
        pct = count / len(lowest_data) * 100 if lowest_data else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print("\nSource distribution for highest gradient norm:")
    highest_sources = count_sources(highest_data, original_data)
    for source, count in sorted(highest_sources.items()):
        pct = count / len(highest_data) * 100 if highest_data else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Configuration
    input_file = "/data/home/cuong/SLM_REASONING/data/math12K_merged_answers_with_gradient_norm.json"
    output_dir = "/data/home/cuong/SLM_REASONING/data"

    output_lowest = f"{output_dir}/math12K_lowest_gradient_norm.json"
    output_highest = f"{output_dir}/math12K_highest_gradient_norm.json"

    print(f"\nConfiguration:")
    print(f"  Input: {input_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output files:")
    print(f"    - {output_lowest}")
    print(f"    - {output_highest}")
    print()

    # Select answers
    stats = select_answers_by_gradient_norm(
        input_file=input_file,
        output_lowest=output_lowest,
        output_highest=output_highest
    )

    # Analyze source distribution
    print("\nStarting source analysis (this may take a while for large files)...")
    analyze_selections(
        lowest_file=output_lowest,
        highest_file=output_highest,
        original_file=input_file
    )

    print("\nDone!")
