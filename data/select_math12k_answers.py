#!/usr/bin/env python3
"""
Select one answer per question from math12K file based on different criteria:
- Highest log-likelihood
- Lowest log-likelihood
- Random selection

Output format matches identity.json structure.
"""

import json
import random
from typing import List, Dict
import os

def select_answers(
    input_file: str,
    output_highest: str,
    output_lowest: str,
    output_random: str,
    instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
):
    """
    Select answers based on different criteria and save in identity.json format.

    Args:
        input_file: Path to merged answers with log-likelihoods
        output_highest: Output file for highest likelihood answers
        output_lowest: Output file for lowest likelihood answers
        output_random: Output file for random answers
        instruction: Instruction text to use for all entries
    """
    print("=" * 80)
    print("SELECTING ANSWERS BY LOG-LIKELIHOOD (MATH12K)")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} questions")

    # Prepare output lists
    highest_data = []
    lowest_data = []
    random_data = []

    # Statistics
    stats = {
        'total_questions': len(data),
        'skipped_no_valid_answers': 0,
        'skipped_all_inf': 0,
        'processed': 0
    }

    print("\nProcessing questions...")

    for q_idx, question_item in enumerate(data):
        question = question_item['question']
        answers = question_item['answers']

        # Filter valid answers (with non-None log-likelihood)
        valid_answers = [
            ans for ans in answers
            if ans.get('log_likelihood') is not None
        ]

        if not valid_answers:
            stats['skipped_no_valid_answers'] += 1
            continue

        # Filter out answers with infinite log-likelihood
        finite_answers = [
            ans for ans in valid_answers
            if ans['log_likelihood'] != float('-inf') and ans['log_likelihood'] != float('inf')
        ]

        # If all answers have infinite log-likelihood, use valid answers but mark it
        if not finite_answers:
            stats['skipped_all_inf'] += 1
            # Still add a random valid answer to keep data
            random_answer = random.choice(valid_answers)
            random_data.append({
                "instruction": instruction,
                "input": question,
                "output": random_answer['answer']
            })
            continue

        # Sort by log-likelihood (higher is better)
        sorted_answers = sorted(finite_answers, key=lambda x: x['log_likelihood'], reverse=True)

        # Select answers
        highest_answer = sorted_answers[0]  # Highest log-likelihood
        lowest_answer = sorted_answers[-1]   # Lowest log-likelihood
        random_answer = random.choice(finite_answers)  # Random

        # Add to output lists
        highest_data.append({
            "instruction": instruction,
            "input": question,
            "output": highest_answer['answer']
        })

        lowest_data.append({
            "instruction": instruction,
            "input": question,
            "output": lowest_answer['answer']
        })

        random_data.append({
            "instruction": instruction,
            "input": question,
            "output": random_answer['answer']
        })

        stats['processed'] += 1

        if (q_idx + 1) % 1000 == 0:
            print(f"  Processed {q_idx + 1}/{len(data)} questions...")

    # Save outputs
    print("\n" + "-" * 80)
    print("Saving output files...")
    print("-" * 80)

    print(f"\nSaving highest likelihood answers to {output_highest}...")
    with open(output_highest, 'w', encoding='utf-8') as f:
        json.dump(highest_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_highest) / (1024 * 1024)
    print(f"  Saved {len(highest_data)} entries ({file_size:.2f} MB)")

    print(f"\nSaving lowest likelihood answers to {output_lowest}...")
    with open(output_lowest, 'w', encoding='utf-8') as f:
        json.dump(lowest_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_lowest) / (1024 * 1024)
    print(f"  Saved {len(lowest_data)} entries ({file_size:.2f} MB)")

    print(f"\nSaving random answers to {output_random}...")
    with open(output_random, 'w', encoding='utf-8') as f:
        json.dump(random_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_random) / (1024 * 1024)
    print(f"  Saved {len(random_data)} entries ({file_size:.2f} MB)")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total questions: {stats['total_questions']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (no valid answers): {stats['skipped_no_valid_answers']}")
    print(f"Skipped (all infinite): {stats['skipped_all_inf']}")
    print(f"\nOutput entries:")
    print(f"  Highest likelihood: {len(highest_data)}")
    print(f"  Lowest likelihood: {len(lowest_data)}")
    print(f"  Random: {len(random_data)}")

    # Show sample entries
    if highest_data:
        print("\n" + "-" * 80)
        print("SAMPLE ENTRIES")
        print("-" * 80)

        print("\nHighest likelihood sample:")
        sample = highest_data[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")

        print("\nLowest likelihood sample:")
        sample = lowest_data[0]
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")

        print("\nRandom sample:")
        sample = random_data[0]
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return stats

def analyze_selections(
    highest_file: str,
    lowest_file: str,
    random_file: str,
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
    with open(highest_file, 'r', encoding='utf-8') as f:
        highest_data = json.load(f)

    with open(lowest_file, 'r', encoding='utf-8') as f:
        lowest_data = json.load(f)

    with open(random_file, 'r', encoding='utf-8') as f:
        random_data = json.load(f)

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
    print("\nSource distribution for highest likelihood:")
    highest_sources = count_sources(highest_data, original_data)
    for source, count in sorted(highest_sources.items()):
        pct = count / len(highest_data) * 100 if highest_data else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print("\nSource distribution for lowest likelihood:")
    lowest_sources = count_sources(lowest_data, original_data)
    for source, count in sorted(lowest_sources.items()):
        pct = count / len(lowest_data) * 100 if lowest_data else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print("\nSource distribution for random:")
    random_sources = count_sources(random_data, original_data)
    for source, count in sorted(random_sources.items()):
        pct = count / len(random_data) * 100 if random_data else 0
        print(f"  {source}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Configuration
    input_file = "/workspace/SLM_REASONING/data/math12K_merged_answers_with_Qwen2.5_Math_7B_loglikelihood.json"
    output_dir = "/workspace/SLM_REASONING/data"

    output_highest = f"{output_dir}/math12K_highest_likelihood.json"
    output_lowest = f"{output_dir}/math12K_lowest_likelihood.json"
    output_random = f"{output_dir}/math12K_random_selection.json"

    print(f"\nConfiguration:")
    print(f"  Input: {input_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output files:")
    print(f"    - {output_highest}")
    print(f"    - {output_lowest}")
    print(f"    - {output_random}")
    print()

    # Select answers
    stats = select_answers(
        input_file=input_file,
        output_highest=output_highest,
        output_lowest=output_lowest,
        output_random=output_random
    )

    # Analyze source distribution
    print("\nStarting source analysis (this may take a while for large files)...")
    analyze_selections(
        highest_file=output_highest,
        lowest_file=output_lowest,
        random_file=output_random,
        original_file=input_file
    )

    print("\nDone!")
