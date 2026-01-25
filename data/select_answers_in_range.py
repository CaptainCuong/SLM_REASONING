#!/usr/bin/env python3
"""
Select the highest likelihood answer for each question, then filter to keep
only answers with log-likelihood in the range [-100, -50].

This targets questions where the best answer has moderate log-likelihood,
potentially indicating medium difficulty problems.
"""

import json
import random
from typing import List, Dict, Tuple
import os


def select_answers_in_range(
    input_file: str,
    output_file: str,
    min_llh: float = -100.0,
    max_llh: float = -50.0,
    instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
):
    """
    Select highest likelihood answer per question, then filter by log-likelihood range.

    Strategy:
    1. For each question, select the answer with highest log-likelihood
    2. Filter to keep only answers with log-likelihood in [min_llh, max_llh]

    Args:
        input_file: Path to merged answers with log-likelihoods
        output_file: Output file for filtered samples
        min_llh: Minimum log-likelihood (default: -100.0)
        max_llh: Maximum log-likelihood (default: -50.0)
        instruction: Instruction text to use for all entries
    """
    print("=" * 80)
    print(f"SELECTING ANSWERS IN LOG-LIKELIHOOD RANGE [{min_llh}, {max_llh}]")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} questions")

    # Statistics
    stats = {
        'total_questions': len(data),
        'questions_with_valid_answers': 0,
        'total_valid_answers': 0,
        'questions_in_range': 0,
        'questions_below_range': 0,
        'questions_above_range': 0
    }

    # Collect best answers for each question
    print("\nProcessing questions...")
    all_best_answers = []

    for q_idx, question_item in enumerate(data):
        question = question_item['question']
        answers = question_item['answers']

        # Filter valid answers (with non-None, finite log-likelihood)
        valid_answers = [
            ans for ans in answers
            if ans.get('log_likelihood') is not None
            and ans['log_likelihood'] != float('-inf')
            and ans['log_likelihood'] != float('inf')
        ]

        if valid_answers:
            stats['questions_with_valid_answers'] += 1
            stats['total_valid_answers'] += len(valid_answers)

            # Find the best answer for this question (highest log-likelihood)
            best_answer = max(valid_answers, key=lambda x: x['log_likelihood'])
            best_llh = best_answer['log_likelihood']

            # Categorize by range
            if min_llh <= best_llh <= max_llh:
                stats['questions_in_range'] += 1
                all_best_answers.append({
                    'question': question,
                    'answer': best_answer['answer'],
                    'log_likelihood': best_llh,
                    'source': best_answer.get('source', 'unknown')
                })
            elif best_llh < min_llh:
                stats['questions_below_range'] += 1
            else:  # best_llh > max_llh
                stats['questions_above_range'] += 1

        if (q_idx + 1) % 1000 == 0:
            print(f"  Processed {q_idx + 1}/{len(data)} questions...")

    print(f"\nFound {len(all_best_answers)} questions with answers in range [{min_llh}, {max_llh}]")

    # Sort by log-likelihood (descending - highest first)
    print(f"\nSorting samples by log-likelihood...")
    sorted_answers = sorted(all_best_answers, key=lambda x: x['log_likelihood'], reverse=True)

    # Prepare output in identity.json format
    output_data = []
    for sample in sorted_answers:
        output_data.append({
            "instruction": instruction,
            "input": sample['question'],
            "output": sample['answer']
        })

    # Save output
    print("\n" + "-" * 80)
    print("Saving output file...")
    print("-" * 80)

    print(f"\nSaving {len(output_data)} samples to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {len(output_data)} entries ({file_size:.2f} MB)")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total questions in input: {stats['total_questions']}")
    print(f"Questions with valid answers: {stats['questions_with_valid_answers']}")
    print(f"Total valid answers: {stats['total_valid_answers']}")
    print(f"\nLog-likelihood range distribution:")
    print(f"  Below range (< {min_llh}): {stats['questions_below_range']} "
          f"({stats['questions_below_range']/stats['questions_with_valid_answers']*100:.1f}%)")
    print(f"  In range [{min_llh}, {max_llh}]: {stats['questions_in_range']} "
          f"({stats['questions_in_range']/stats['questions_with_valid_answers']*100:.1f}%)")
    print(f"  Above range (> {max_llh}): {stats['questions_above_range']} "
          f"({stats['questions_above_range']/stats['questions_with_valid_answers']*100:.1f}%)")

    # Log-likelihood statistics for selected samples
    if sorted_answers:
        log_likelihoods = [s['log_likelihood'] for s in sorted_answers]
        print(f"\nLog-likelihood statistics for selected samples:")
        print(f"  Maximum: {max(log_likelihoods):.4f}")
        print(f"  Minimum: {min(log_likelihoods):.4f}")
        print(f"  Average: {sum(log_likelihoods) / len(log_likelihoods):.4f}")
        print(f"  Median: {sorted(log_likelihoods)[len(log_likelihoods)//2]:.4f}")

        # Show distribution in sub-ranges
        range_size = (max_llh - min_llh) / 5
        print(f"\nDistribution within range (5 bins):")
        for i in range(5):
            bin_min = min_llh + i * range_size
            bin_max = min_llh + (i + 1) * range_size
            count = sum(1 for ll in log_likelihoods if bin_min <= ll < bin_max)
            if i == 4:  # Last bin includes upper bound
                count = sum(1 for ll in log_likelihoods if bin_min <= ll <= bin_max)
            pct = count / len(log_likelihoods) * 100
            print(f"  [{bin_min:.1f}, {bin_max:.1f}]: {count} ({pct:.1f}%)")

    # Source distribution
    if sorted_answers:
        source_counts = {}
        for sample in sorted_answers:
            source = sample['source']
            source_counts[source] = source_counts.get(source, 0) + 1

        print(f"\nSource distribution:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(sorted_answers) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")

    # Show sample entries
    if output_data:
        print("\n" + "-" * 80)
        print("SAMPLE ENTRIES")
        print("-" * 80)

        print("\nFirst sample (highest log-likelihood in range):")
        sample = output_data[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Input: {sample['input'][:150]}...")
        print(f"  Output: {sample['output'][:150]}...")
        print(f"  Log-likelihood: {sorted_answers[0]['log_likelihood']:.4f}")

        if len(output_data) > 1:
            print(f"\nLast sample (lowest log-likelihood in range):")
            sample = output_data[-1]
            print(f"  Input: {sample['input'][:150]}...")
            print(f"  Output: {sample['output'][:150]}...")
            print(f"  Log-likelihood: {sorted_answers[-1]['log_likelihood']:.4f}")

        if len(output_data) > 10:
            print(f"\nMiddle sample (rank {len(output_data)//2}):")
            mid_idx = len(output_data) // 2
            sample = output_data[mid_idx]
            print(f"  Input: {sample['input'][:150]}...")
            print(f"  Output: {sample['output'][:150]}...")
            print(f"  Log-likelihood: {sorted_answers[mid_idx]['log_likelihood']:.4f}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return stats, sorted_answers


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Configuration
    input_file = "/home/cuongdc/SLM_REASONING/data/math12K_merged_answers_with_Qwen2.5_Math_7B_loglikelihood.json"
    output_dir = "/home/cuongdc/SLM_REASONING/data"

    # Log-likelihood range
    min_llh = -100.0
    max_llh = -60.0

    output_file = f"{output_dir}/math12K_llh_range_{int(abs(min_llh))}_{int(abs(max_llh))}.json"

    print(f"\nConfiguration:")
    print(f"  Input: {input_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Log-likelihood range: [{min_llh}, {max_llh}]")
    print(f"  Output file: {output_file}")
    print()

    # Select answers in range
    stats, selected_answers = select_answers_in_range(
        input_file=input_file,
        output_file=output_file,
        min_llh=min_llh,
        max_llh=max_llh
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
