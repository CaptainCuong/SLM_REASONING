#!/usr/bin/env python3
"""
Select samples with the highest likelihood answer per question,
then choose the top 1K with the LOWEST log-likelihood from those selected.

Strategy:
1. For each question, select the answer with highest log-likelihood
2. Among all those best answers, select the 1K samples with LOWEST likelihood

Output format matches identity.json structure.
"""

import json
import random
from typing import List, Dict, Tuple
import os

def select_top_samples(
    input_file: str,
    output_file: str,
    top_k: int = 1000,
    instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
):
    """
    Select top K samples with LOWEST log-likelihood among best answers per question.

    Strategy:
    1. For each question, select the answer with highest log-likelihood
    2. Among all those best answers, select the 1K samples with LOWEST likelihood

    Args:
        input_file: Path to merged answers with log-likelihoods
        output_file: Output file for top K unique question samples
        top_k: Number of top samples to select (default: 1000)
        instruction: Instruction text to use for all entries
    """
    print("=" * 80)
    print(f"SELECTING TOP {top_k} LOWEST LIKELIHOOD SAMPLES")
    print("(from highest likelihood answer per question)")
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
        'selected_samples': 0
    }

    # Collect all valid answers with their questions
    print("\nCollecting all valid answers...")
    all_answers = []

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

            all_answers.append({
                'question': question,
                'answer': best_answer['answer'],
                'log_likelihood': best_answer['log_likelihood'],
                'source': best_answer.get('source', 'unknown')
            })

        if (q_idx + 1) % 1000 == 0:
            print(f"  Processed {q_idx + 1}/{len(data)} questions...")

    print(f"\nFound {len(all_answers)} questions with valid answers")
    print(f"Average valid answers per question: {stats['total_valid_answers'] / stats['questions_with_valid_answers']:.2f}")

    # Sort by log-likelihood (LOWEST first - ascending order)
    print(f"\nSorting samples by log-likelihood (ascending)...")
    sorted_answers = sorted(all_answers, key=lambda x: x['log_likelihood'], reverse=False)

    # Select top K with LOWEST likelihood
    print(f"Selecting top {top_k} samples with LOWEST likelihood...")
    top_samples = sorted_answers[:top_k]
    stats['selected_samples'] = len(top_samples)

    # Prepare output in identity.json format
    output_data = []
    for sample in top_samples:
        output_data.append({
            "instruction": instruction,
            "input": sample['question'],
            "output": sample['answer']
        })

    # Save output
    print("\n" + "-" * 80)
    print("Saving output file...")
    print("-" * 80)

    print(f"\nSaving top {top_k} samples to {output_file}...")
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
    print(f"Selected top samples: {stats['selected_samples']}")

    # Log-likelihood statistics
    if top_samples:
        log_likelihoods = [s['log_likelihood'] for s in top_samples]
        print(f"\nLog-likelihood statistics for selected samples:")
        print(f"  Maximum: {max(log_likelihoods):.4f}")
        print(f"  Minimum: {min(log_likelihoods):.4f}")
        print(f"  Average: {sum(log_likelihoods) / len(log_likelihoods):.4f}")

        # Show top 10 and bottom 10 log-likelihoods
        print(f"\n  Top 10 log-likelihoods:")
        for i, ll in enumerate(log_likelihoods[:10], 1):
            print(f"    {i}. {ll:.4f}")

        print(f"\n  Bottom 10 log-likelihoods (of selected top {top_k}):")
        for i, ll in enumerate(log_likelihoods[-10:], len(log_likelihoods) - 9):
            print(f"    {i}. {ll:.4f}")

    # Source distribution
    if top_samples:
        source_counts = {}
        for sample in top_samples:
            source = sample['source']
            source_counts[source] = source_counts.get(source, 0) + 1

        print(f"\nSource distribution in top {top_k} samples:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(top_samples) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")

    # Show sample entries
    if output_data:
        print("\n" + "-" * 80)
        print("SAMPLE ENTRIES")
        print("-" * 80)

        print("\nFirst sample (lowest log-likelihood):")
        sample = output_data[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Input: {sample['input'][:150]}...")
        print(f"  Output: {sample['output'][:150]}...")
        print(f"  Log-likelihood: {top_samples[0]['log_likelihood']:.4f}")

        if len(output_data) > 1:
            print(f"\n{top_k}th sample (highest in selected {top_k}):")
            sample = output_data[-1]
            print(f"  Input: {sample['input'][:150]}...")
            print(f"  Output: {sample['output'][:150]}...")
            print(f"  Log-likelihood: {top_samples[-1]['log_likelihood']:.4f}")

        if len(output_data) > 100:
            print(f"\nMiddle sample (rank {len(output_data)//2}):")
            mid_idx = len(output_data) // 2
            sample = output_data[mid_idx]
            print(f"  Input: {sample['input'][:150]}...")
            print(f"  Output: {sample['output'][:150]}...")
            print(f"  Log-likelihood: {top_samples[mid_idx]['log_likelihood']:.4f}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return stats, top_samples

def analyze_distribution(
    selected_file: str,
    original_file: str,
    top_k: int = 1000
):
    """
    Analyze the distribution of selected samples vs all available samples.
    """
    print("\n" + "=" * 80)
    print("ANALYZING LOG-LIKELIHOOD DISTRIBUTION")
    print("=" * 80)

    # Load original data
    print("Loading original data...")
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # Collect all valid log-likelihoods from original
    all_log_likelihoods = []
    for question_item in original_data:
        for ans in question_item['answers']:
            ll = ans.get('log_likelihood')
            if ll is not None and ll != float('-inf') and ll != float('inf'):
                all_log_likelihoods.append(ll)

    all_log_likelihoods.sort(reverse=True)

    print(f"\nTotal valid answers in dataset: {len(all_log_likelihoods)}")
    print(f"Selected top samples: {top_k}")
    print(f"Percentile: {top_k / len(all_log_likelihoods) * 100:.2f}%")

    print(f"\nOverall log-likelihood statistics:")
    print(f"  Maximum: {all_log_likelihoods[0]:.4f}")
    print(f"  Minimum: {all_log_likelihoods[-1]:.4f}")
    print(f"  Median: {all_log_likelihoods[len(all_log_likelihoods)//2]:.4f}")
    print(f"  Average: {sum(all_log_likelihoods) / len(all_log_likelihoods):.4f}")

    # Show percentile boundaries
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentile boundaries:")
    for p in percentiles:
        idx = int(len(all_log_likelihoods) * p / 100) - 1
        idx = max(0, min(idx, len(all_log_likelihoods) - 1))
        print(f"  {p}th percentile: {all_log_likelihoods[idx]:.4f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Configuration
    input_file = "/home/cuongdc/SLM_REASONING/data/math12K_merged_answers_with_Qwen2.5_Math_7B_loglikelihood.json"
    output_dir = "/home/cuongdc/SLM_REASONING/data"
    top_k = 1000

    output_file = f"{output_dir}/math12K_top{top_k}_lowest_likelihood_unique.json"

    print(f"\nConfiguration:")
    print(f"  Input: {input_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Top K samples: {top_k}")
    print(f"  Output file: {output_file}")
    print()

    # Select top K samples
    stats, top_samples = select_top_samples(
        input_file=input_file,
        output_file=output_file,
        top_k=top_k
    )

    # Analyze distribution
    print("\nAnalyzing distribution...")
    analyze_distribution(
        selected_file=output_file,
        original_file=input_file,
        top_k=top_k
    )

    print("\nDone!")
