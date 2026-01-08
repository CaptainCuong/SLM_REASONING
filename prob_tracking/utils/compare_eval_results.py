#!/usr/bin/env python3
"""
Compare evaluation results between base model and fine-tuned checkpoint.

This script identifies samples that were incorrect in the base model but
became correct after fine-tuning.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import os


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def find_jsonl_files(directory: str) -> List[str]:
    """Find all JSONL files in directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return [str(f) for f in dir_path.glob("*.jsonl")]


def extract_question_id(sample: Dict[str, Any]) -> str:
    """
    Extract unique identifier for a question.

    Tries multiple fields: idx, id, question_id, or uses question text.
    """
    for field in ['idx', 'id', 'question_id', 'qid']:
        if field in sample:
            return str(sample[field])

    # Fallback to question text
    if 'question' in sample:
        return sample['question'][:100]  # First 100 chars

    return str(hash(str(sample)))


def is_correct(sample: Dict[str, Any]) -> bool:
    """
    Determine if a sample is correct.

    Checks multiple possible fields for correctness.
    """
    # Check common correctness fields
    for field in ['correct', 'is_correct', 'score', 'accuracy']:
        if field in sample:
            value = sample[field]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value > 0

    # Check if prediction matches ground truth
    if 'prediction' in sample and 'answer' in sample:
        return sample['prediction'] == sample['answer']

    if 'pred' in sample and 'answer' in sample:
        return sample['pred'] == sample['answer']

    # Default to False if we can't determine
    return False


def compare_results(
    base_results: List[Dict[str, Any]],
    finetuned_results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare base and fine-tuned results.

    Returns:
        Dictionary with categories:
        - incorrect_to_correct: Was wrong, now correct
        - correct_to_incorrect: Was correct, now wrong
        - stayed_correct: Correct in both
        - stayed_incorrect: Incorrect in both
    """
    # Build lookup by question ID
    base_lookup = {extract_question_id(s): s for s in base_results}
    finetuned_lookup = {extract_question_id(s): s for s in finetuned_results}

    # Find common questions
    common_ids = set(base_lookup.keys()) & set(finetuned_lookup.keys())

    categories = {
        'incorrect_to_correct': [],
        'correct_to_incorrect': [],
        'stayed_correct': [],
        'stayed_incorrect': []
    }

    for qid in common_ids:
        base_sample = base_lookup[qid]
        finetuned_sample = finetuned_lookup[qid]

        base_correct = is_correct(base_sample)
        finetuned_correct = is_correct(finetuned_sample)

        comparison = {
            'question_id': qid,
            'base_sample': base_sample,
            'finetuned_sample': finetuned_sample,
            'base_correct': base_correct,
            'finetuned_correct': finetuned_correct
        }

        if not base_correct and finetuned_correct:
            categories['incorrect_to_correct'].append(comparison)
        elif base_correct and not finetuned_correct:
            categories['correct_to_incorrect'].append(comparison)
        elif base_correct and finetuned_correct:
            categories['stayed_correct'].append(comparison)
        else:
            categories['stayed_incorrect'].append(comparison)

    return categories


def print_summary(categories: Dict[str, List[Dict[str, Any]]]):
    """Print summary statistics."""
    total = sum(len(v) for v in categories.values())

    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(f"\nTotal compared samples: {total}")
    print(f"\nIncorrect → Correct: {len(categories['incorrect_to_correct'])} "
          f"({len(categories['incorrect_to_correct'])/total*100:.1f}%)")
    print(f"Correct → Incorrect: {len(categories['correct_to_incorrect'])} "
          f"({len(categories['correct_to_incorrect'])/total*100:.1f}%)")
    print(f"Stayed Correct:      {len(categories['stayed_correct'])} "
          f"({len(categories['stayed_correct'])/total*100:.1f}%)")
    print(f"Stayed Incorrect:    {len(categories['stayed_incorrect'])} "
          f"({len(categories['stayed_incorrect'])/total*100:.1f}%)")

    # Calculate accuracy
    base_correct = len(categories['stayed_correct']) + len(categories['correct_to_incorrect'])
    finetuned_correct = len(categories['stayed_correct']) + len(categories['incorrect_to_correct'])

    print(f"\nBase Model Accuracy:      {base_correct}/{total} ({base_correct/total*100:.2f}%)")
    print(f"Fine-tuned Model Accuracy: {finetuned_correct}/{total} ({finetuned_correct/total*100:.2f}%)")
    print(f"Improvement: {finetuned_correct - base_correct} samples "
          f"({(finetuned_correct - base_correct)/total*100:.2f}%)")
    print("="*80)


def save_detailed_results(categories: Dict[str, List[Dict[str, Any]]], output_path: str):
    """Save detailed comparison results to JSON."""
    # Create a more readable format
    output = {
        'summary': {
            'total_samples': sum(len(v) for v in categories.values()),
            'incorrect_to_correct': len(categories['incorrect_to_correct']),
            'correct_to_incorrect': len(categories['correct_to_incorrect']),
            'stayed_correct': len(categories['stayed_correct']),
            'stayed_incorrect': len(categories['stayed_incorrect'])
        },
        'samples': {}
    }

    for category, samples in categories.items():
        output['samples'][category] = []
        for sample in samples:
            output['samples'][category].append({
                'question_id': sample['question_id'],
                'question': sample['base_sample'].get('question', 'N/A'),
                'base_answer': sample['base_sample'].get('answer', 'N/A'),
                'base_prediction': sample['base_sample'].get('prediction',
                                                            sample['base_sample'].get('pred', 'N/A')),
                'finetuned_answer': sample['finetuned_sample'].get('answer', 'N/A'),
                'finetuned_prediction': sample['finetuned_sample'].get('prediction',
                                                                       sample['finetuned_sample'].get('pred', 'N/A')),
                'base_correct': sample['base_correct'],
                'finetuned_correct': sample['finetuned_correct']
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between base and fine-tuned models"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="eval/outputs/Qwen/Qwen2.5-Math-7B/cn_math_2024",
        help="Directory containing base model results"
    )
    parser.add_argument(
        "--finetuned_dir",
        type=str,
        default="eval/outputs/model/Qwen_Math_high/checkpoint-555/cn_math_2024",
        help="Directory containing fine-tuned model results"
    )
    parser.add_argument(
        "--base_file",
        type=str,
        default=None,
        help="Specific JSONL file for base model (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--finetuned_file",
        type=str,
        default=None,
        help="Specific JSONL file for fine-tuned model (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="temp",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="comparison_results.json",
        help="Output filename"
    )

    args = parser.parse_args()

    # Find or use specified files
    if args.base_file:
        base_file = args.base_file
    else:
        base_files = find_jsonl_files(args.base_dir)
        if not base_files:
            print(f"Error: No JSONL files found in {args.base_dir}")
            return
        base_file = base_files[0]
        print(f"Using base file: {base_file}")

    if args.finetuned_file:
        finetuned_file = args.finetuned_file
    else:
        finetuned_files = find_jsonl_files(args.finetuned_dir)
        if not finetuned_files:
            print(f"Error: No JSONL files found in {args.finetuned_dir}")
            return
        finetuned_file = finetuned_files[0]
        print(f"Using fine-tuned file: {finetuned_file}")

    # Load results
    print("\nLoading base model results...")
    base_results = load_jsonl(base_file)
    print(f"Loaded {len(base_results)} samples")

    print("\nLoading fine-tuned model results...")
    finetuned_results = load_jsonl(finetuned_file)
    print(f"Loaded {len(finetuned_results)} samples")

    # Compare results
    print("\nComparing results...")
    categories = compare_results(base_results, finetuned_results)

    # Print summary
    print_summary(categories)

    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    save_detailed_results(categories, output_path)

    # Print sample of incorrect → correct
    print("\n" + "="*80)
    print("Sample: Incorrect → Correct (first 3)")
    print("="*80)
    for i, sample in enumerate(categories['incorrect_to_correct'][:3]):
        print(f"\n{i+1}. Question ID: {sample['question_id']}")
        print(f"   Question: {sample['base_sample'].get('question', 'N/A')[:100]}...")
        print(f"   Ground Truth: {sample['base_sample'].get('answer', 'N/A')}")
        print(f"   Base Prediction: {sample['base_sample'].get('prediction', sample['base_sample'].get('pred', 'N/A'))}")
        print(f"   Fine-tuned Prediction: {sample['finetuned_sample'].get('prediction', sample['finetuned_sample'].get('pred', 'N/A'))}")
    print("="*80)


if __name__ == "__main__":
    main()
