#!/usr/bin/env python3
"""
Compare evaluation results between two model checkpoints or directories.

This script compares JSONL evaluation results from two specified directories
and identifies samples that changed correctness between the two models.
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
    model1_results: List[Dict[str, Any]],
    model2_results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare results from two models.

    Returns:
        Dictionary with categories:
        - incorrect_to_correct: Was wrong in model1, correct in model2
        - correct_to_incorrect: Was correct in model1, wrong in model2
        - stayed_correct: Correct in both
        - stayed_incorrect: Incorrect in both
    """
    # Build lookup by question ID
    model1_lookup = {extract_question_id(s): s for s in model1_results}
    model2_lookup = {extract_question_id(s): s for s in model2_results}

    # Find common questions
    common_ids = set(model1_lookup.keys()) & set(model2_lookup.keys())

    categories = {
        'incorrect_to_correct': [],
        'correct_to_incorrect': [],
        'stayed_correct': [],
        'stayed_incorrect': []
    }

    for qid in common_ids:
        model1_sample = model1_lookup[qid]
        model2_sample = model2_lookup[qid]

        model1_correct = is_correct(model1_sample)
        model2_correct = is_correct(model2_sample)

        comparison = {
            'question_id': qid,
            'model1_sample': model1_sample,
            'model2_sample': model2_sample,
            'model1_correct': model1_correct,
            'model2_correct': model2_correct
        }

        if not model1_correct and model2_correct:
            categories['incorrect_to_correct'].append(comparison)
        elif model1_correct and not model2_correct:
            categories['correct_to_incorrect'].append(comparison)
        elif model1_correct and model2_correct:
            categories['stayed_correct'].append(comparison)
        else:
            categories['stayed_incorrect'].append(comparison)

    return categories


def print_summary(categories: Dict[str, List[Dict[str, Any]]], model1_name: str = "Model 1", model2_name: str = "Model 2"):
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
    model1_correct = len(categories['stayed_correct']) + len(categories['correct_to_incorrect'])
    model2_correct = len(categories['stayed_correct']) + len(categories['incorrect_to_correct'])

    print(f"\n{model1_name} Accuracy: {model1_correct}/{total} ({model1_correct/total*100:.2f}%)")
    print(f"{model2_name} Accuracy: {model2_correct}/{total} ({model2_correct/total*100:.2f}%)")

    diff = model2_correct - model1_correct
    if diff > 0:
        print(f"Improvement: +{diff} samples (+{(diff/total)*100:.2f}%)")
    elif diff < 0:
        print(f"Regression: {diff} samples ({(diff/total)*100:.2f}%)")
    else:
        print(f"No change: Same accuracy")
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
                'question': sample['model1_sample'].get('question', 'N/A'),
                'answer': sample['model1_sample'].get('answer', 'N/A'),
                'model1_prediction': sample['model1_sample'].get('prediction',
                                                                sample['model1_sample'].get('pred', 'N/A')),
                'model2_prediction': sample['model2_sample'].get('prediction',
                                                                 sample['model2_sample'].get('pred', 'N/A')),
                'model1_correct': sample['model1_correct'],
                'model2_correct': sample['model2_correct']
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between two model checkpoints"
    )
    parser.add_argument(
        "--dir1",
        type=str,
        default="eval/outputs/Qwen/Qwen2.5-Math-7B/amc/",
        help="Directory containing first model results (default: checkpoint-555/amc)"
    )
    parser.add_argument(
        "--dir2",
        type=str,
        default="eval/outputs/cuongdc/Qwen_Math_high/checkpoint-555/amc/",
        help="Directory containing second model results (default: checkpoint-1110/amc)"
    )
    parser.add_argument(
        "--file1",
        type=str,
        default=None,
        help="Specific JSONL file for model 1 (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--file2",
        type=str,
        default=None,
        help="Specific JSONL file for model 2 (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--name1",
        type=str,
        default=None,
        help="Display name for model 1 (default: extracted from directory)"
    )
    parser.add_argument(
        "--name2",
        type=str,
        default=None,
        help="Display name for model 2 (default: extracted from directory)"
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

    # Extract model names from directories if not provided
    if args.name1 is None:
        args.name1 = Path(args.dir1).parent.name if Path(args.dir1).parent.name else "Model 1"
    if args.name2 is None:
        args.name2 = Path(args.dir2).parent.name if Path(args.dir2).parent.name else "Model 2"

    print(f"\n{'='*80}")
    print(f"Comparing: {args.name1} vs {args.name2}")
    print(f"{'='*80}")

    # Find or use specified files
    if args.file1:
        file1 = args.file1
    else:
        files1 = find_jsonl_files(args.dir1)
        if not files1:
            print(f"Error: No JSONL files found in {args.dir1}")
            return
        file1 = files1[0]
        print(f"\nModel 1 file: {file1}")

    if args.file2:
        file2 = args.file2
    else:
        files2 = find_jsonl_files(args.dir2)
        if not files2:
            print(f"Error: No JSONL files found in {args.dir2}")
            return
        file2 = files2[0]
        print(f"Model 2 file: {file2}")

    # Load results
    print(f"\nLoading {args.name1} results...")
    model1_results = load_jsonl(file1)
    print(f"  Loaded {len(model1_results)} samples")

    print(f"\nLoading {args.name2} results...")
    model2_results = load_jsonl(file2)
    print(f"  Loaded {len(model2_results)} samples")

    # Compare results
    print("\nComparing results...")
    categories = compare_results(model1_results, model2_results)

    # Print summary
    print_summary(categories, args.name1, args.name2)

    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    save_detailed_results(categories, output_path)

    # Print sample of incorrect → correct
    if categories['incorrect_to_correct']:
        print("\n" + "="*80)
        print(f"Sample: Incorrect → Correct (first 3)")
        print(f"({args.name1} wrong → {args.name2} correct)")
        print("="*80)
        for i, sample in enumerate(categories['incorrect_to_correct'][:3]):
            print(f"\n{i+1}. Question ID: {sample['question_id']}")
            print(f"   Question: {sample['model1_sample'].get('question', 'N/A')[:100]}...")
            print(f"   Ground Truth: {sample['model1_sample'].get('answer', 'N/A')}")
            print(f"   {args.name1} Prediction: {sample['model1_sample'].get('prediction', sample['model1_sample'].get('pred', 'N/A'))}")
            print(f"   {args.name2} Prediction: {sample['model2_sample'].get('prediction', sample['model2_sample'].get('pred', 'N/A'))}")
        print("="*80)


if __name__ == "__main__":
    main()
