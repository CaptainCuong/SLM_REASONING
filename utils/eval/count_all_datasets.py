#!/usr/bin/env python3
"""
Utility script to count accuracy of all datasets in a folder.
Assumes folder structure: eval/outputs/workspace/model/Qwen_Math_high/<dataset_name>/*.jsonl
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def count_correct_in_file(file_path):
    """
    Count correct answers in a single JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        dict with counts and statistics
    """
    data = load_jsonl(file_path)

    total = len(data)
    correct = sum(1 for entry in data if entry.get('is_correct', False))
    incorrect = total - correct

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy
    }


def count_dataset_accuracy(dataset_folder):
    """
    Count accuracy for all JSONL files in a dataset folder.

    Args:
        dataset_folder: Path to dataset folder containing JSONL files

    Returns:
        dict with aggregated statistics
    """
    dataset_folder = Path(dataset_folder)
    jsonl_files = list(dataset_folder.glob("*.jsonl"))

    if not jsonl_files:
        return None

    total_questions = 0
    total_correct = 0

    for jsonl_file in jsonl_files:
        try:
            result = count_correct_in_file(str(jsonl_file))
            total_questions += result['total']
            total_correct += result['correct']
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")

    accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    return {
        'dataset': dataset_folder.name,
        'total': total_questions,
        'correct': total_correct,
        'incorrect': total_questions - total_correct,
        'accuracy': accuracy,
        'num_files': len(jsonl_files)
    }


def count_all_datasets_in_folder(base_folder):
    """
    Count accuracy for all datasets in a base folder.

    Args:
        base_folder: Path to base folder containing dataset subfolders

    Returns:
        List of result dictionaries for each dataset
    """
    base_folder = Path(base_folder)

    if not base_folder.exists():
        print(f"Error: Folder {base_folder} does not exist!")
        return []

    # Get all subdirectories (each represents a dataset)
    dataset_folders = [d for d in base_folder.iterdir() if d.is_dir()]

    results = []
    for dataset_folder in sorted(dataset_folders):
        result = count_dataset_accuracy(dataset_folder)
        if result:
            results.append(result)

    return results


def print_results_table(results):
    """Print results in a formatted table."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "="*90)
    print(f"{'Dataset':<25} {'Total':<10} {'Correct':<10} {'Incorrect':<12} {'Accuracy':<10}")
    print("="*90)

    total_sum = 0
    correct_sum = 0

    for result in results:
        print(f"{result['dataset']:<25} {result['total']:<10} {result['correct']:<10} "
              f"{result['incorrect']:<12} {result['accuracy']:>8.2f}%")

        total_sum += result['total']
        correct_sum += result['correct']

    print("="*90)

    if len(results) > 1:
        overall_accuracy = (correct_sum / total_sum * 100) if total_sum > 0 else 0
        print(f"{'OVERALL':<25} {total_sum:<10} {correct_sum:<10} "
              f"{total_sum - correct_sum:<12} {overall_accuracy:>8.2f}%")
        print("="*90)


def save_results_to_json(results, output_file):
    """Save results to a JSON file."""
    # Calculate overall statistics
    total_sum = sum(r['total'] for r in results)
    correct_sum = sum(r['correct'] for r in results)
    overall_accuracy = (correct_sum / total_sum * 100) if total_sum > 0 else 0

    output_data = {
        'datasets': results,
        'summary': {
            'total_datasets': len(results),
            'total_questions': total_sum,
            'total_correct': correct_sum,
            'total_incorrect': total_sum - correct_sum,
            'overall_accuracy': overall_accuracy
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Count accuracy of all datasets in a folder"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="eval/outputs/model/Qwen_Math_high/checkpoint-10545/",
        help="Base folder containing dataset subfolders (default: eval/outputs/model/Qwen_Math_high/checkpoint-5550/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results.json",
        help="Optional JSON file to save results"
    )

    args = parser.parse_args()

    # Count accuracy for all datasets
    results = count_all_datasets_in_folder(args.folder)

    if not results:
        print(f"No datasets found in {args.folder}")
        return

    # Print results table
    print(f"\nAnalyzing datasets in: {args.folder}")
    print_results_table(results)

    # Save to JSON if requested
    if args.output:
        save_results_to_json(results, args.output)


if __name__ == "__main__":
    main()
