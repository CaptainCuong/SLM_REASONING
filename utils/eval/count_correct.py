#!/usr/bin/env python3
"""
Utility script to count correct answers from JSONL files in eval/outputs folder.
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
        'file': file_path,
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy
    }


def count_correct_in_folder(folder_path, pattern="*.jsonl", recursive=True):
    """
    Count correct answers in all JSONL files in a folder.

    Args:
        folder_path: Path to folder containing JSONL files
        pattern: File pattern to match (default: *.jsonl)
        recursive: Whether to search recursively (default: True)

    Returns:
        List of result dictionaries
    """
    folder = Path(folder_path)

    if recursive:
        files = list(folder.rglob(pattern))
    else:
        files = list(folder.glob(pattern))

    results = []
    for file_path in sorted(files):
        try:
            result = count_correct_in_file(str(file_path))
            results.append(result)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results


def print_results(results, verbose=False):
    """Print results in a formatted table."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "="*100)
    print(f"{'File':<70} {'Total':<8} {'Correct':<8} {'Incorrect':<10} {'Accuracy':<10}")
    print("="*100)

    total_sum = 0
    correct_sum = 0

    for result in results:
        file_name = result['file']
        if not verbose:
            # Shorten file path for display
            file_name = str(Path(file_name).relative_to(Path(file_name).parts[0]))
            if len(file_name) > 67:
                file_name = "..." + file_name[-64:]

        print(f"{file_name:<70} {result['total']:<8} {result['correct']:<8} "
              f"{result['incorrect']:<10} {result['accuracy']:<10.2f}%")

        total_sum += result['total']
        correct_sum += result['correct']

    print("="*100)

    if len(results) > 1:
        overall_accuracy = (correct_sum / total_sum * 100) if total_sum > 0 else 0
        print(f"{'TOTAL':<70} {total_sum:<8} {correct_sum:<8} "
              f"{total_sum - correct_sum:<10} {overall_accuracy:<10.2f}%")
        print("="*100)


def group_by_model(results):
    """Group results by model name (extracted from file path)."""
    grouped = defaultdict(list)

    for result in results:
        # Extract model name from path (e.g., "Qwen/Qwen2.5-32B-Instruct")
        path_parts = Path(result['file']).parts
        if len(path_parts) >= 5:  # outputs/model/submodel/...
            model = f"{path_parts[-4]}/{path_parts[-3]}"
        elif len(path_parts) >= 4:
            model = path_parts[-3]
        else:
            model = "Unknown"

        grouped[model].append(result)

    return grouped


def print_summary_by_model(results):
    """Print a summary grouped by model."""
    grouped = group_by_model(results)

    print("\n" + "="*80)
    print("SUMMARY BY MODEL")
    print("="*80)

    for model, model_results in sorted(grouped.items()):
        total = sum(r['total'] for r in model_results)
        correct = sum(r['correct'] for r in model_results)
        accuracy = (correct / total * 100) if total > 0 else 0

        print(f"\nModel: {model}")
        print(f"  Files: {len(model_results)}")
        print(f"  Total questions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Count correct answers in JSONL files from eval/outputs"
    )
    parser.add_argument(
        "file",
        type=str,
        help="JSONL file to analyze"
    )

    args = parser.parse_args()

    # Analyze single file only
    result = count_correct_in_file(args.file)

    print(f"\nFile: {args.file}")
    print(f"Total: {result['total']}")
    print(f"Correct: {result['correct']}")
    print(f"Incorrect: {result['incorrect']}")
    print(f"Accuracy: {result['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
