#!/usr/bin/env python3
"""
Utility script to report accuracy for all checkpoints in a model folder.
For a given path like eval/outputs/model/Qwen_Math_high/, this script will:
1. Find all checkpoint-* folders
2. Calculate accuracy for each dataset in each checkpoint
3. Export results to a CSV file with datasets as columns and checkpoints as rows

Example usage:
    python count_all_checkpoints.py --folder eval/outputs/model/Qwen_Math_high/
    python count_all_checkpoints.py --folder eval/outputs/model/Qwen_Math_high/ --output results.csv
"""

import json
import os
import argparse
import csv
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
        result = count_correct_in_file(str(jsonl_file))
        total_questions += result['total']
        total_correct += result['correct']

    accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    return {
        'dataset': dataset_folder.name,
        'total': total_questions,
        'correct': total_correct,
        'incorrect': total_questions - total_correct,
        'accuracy': accuracy,
        'num_files': len(jsonl_files)
    }


def count_all_datasets_in_checkpoint(checkpoint_folder):
    """
    Count accuracy for all datasets in a checkpoint folder.

    Args:
        checkpoint_folder: Path to checkpoint folder containing dataset subfolders

    Returns:
        Dict mapping dataset names to their statistics (accuracy and total questions)
    """
    checkpoint_folder = Path(checkpoint_folder)

    if not checkpoint_folder.exists():
        print(f"Warning: Folder {checkpoint_folder} does not exist!")
        return {}

    # Get all subdirectories (each represents a dataset)
    dataset_folders = [d for d in checkpoint_folder.iterdir() if d.is_dir()]

    results = {}
    for dataset_folder in sorted(dataset_folders):
        result = count_dataset_accuracy(dataset_folder)
        if result:
            results[result['dataset']] = {
                'accuracy': result['accuracy'],
                'total': result['total'],
                'correct': result['correct']
            }

    return results


def extract_checkpoint_number(checkpoint_name):
    """
    Extract the numeric part from checkpoint name for sorting.
    E.g., 'checkpoint-555' -> 555
    """
    parts = checkpoint_name.split('-')
    if len(parts) >= 2:
        return int(parts[-1])
    return 0


def count_all_checkpoints(model_folder):
    """
    Count accuracy for all checkpoints in a model folder.

    Args:
        model_folder: Path to model folder containing checkpoint-* subfolders

    Returns:
        Dict with checkpoint data and list of all dataset names
    """
    model_folder = Path(model_folder)

    if not model_folder.exists():
        print(f"Error: Folder {model_folder} does not exist!")
        return {}, []

    # Find all checkpoint folders
    checkpoint_folders = [d for d in model_folder.iterdir()
                         if d.is_dir() and d.name.startswith('checkpoint')]

    if not checkpoint_folders:
        print(f"No checkpoint folders found in {model_folder}")
        return {}, []

    # Sort checkpoints by number
    checkpoint_folders.sort(key=lambda x: extract_checkpoint_number(x.name))

    # Collect all unique dataset names across all checkpoints
    all_datasets = set()
    checkpoint_results = {}

    for checkpoint_folder in checkpoint_folders:
        print(f"Processing {checkpoint_folder.name}...")
        dataset_accuracies = count_all_datasets_in_checkpoint(checkpoint_folder)

        checkpoint_results[checkpoint_folder.name] = dataset_accuracies
        all_datasets.update(dataset_accuracies.keys())

    # Sort dataset names for consistent ordering
    all_datasets = sorted(all_datasets)

    return checkpoint_results, all_datasets


def save_results_to_csv(checkpoint_results, all_datasets, output_file):
    """
    Save results to a CSV file with datasets as columns and checkpoints as rows.

    Args:
        checkpoint_results: Dict mapping checkpoint names to their dataset statistics
        all_datasets: List of all dataset names
        output_file: Path to output CSV file
    """
    # Sort checkpoints by number
    checkpoint_names = sorted(checkpoint_results.keys(),
                             key=lambda x: extract_checkpoint_number(x))

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header with two average columns
        header = ['checkpoint'] + all_datasets + ['weighted_avg', 'simple_avg']
        writer.writerow(header)

        # Write data for each checkpoint
        for checkpoint_name in checkpoint_names:
            row = [checkpoint_name]
            dataset_stats = checkpoint_results[checkpoint_name]

            total_questions = 0
            total_correct = 0
            accuracies = []

            for dataset in all_datasets:
                # Get accuracy if available, otherwise empty
                if dataset in dataset_stats:
                    stats = dataset_stats[dataset]
                    accuracy = stats['accuracy']
                    row.append(f"{accuracy:.2f}")
                    total_questions += stats['total']
                    total_correct += stats['correct']
                    accuracies.append(accuracy)
                else:
                    row.append('')

            # Calculate weighted average (total correct / total questions)
            if total_questions > 0:
                weighted_avg = (total_correct / total_questions) * 100
                row.append(f"{weighted_avg:.2f}")
            else:
                row.append('')

            # Calculate simple average (mean of accuracy percentages)
            if accuracies:
                simple_avg = sum(accuracies) / len(accuracies)
                row.append(f"{simple_avg:.2f}")
            else:
                row.append('')

            writer.writerow(row)

    print(f"\nResults saved to: {output_file}")


def print_results_table(checkpoint_results, all_datasets):
    """Print results in a formatted table."""
    if not checkpoint_results:
        print("No results to display.")
        return

    # Sort checkpoints by number
    checkpoint_names = sorted(checkpoint_results.keys(),
                             key=lambda x: extract_checkpoint_number(x))

    # Print header with two average columns
    col_width = 15
    print("\n" + "="*(col_width * (len(all_datasets) + 3)))
    header = f"{'Checkpoint':<{col_width}}"
    for dataset in all_datasets:
        header += f"{dataset:<{col_width}}"
    header += f"{'Weighted Avg':<{col_width}}{'Simple Avg':<{col_width}}"
    print(header)
    print("="*(col_width * (len(all_datasets) + 3)))

    # Print data for each checkpoint
    for checkpoint_name in checkpoint_names:
        row = f"{checkpoint_name:<{col_width}}"
        dataset_stats = checkpoint_results[checkpoint_name]

        total_questions = 0
        total_correct = 0
        accuracies = []

        for dataset in all_datasets:
            if dataset in dataset_stats:
                stats = dataset_stats[dataset]
                accuracy = stats['accuracy']
                row += f"{accuracy:>14.2f} "
                total_questions += stats['total']
                total_correct += stats['correct']
                accuracies.append(accuracy)
            else:
                row += f"{'':>{col_width}}"

        # Calculate weighted average (total correct / total questions)
        if total_questions > 0:
            weighted_avg = (total_correct / total_questions) * 100
            row += f"{weighted_avg:>14.2f} "
        else:
            row += f"{'':>{col_width}}"

        # Calculate simple average (mean of accuracy percentages)
        if accuracies:
            simple_avg = sum(accuracies) / len(accuracies)
            row += f"{simple_avg:>14.2f} "
        else:
            row += f"{'':>{col_width}}"

        print(row)

    print("="*(col_width * (len(all_datasets) + 3)))


def main():
    parser = argparse.ArgumentParser(
        description="Report accuracy for all checkpoints in a model folder"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="eval/outputs/model/Qwen_Math_high/",
        help="Model folder containing checkpoint-* subfolders (default: eval/outputs/model/Qwen_Math_high/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoint_results.csv",
        help="Output CSV file (default: checkpoint_results.csv)"
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip printing the results table to console"
    )

    args = parser.parse_args()

    # Count accuracy for all checkpoints
    print(f"Analyzing checkpoints in: {args.folder}")
    checkpoint_results, all_datasets = count_all_checkpoints(args.folder)

    if not checkpoint_results:
        print(f"No checkpoint results found in {args.folder}")
        return

    # Print results table
    if not args.no_table:
        print_results_table(checkpoint_results, all_datasets)

    # Save to CSV
    save_results_to_csv(checkpoint_results, all_datasets, args.output)
    print(f"\nProcessed {len(checkpoint_results)} checkpoints across {len(all_datasets)} datasets")


if __name__ == "__main__":
    main()
