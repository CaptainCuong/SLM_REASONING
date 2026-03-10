#!/usr/bin/env python3
"""
Utility script to report pass@k-style average correctness for all checkpoints.
For each question with k generated answers, the score is (# correct) / k.
The dataset score is the mean of per-question scores.

Example usage:
    python count_avg_correct.py --folder eval/outputs/model/Qwen_Math_high/
    python count_avg_correct.py --folder eval/outputs/model/Qwen_Math_high/ --output results.csv
"""

import json
import csv
import argparse
from pathlib import Path


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def avg_correct_in_file(file_path):
    """
    For each question, compute (# correct answers) / (# generated answers).
    Return the mean over all questions and the question count.
    """
    data = load_jsonl(file_path)
    scores = []
    for entry in data:
        correctness = entry.get('answers_correctness')
        if correctness and len(correctness) > 0:
            scores.append(sum(correctness) / len(correctness))
        else:
            # Fall back to is_correct if answers_correctness is missing
            scores.append(1.0 if entry.get('is_correct', False) else 0.0)
    return scores


def avg_correct_for_dataset(dataset_folder, temp=None):
    """
    Aggregate per-question scores across all JSONL files in a dataset folder.
    """
    dataset_folder = Path(dataset_folder)
    if temp is not None:
        temp_str = f"{float(temp):.1f}"
        jsonl_files = list(dataset_folder.glob(f"*_t{temp_str}_*.jsonl"))
    else:
        jsonl_files = list(dataset_folder.glob("*.jsonl"))

    if not jsonl_files:
        return None

    all_scores = []
    for jsonl_file in jsonl_files:
        all_scores.extend(avg_correct_in_file(str(jsonl_file)))

    mean_score = sum(all_scores) / len(all_scores) * 100 if all_scores else 0

    return {
        'dataset': dataset_folder.name,
        'num_questions': len(all_scores),
        'avg_correct': mean_score,
    }


def process_checkpoint(checkpoint_folder, temp=None):
    checkpoint_folder = Path(checkpoint_folder)
    if not checkpoint_folder.exists():
        print(f"Warning: {checkpoint_folder} does not exist!")
        return {}

    results = {}
    for d in sorted(checkpoint_folder.iterdir()):
        if d.is_dir():
            result = avg_correct_for_dataset(d, temp=temp)
            if result:
                results[result['dataset']] = result
    return results


def extract_checkpoint_number(name):
    parts = name.split('-')
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


def process_all_checkpoints(model_folder, temp=None):
    model_folder = Path(model_folder)
    if not model_folder.exists():
        print(f"Error: {model_folder} does not exist!")
        return {}, []

    checkpoint_folders = sorted(
        [d for d in model_folder.iterdir() if d.is_dir() and d.name.startswith('checkpoint')],
        key=lambda x: extract_checkpoint_number(x.name)
    )

    if not checkpoint_folders:
        print(f"No checkpoint folders found in {model_folder}")
        return {}, []

    all_datasets = set()
    checkpoint_results = {}

    for cp in checkpoint_folders:
        print(f"Processing {cp.name}...")
        stats = process_checkpoint(cp, temp=temp)
        checkpoint_results[cp.name] = stats
        all_datasets.update(stats.keys())

    return checkpoint_results, sorted(all_datasets)


def print_results_table(checkpoint_results, all_datasets):
    if not checkpoint_results:
        print("No results to display.")
        return

    checkpoint_names = sorted(checkpoint_results.keys(), key=extract_checkpoint_number)
    col_width = 15

    sep = "=" * (col_width * (len(all_datasets) + 3))
    print("\n" + sep)
    header = f"{'Checkpoint':<{col_width}}"
    for ds in all_datasets:
        header += f"{ds:<{col_width}}"
    header += f"{'Weighted Avg':<{col_width}}{'Simple Avg':<{col_width}}"
    print(header)
    print(sep)

    for cp_name in checkpoint_names:
        row = f"{cp_name:<{col_width}}"
        stats = checkpoint_results[cp_name]

        total_questions = 0
        weighted_sum = 0.0
        scores = []

        for ds in all_datasets:
            if ds in stats:
                s = stats[ds]
                score = s['avg_correct']
                row += f"{score:>14.2f} "
                total_questions += s['num_questions']
                weighted_sum += score * s['num_questions']
                scores.append(score)
            else:
                row += f"{'':>{col_width}}"

        if total_questions > 0:
            weighted_avg = weighted_sum / total_questions
            row += f"{weighted_avg:>14.2f} "
        else:
            row += f"{'':>{col_width}}"

        if scores:
            simple_avg = sum(scores) / len(scores)
            row += f"{simple_avg:>14.2f} "
        else:
            row += f"{'':>{col_width}}"

        print(row)

    print(sep)


def save_results_to_csv(checkpoint_results, all_datasets, output_file):
    checkpoint_names = sorted(checkpoint_results.keys(), key=extract_checkpoint_number)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['checkpoint'] + all_datasets + ['weighted_avg', 'simple_avg'])

        for cp_name in checkpoint_names:
            row = [cp_name]
            stats = checkpoint_results[cp_name]

            total_questions = 0
            weighted_sum = 0.0
            scores = []

            for ds in all_datasets:
                if ds in stats:
                    s = stats[ds]
                    score = s['avg_correct']
                    row.append(f"{score:.2f}")
                    total_questions += s['num_questions']
                    weighted_sum += score * s['num_questions']
                    scores.append(score)
                else:
                    row.append('')

            if total_questions > 0:
                row.append(f"{weighted_sum / total_questions:.2f}")
            else:
                row.append('')

            if scores:
                row.append(f"{sum(scores) / len(scores):.2f}")
            else:
                row.append('')

            writer.writerow(row)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Report avg-correct-per-question for all checkpoints in a model folder"
    )
    parser.add_argument('--folder', type=str, default='eval/outputs/cuongdc/Qwen_Math_high/',
                        help='Model folder containing checkpoint-* subfolders')
    parser.add_argument('--output', type=str, default='checkpoint_avg_correct.csv',
                        help='Output CSV file (default: checkpoint_avg_correct.csv)')
    parser.add_argument('--no-table', action='store_true',
                        help='Skip printing the results table to console')
    parser.add_argument('--temp', type=str, default=None,
                        help='Temperature filter (e.g. 0.6 → *_t0.6_*.jsonl). '
                             'If not set, all JSONL files are used.')
    args = parser.parse_args()

    if args.temp is not None:
        print(f"Temperature filter: t={float(args.temp):.1f}")

    print(f"Analyzing checkpoints in: {args.folder}")
    checkpoint_results, all_datasets = process_all_checkpoints(args.folder, temp=args.temp)

    if not checkpoint_results:
        print(f"No checkpoint results found in {args.folder}")
        return

    if not args.no_table:
        print_results_table(checkpoint_results, all_datasets)

    save_results_to_csv(checkpoint_results, all_datasets, args.output)
    print(f"\nProcessed {len(checkpoint_results)} checkpoints across {len(all_datasets)} datasets")


if __name__ == '__main__':
    main()