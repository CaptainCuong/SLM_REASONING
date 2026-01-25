#!/usr/bin/env python3
"""
Convert checkpoint generation results from a folder into a pool file.
Each entry in the pool will include checkpoint information and correctness status.
"""

import json
import os
import glob
from typing import List, Dict
import argparse


def extract_checkpoint_name(filename: str) -> str:
    """
    Extract checkpoint name from filename.

    Args:
        filename: Result filename (e.g., 'checkpoint-25_results.json' or 'base_model_results.json')

    Returns:
        checkpoint name (e.g., 'checkpoint-25' or 'base_model')
    """
    basename = os.path.basename(filename)
    # Remove '_results.json' suffix
    checkpoint_name = basename.replace('_results.json', '')
    return checkpoint_name


def convert_results_to_pool(
    results_dir: str,
    output_file: str,
    include_base_model: bool = True,
    checkpoint_pattern: str = "*_results.json"
):
    """
    Convert all checkpoint results in a directory to a pool file.

    Args:
        results_dir: Directory containing checkpoint result files
        output_file: Path to output pool file
        include_base_model: Whether to include base_model_results.json
        checkpoint_pattern: Glob pattern for result files
    """
    print("=" * 80)
    print("CONVERTING CHECKPOINT RESULTS TO POOL")
    print("=" * 80)

    print(f"\nInput directory: {results_dir}")
    print(f"Output file: {output_file}")
    print(f"Include base model: {include_base_model}")

    # Find all result files
    pattern = os.path.join(results_dir, checkpoint_pattern)
    result_files = sorted(glob.glob(pattern))

    if not include_base_model:
        result_files = [f for f in result_files if 'base_model' not in f]

    print(f"\nFound {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {os.path.basename(f)}")

    # Collect all pool entries
    pool_entries = []

    # Statistics
    stats = {
        'total_checkpoints': 0,
        'total_questions': 0,
        'total_entries': 0,
        'entries_by_checkpoint': {},
        'correct_by_checkpoint': {},
        'incorrect_by_checkpoint': {}
    }

    print("\n" + "-" * 80)
    print("Processing checkpoint results...")
    print("-" * 80)

    for result_file in result_files:
        checkpoint_name = extract_checkpoint_name(result_file)

        print(f"\nProcessing {checkpoint_name}...")

        # Load result file
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        # Extract results
        results = result_data.get('results', [])
        checkpoint_accuracy = result_data.get('accuracy', 0.0)

        stats['total_checkpoints'] += 1
        stats['entries_by_checkpoint'][checkpoint_name] = 0
        stats['correct_by_checkpoint'][checkpoint_name] = 0
        stats['incorrect_by_checkpoint'][checkpoint_name] = 0

        # Process each result
        for result_item in results:
            instruction = result_item.get('instruction', '')
            question = result_item.get('question', '')
            generated_response = result_item.get('generated_response', '')
            is_correct = result_item.get('is_correct', False)

            # Determine type field (checkpoint + correctness)
            # Format: "cp_25_correct" or "base_incorrect"
            if checkpoint_name == 'base_model':
                type_prefix = 'base'
            else:
                # Extract checkpoint number from "checkpoint-25" -> "cp_25"
                type_prefix = checkpoint_name.replace('checkpoint-', 'cp_')

            type_suffix = 'correct' if is_correct else 'incorrect'
            type_field = f"{type_prefix}_{type_suffix}"

            # Create pool entry
            pool_entry = {
                'instruction': instruction,
                'input': question,
                'output': generated_response,
                'type': type_field
            }

            pool_entries.append(pool_entry)

            stats['total_entries'] += 1
            stats['entries_by_checkpoint'][checkpoint_name] += 1

            if is_correct:
                stats['correct_by_checkpoint'][checkpoint_name] += 1
            else:
                stats['incorrect_by_checkpoint'][checkpoint_name] += 1

        print(f"  Processed {len(results)} results from {checkpoint_name}")
        print(f"  Accuracy: {checkpoint_accuracy:.2%}")
        print(f"  Correct: {stats['correct_by_checkpoint'][checkpoint_name]}")
        print(f"  Incorrect: {stats['incorrect_by_checkpoint'][checkpoint_name]}")

    # Count unique questions
    unique_questions = set()
    for entry in pool_entries:
        unique_questions.add(entry['input'])
    stats['total_questions'] = len(unique_questions)

    # Save pool file
    print("\n" + "-" * 80)
    print("Saving pool file...")
    print("-" * 80)

    print(f"\nWriting {len(pool_entries)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pool_entries, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {len(pool_entries)} entries ({file_size:.2f} MB)")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    print(f"\nTotal checkpoints processed: {stats['total_checkpoints']}")
    print(f"Total unique questions: {stats['total_questions']}")
    print(f"Total pool entries: {stats['total_entries']}")
    print(f"Average entries per checkpoint: {stats['total_entries'] / stats['total_checkpoints']:.1f}")

    print("\n" + "-" * 80)
    print("Entries per checkpoint:")
    print("-" * 80)
    for checkpoint in sorted(stats['entries_by_checkpoint'].keys()):
        total = stats['entries_by_checkpoint'][checkpoint]
        correct = stats['correct_by_checkpoint'][checkpoint]
        incorrect = stats['incorrect_by_checkpoint'][checkpoint]
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"  {checkpoint:30s}: {total:4d} total, {correct:4d} correct, {incorrect:4d} incorrect ({accuracy:.1f}%)")

    # Show sample entries
    print("\n" + "-" * 80)
    print("SAMPLE POOL ENTRIES")
    print("-" * 80)

    if pool_entries:
        print("\nFirst entry (correct):")
        # Find first correct entry
        correct_entry = next((e for e in pool_entries if 'correct' in e['type']), None)
        if correct_entry:
            print(f"  Type: {correct_entry['type']}")
            print(f"  Instruction: {correct_entry['instruction'][:80]}...")
            print(f"  Input: {correct_entry['input'][:100]}...")
            print(f"  Output: {correct_entry['output'][:150]}...")

        print("\nFirst entry (incorrect):")
        # Find first incorrect entry
        incorrect_entry = next((e for e in pool_entries if 'incorrect' in e['type']), None)
        if incorrect_entry:
            print(f"  Type: {incorrect_entry['type']}")
            print(f"  Instruction: {incorrect_entry['instruction'][:80]}...")
            print(f"  Input: {incorrect_entry['input'][:100]}...")
            print(f"  Output: {incorrect_entry['output'][:150]}...")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return stats, pool_entries


def create_filtered_pools(
    pool_entries: List[Dict],
    output_dir: str,
    prefix: str = ""
):
    """
    Create filtered pool files (correct only, incorrect only, by type).

    Args:
        pool_entries: List of pool entries
        output_dir: Output directory for filtered pools
        prefix: Prefix for output filenames
    """
    print("\n" + "=" * 80)
    print("CREATING FILTERED POOLS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Filter by correctness
    correct_entries = [e for e in pool_entries if 'correct' in e['type']]
    incorrect_entries = [e for e in pool_entries if 'incorrect' in e['type']]

    # Save correct entries
    if correct_entries:
        correct_file = os.path.join(output_dir, f"{prefix}pool_correct.json")
        print(f"\nSaving {len(correct_entries)} correct entries to {correct_file}...")
        with open(correct_file, 'w', encoding='utf-8') as f:
            json.dump(correct_entries, f, indent=2, ensure_ascii=False)
        print(f"  Saved ({os.path.getsize(correct_file) / (1024 * 1024):.2f} MB)")

    # Save incorrect entries
    if incorrect_entries:
        incorrect_file = os.path.join(output_dir, f"{prefix}pool_incorrect.json")
        print(f"\nSaving {len(incorrect_entries)} incorrect entries to {incorrect_file}...")
        with open(incorrect_file, 'w', encoding='utf-8') as f:
            json.dump(incorrect_entries, f, indent=2, ensure_ascii=False)
        print(f"  Saved ({os.path.getsize(incorrect_file) / (1024 * 1024):.2f} MB)")

    # Group by type
    type_groups = {}
    for entry in pool_entries:
        entry_type = entry['type']
        if entry_type not in type_groups:
            type_groups[entry_type] = []
        type_groups[entry_type].append(entry)

    # Save by type
    print(f"\nSaving {len(type_groups)} type-specific pools...")
    for entry_type, entries in sorted(type_groups.items()):
        type_file = os.path.join(output_dir, f"{prefix}pool_{entry_type}.json")
        with open(type_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"  {entry_type}: {len(entries)} entries ({os.path.getsize(type_file) / (1024 * 1024):.2f} MB)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Convert checkpoint generation results to pool format'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing checkpoint result files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output pool file'
    )
    parser.add_argument(
        '--no-base-model',
        action='store_true',
        help='Exclude base_model_results.json'
    )
    parser.add_argument(
        '--create-filtered',
        action='store_true',
        help='Create filtered pools (correct/incorrect/by checkpoint)'
    )
    parser.add_argument(
        '--filtered-dir',
        type=str,
        default=None,
        help='Directory for filtered pools (default: same as output_file directory)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefix for filtered pool filenames'
    )

    args = parser.parse_args()

    # Convert results to pool
    stats, pool_entries = convert_results_to_pool(
        results_dir=args.results_dir,
        output_file=args.output_file,
        include_base_model=not args.no_base_model
    )

    # Create filtered pools if requested
    if args.create_filtered:
        filtered_dir = args.filtered_dir or os.path.dirname(args.output_file)
        create_filtered_pools(
            pool_entries=pool_entries,
            output_dir=filtered_dir,
            prefix=args.prefix
        )

    print("\nDone!")


if __name__ == "__main__":
    # Example usage (can be uncommented for testing)
    # results_dir = "/home/cuongdc/SLM_REASONING/prob_tracking/results/Qwen_Math_high_1k_generation_results"
    # output_file = "/home/cuongdc/SLM_REASONING/prob_tracking/data/traceback_pool_high_all.json"
    #
    # stats, pool_entries = convert_results_to_pool(
    #     results_dir=results_dir,
    #     output_file=output_file,
    #     include_base_model=True
    # )
    #
    # # Create filtered pools
    # create_filtered_pools(
    #     pool_entries=pool_entries,
    #     output_dir="/home/cuongdc/SLM_REASONING/prob_tracking/data",
    #     prefix="high_"
    # )

    main()
