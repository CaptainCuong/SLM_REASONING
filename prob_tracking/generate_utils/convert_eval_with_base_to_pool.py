#!/usr/bin/env python3
"""
Convert evaluation results from all checkpoints AND a base model into a pool.json file.

This script reads JSONL evaluation output files from:
1. Checkpoint directories (e.g., outputs/cuongdc/Qwen_Math_high/checkpoint-*)
2. Base model directory (e.g., eval/outputs/Qwen/Qwen2.5-Math-7B/)

The output pool format has fields: instruction, input, output, type.
The type field format is: id_{question_id}_cp{checkpoint_number}_{correct/incorrect}
For base model: id_{question_id}_base_{correct/incorrect}

Examples:
- "id_1_cp555_incorrect"
- "id_42_cp1110_correct"
- "id_1_base_correct"
"""

import json
import os
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional


def extract_checkpoint_number(checkpoint_dir: str) -> int:
    """
    Extract checkpoint number from checkpoint directory name.

    Args:
        checkpoint_dir: Directory name like "checkpoint-555" or "checkpoint-10545"

    Returns:
        Checkpoint number as integer
    """
    match = re.search(r'checkpoint-(\d+)', checkpoint_dir)
    if match:
        return int(match.group(1))
    return 0


def find_eval_file(base_path: str, dataset: str) -> str:
    """
    Find the evaluation JSONL file for a given dataset.

    Args:
        base_path: Full path to directory containing dataset subdirectories
        dataset: Dataset name (e.g., "math", "test")

    Returns:
        Path to the JSONL file if found, empty string otherwise
    """
    dataset_path = os.path.join(base_path, dataset)
    if not os.path.isdir(dataset_path):
        return ""

    # Look for JSONL files
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jsonl'):
            return os.path.join(dataset_path, filename)

    return ""


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries from the JSONL file
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def process_eval_file(
    eval_file: str,
    source_name: str,
    question_to_id: Dict[str, int],
    next_question_id: int,
    instruction: str,
    stats: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, int], int]:
    """
    Process a single evaluation file and return pool entries.

    Args:
        eval_file: Path to JSONL evaluation file
        source_name: Name to use in type field (e.g., "cp555", "base")
        question_to_id: Mapping from question text to question ID
        next_question_id: Next available question ID
        instruction: Instruction text to include in pool entries
        stats: Statistics dictionary to update

    Returns:
        Tuple of (pool_entries, updated_question_to_id, updated_next_question_id)
    """
    pool_entries = []
    eval_data = load_jsonl(eval_file)

    correct_count = 0
    incorrect_count = 0

    for item in eval_data:
        question = item.get('question', '')
        generated_responses = item.get('generated_responses', [])
        is_correct = item.get('is_correct', False)

        # Get or assign question ID
        if question not in question_to_id:
            question_to_id[question] = next_question_id
            next_question_id += 1

        question_id = question_to_id[question]

        # Get the first generated response (or empty if none)
        output = generated_responses[0] if generated_responses else ""

        # Create type string: id_{question_id}_{source_name}_{correct/incorrect}
        correctness = "correct" if is_correct else "incorrect"
        type_str = f"id_{question_id}_{source_name}_{correctness}"

        # Create pool entry
        pool_entry = {
            "instruction": instruction,
            "input": question,
            "output": output,
            "type": type_str
        }

        pool_entries.append(pool_entry)

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

    return pool_entries, question_to_id, next_question_id, correct_count, incorrect_count


def convert_eval_with_base_to_pool(
    model_dir: str,
    base_model_dir: Optional[str] = None,
    dataset: str = "math",
    output_file: str = None,
    instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert evaluation results from all checkpoints and base model to pool format.

    Args:
        model_dir: Directory containing checkpoint subdirectories
                   (e.g., "eval/outputs/cuongdc/Qwen_Math_high")
        base_model_dir: Directory containing base model evaluation results
                        (e.g., "eval/outputs/Qwen/Qwen2.5-Math-7B")
        dataset: Dataset name to look for (e.g., "math", "test")
        output_file: Path to output pool JSON file
        instruction: Instruction text to include in pool entries

    Returns:
        Tuple of (pool_entries, statistics)
    """
    print("=" * 80)
    print("CONVERTING EVALUATION RESULTS TO POOL (WITH BASE MODEL)")
    print("=" * 80)
    print(f"\nModel directory: {model_dir}")
    print(f"Base model directory: {base_model_dir}")
    print(f"Dataset: {dataset}")
    print(f"Output file: {output_file}")

    # Statistics
    stats = {
        'total_sources': 0,
        'total_questions': 0,
        'total_entries': 0,
        'correct_entries': 0,
        'incorrect_entries': 0,
        'sources': {},
        'questions_seen': set()
    }

    # Pool entries
    pool_entries = []

    # Question ID mapping (to ensure consistent IDs across all sources)
    question_to_id = {}
    next_question_id = 1

    # ========================================
    # Process base model first (if provided)
    # ========================================
    if base_model_dir and os.path.isdir(base_model_dir):
        print("\n" + "-" * 80)
        print("Processing base model...")
        print("-" * 80)

        eval_file = find_eval_file(base_model_dir, dataset)
        if eval_file:
            print(f"  Eval file: {eval_file}")

            entries, question_to_id, next_question_id, correct, incorrect = process_eval_file(
                eval_file=eval_file,
                source_name="base",
                question_to_id=question_to_id,
                next_question_id=next_question_id,
                instruction=instruction,
                stats=stats
            )

            pool_entries.extend(entries)

            stats['total_sources'] += 1
            stats['sources']['base'] = {
                'total': correct + incorrect,
                'correct': correct,
                'incorrect': incorrect
            }
            stats['total_entries'] += correct + incorrect
            stats['correct_entries'] += correct
            stats['incorrect_entries'] += incorrect

            accuracy = correct / (correct + incorrect) * 100 if (correct + incorrect) > 0 else 0
            print(f"  Total: {correct + incorrect}, Correct: {correct}, "
                  f"Incorrect: {incorrect} ({accuracy:.1f}%)")
        else:
            print(f"  WARNING: No {dataset} evaluation file found in base model directory")
    else:
        if base_model_dir:
            print(f"\nWARNING: Base model directory not found: {base_model_dir}")

    # ========================================
    # Process checkpoint directories
    # ========================================
    print("\n" + "-" * 80)
    print("Processing checkpoints...")
    print("-" * 80)

    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    checkpoint_dirs = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            checkpoint_dirs.append(item)

    # Sort by checkpoint number
    checkpoint_dirs = sorted(checkpoint_dirs, key=extract_checkpoint_number)

    print(f"\nFound {len(checkpoint_dirs)} checkpoint directories:")
    for cp_dir in checkpoint_dirs:
        cp_num = extract_checkpoint_number(cp_dir)
        print(f"  - {cp_dir} (checkpoint {cp_num})")

    for cp_dir in checkpoint_dirs:
        checkpoint_path = os.path.join(model_dir, cp_dir)
        checkpoint_num = extract_checkpoint_number(cp_dir)

        # Find evaluation file
        eval_file = find_eval_file(checkpoint_path, dataset)
        if not eval_file:
            print(f"\nSkipping {cp_dir}: No {dataset} evaluation file found")
            continue

        print(f"\nProcessing {cp_dir}...")
        print(f"  Eval file: {os.path.basename(eval_file)}")

        entries, question_to_id, next_question_id, correct, incorrect = process_eval_file(
            eval_file=eval_file,
            source_name=f"cp{checkpoint_num}",
            question_to_id=question_to_id,
            next_question_id=next_question_id,
            instruction=instruction,
            stats=stats
        )

        pool_entries.extend(entries)

        stats['total_sources'] += 1
        stats['sources'][f'cp{checkpoint_num}'] = {
            'total': correct + incorrect,
            'correct': correct,
            'incorrect': incorrect
        }
        stats['total_entries'] += correct + incorrect
        stats['correct_entries'] += correct
        stats['incorrect_entries'] += incorrect

        accuracy = correct / (correct + incorrect) * 100 if (correct + incorrect) > 0 else 0
        print(f"  Total: {correct + incorrect}, Correct: {correct}, "
              f"Incorrect: {incorrect} ({accuracy:.1f}%)")

    stats['total_questions'] = len(question_to_id)

    # Save pool file
    if output_file:
        print("\n" + "-" * 80)
        print("Saving pool file...")
        print("-" * 80)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pool_entries, f, indent=2, ensure_ascii=False)

        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nSaved {len(pool_entries)} entries to {output_file} ({file_size:.2f} MB)")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal sources processed: {stats['total_sources']}")
    print(f"  - Base model: {'Yes' if 'base' in stats['sources'] else 'No'}")
    print(f"  - Checkpoints: {stats['total_sources'] - (1 if 'base' in stats['sources'] else 0)}")
    print(f"Total unique questions: {stats['total_questions']}")
    print(f"Total pool entries: {stats['total_entries']}")
    print(f"Total correct entries: {stats['correct_entries']}")
    print(f"Total incorrect entries: {stats['incorrect_entries']}")

    if stats['total_entries'] > 0:
        overall_accuracy = stats['correct_entries'] / stats['total_entries'] * 100
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

    print("\n" + "-" * 80)
    print("Per-source statistics:")
    print("-" * 80)
    print(f"{'Source':>15} {'Total':>8} {'Correct':>8} {'Incorrect':>10} {'Accuracy':>10}")
    print("-" * 55)

    # Print base model first if exists
    if 'base' in stats['sources']:
        src_stats = stats['sources']['base']
        accuracy = src_stats['correct'] / src_stats['total'] * 100 if src_stats['total'] > 0 else 0
        print(f"{'base':>15} {src_stats['total']:>8} {src_stats['correct']:>8} "
              f"{src_stats['incorrect']:>10} {accuracy:>9.1f}%")

    # Then print checkpoints sorted by number
    checkpoint_sources = [s for s in stats['sources'].keys() if s.startswith('cp')]
    for source in sorted(checkpoint_sources, key=lambda x: int(x[2:])):
        src_stats = stats['sources'][source]
        accuracy = src_stats['correct'] / src_stats['total'] * 100 if src_stats['total'] > 0 else 0
        print(f"{source:>15} {src_stats['total']:>8} {src_stats['correct']:>8} "
              f"{src_stats['incorrect']:>10} {accuracy:>9.1f}%")

    print("\n" + "=" * 80)
    print("SAMPLE ENTRIES")
    print("=" * 80)

    if pool_entries:
        # Show samples from base model and checkpoints
        base_entry = next((e for e in pool_entries if '_base_' in e['type']), None)
        cp_correct = next((e for e in pool_entries if '_cp' in e['type'] and 'incorrect' not in e['type']), None)
        cp_incorrect = next((e for e in pool_entries if '_cp' in e['type'] and 'incorrect' in e['type']), None)

        if base_entry:
            print("\nSample base model entry:")
            print(f"  Type: {base_entry['type']}")
            print(f"  Input: {base_entry['input'][:100]}...")

        if cp_correct:
            print("\nSample checkpoint correct entry:")
            print(f"  Type: {cp_correct['type']}")
            print(f"  Input: {cp_correct['input'][:100]}...")

        if cp_incorrect:
            print("\nSample checkpoint incorrect entry:")
            print(f"  Type: {cp_incorrect['type']}")
            print(f"  Input: {cp_incorrect['input'][:100]}...")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return pool_entries, stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert evaluation results from checkpoints and base model to pool format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert math dataset results with base model
  python convert_eval_with_base_to_pool.py \\
      --model_dir eval/outputs/cuongdc/Qwen_Math_high \\
      --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B \\
      --dataset math \\
      --output_file prob_tracking/data/pool_math_high_with_base.json

  # Convert test dataset results (checkpoints only, no base model)
  python convert_eval_with_base_to_pool.py \\
      --model_dir eval/outputs/cuongdc/Qwen_Math_high \\
      --dataset test \\
      --output_file prob_tracking/data/pool_test_high.json

Type format:
  - Checkpoints: id_{question_id}_cp{checkpoint_number}_{correct/incorrect}
  - Base model:  id_{question_id}_base_{correct/incorrect}

Examples:
  - "id_1_cp555_incorrect"
  - "id_42_cp1110_correct"
  - "id_1_base_correct"
        """
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing checkpoint subdirectories (e.g., eval/outputs/cuongdc/Qwen_Math_high)'
    )

    parser.add_argument(
        '--base_model_dir',
        type=str,
        default=None,
        help='Directory containing base model evaluation results (e.g., eval/outputs/Qwen/Qwen2.5-Math-7B)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='math',
        help='Dataset name to look for in directories (default: math)'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output pool JSON file'
    )

    parser.add_argument(
        '--instruction',
        type=str,
        default='Please reason step by step, and put your final answer within \\boxed{}.',
        help='Instruction text to include in pool entries'
    )

    args = parser.parse_args()

    # Convert results
    pool_entries, stats = convert_eval_with_base_to_pool(
        model_dir=args.model_dir,
        base_model_dir=args.base_model_dir,
        dataset=args.dataset,
        output_file=args.output_file,
        instruction=args.instruction
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
