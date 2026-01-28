#!/usr/bin/env python3
"""
Evaluate all checkpoints in a folder and aggregate metrics.

This script processes all checkpoints in a model folder, calculates metrics for each,
and outputs lists of perplexity and log-likelihood values across training.
"""

import gc
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import torch
from tqdm import tqdm

from calculate_llh import (
    load_model_and_tokenizer,
    load_pool_data,
    calculate_metrics_for_pool
)


def find_checkpoints(model_folder: str) -> List[Tuple[int, str]]:
    """
    Find all checkpoint directories in the model folder.

    Args:
        model_folder: Path to model folder containing checkpoints

    Returns:
        List of (step_number, checkpoint_path) tuples, sorted by step number
    """
    model_path = Path(model_folder)
    checkpoints = []

    for item in model_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint'):
            # Extract step number from checkpoint name
            match = re.search(r'checkpoint-(\d+)', item.name)
            if match:
                step_num = int(match.group(1))
                checkpoints.append((step_num, str(item)))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])

    return checkpoints


def aggregate_metrics_by_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate metrics by sample type.

    Args:
        results: List of result dictionaries with 'type' and 'metrics'

    Returns:
        Dictionary mapping sample types to their aggregated metrics
    """
    type_metrics = {}

    for result in results:
        sample_type = result['type']
        if sample_type not in type_metrics:
            type_metrics[sample_type] = {
                'perplexity': [],
                'log_likelihood': [],
                'token_count': []
            }

        metrics = result['metrics']
        type_metrics[sample_type]['perplexity'].append(metrics['perplexity'])
        type_metrics[sample_type]['log_likelihood'].append(metrics['log_likelihood'])
        type_metrics[sample_type]['token_count'].append(metrics['token_count'])

    return type_metrics


def evaluate_all_checkpoints(
    model_folder: str,
    pool_path: str,
    output_path: str = None,
    device: str = None,
    base_model_path: str = None,
    include_instruction: bool = False
):
    """
    Evaluate all checkpoints in a folder.

    Args:
        model_folder: Path to model folder containing checkpoints
        pool_path: Path to pool.json file
        output_path: Path to save aggregated results
        device: Device to use (cuda/cpu)
        base_model_path: Path to base model (before training). If None, uses model_folder
        include_instruction: If True, calculate metrics on the whole sequence (prompt + output).
                           If False (default), calculate only on output tokens.
    """
    # Load pool data once
    print(f"\nLoading pool data from {pool_path}...")
    pool_data = load_pool_data(pool_path)
    print(f"Loaded {len(pool_data)} samples")

    # Store results for all checkpoints
    all_checkpoint_results = []

    # Evaluate base model first (step 0)
    mode_str = "whole sequence (prompt + output)" if include_instruction else "output tokens only"
    print(f"\n{'='*80}")
    print(f"Processing BASE MODEL (step 0) - Mode: {mode_str}")
    print(f"{'='*80}")

    base_model_path_to_use = base_model_path if base_model_path else model_folder
    print(f"Base model path: {base_model_path_to_use}")

    # Load base model
    model, tokenizer, device_used = load_model_and_tokenizer(base_model_path_to_use, device)

    # Calculate metrics for base model
    results = calculate_metrics_for_pool(model, tokenizer, pool_data, device_used, include_instruction)

    # Aggregate by type
    type_metrics = aggregate_metrics_by_type(results)

    # Store base model results
    base_model_result = {
        'checkpoint': 'base_model',
        'step': 0,
        'metrics_by_type': type_metrics
    }
    all_checkpoint_results.append(base_model_result)

    # Clean up model to free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"✓ Completed base model evaluation")

    # Find all checkpoints
    print(f"\nScanning for checkpoints in {model_folder}...")
    checkpoints = find_checkpoints(model_folder)

    if not checkpoints:
        print(f"No checkpoints found in {model_folder}")
        print("Only base model results will be saved.")
    else:
        print(f"Found {len(checkpoints)} checkpoints:")
        for step_num, checkpoint_path in checkpoints:
            print(f"  - checkpoint-{step_num}")

    # Process each checkpoint
    for step_num, checkpoint_path in tqdm(checkpoints, desc="Processing checkpoints"):
        print(f"\n{'='*80}")
        print(f"Processing checkpoint-{step_num}")
        print(f"{'='*80}")

        # Load model
        model, tokenizer, device_used = load_model_and_tokenizer(checkpoint_path, device)

        # Calculate metrics
        results = calculate_metrics_for_pool(model, tokenizer, pool_data, device_used, include_instruction)

        # Aggregate by type
        type_metrics = aggregate_metrics_by_type(results)

        # Store checkpoint results
        checkpoint_result = {
            'checkpoint': f'checkpoint-{step_num}',
            'step': step_num,
            'metrics_by_type': type_metrics
        }
        all_checkpoint_results.append(checkpoint_result)

        # Clean up model to free memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"✓ Completed checkpoint-{step_num}")

    # Create summary with lists of perplexity and log_likelihood across checkpoints
    print(f"\n{'='*80}")
    print("Creating summary...")
    print(f"{'='*80}")

    # Include base model (step 0) in the summary
    all_steps = [0] + [step for step, _ in checkpoints]
    all_checkpoint_names = ['base_model'] + [f'checkpoint-{step}' for step, _ in checkpoints]

    summary = {
        'model_folder': model_folder,
        'num_checkpoints': len(checkpoints) + 1,  # +1 for base model
        'checkpoints': all_checkpoint_names,
        'steps': all_steps,
        'results_by_type': {}
    }

    # Get all unique types
    all_types = set()
    for checkpoint_result in all_checkpoint_results:
        all_types.update(checkpoint_result['metrics_by_type'].keys())

    # For each type, create lists of metrics across checkpoints
    for sample_type in all_types:
        summary['results_by_type'][sample_type] = {
            'perplexity': [],
            'log_likelihood': [],
            'avg_perplexity': [],
            'avg_log_likelihood': []
        }

        for checkpoint_result in all_checkpoint_results:
            if sample_type in checkpoint_result['metrics_by_type']:
                type_data = checkpoint_result['metrics_by_type'][sample_type]

                # Store all values for this checkpoint
                summary['results_by_type'][sample_type]['perplexity'].append(
                    type_data['perplexity']
                )
                summary['results_by_type'][sample_type]['log_likelihood'].append(
                    type_data['log_likelihood']
                )

                # Calculate and store averages
                avg_ppl = sum(type_data['perplexity']) / len(type_data['perplexity'])
                avg_llh = sum(type_data['log_likelihood']) / len(type_data['log_likelihood'])

                summary['results_by_type'][sample_type]['avg_perplexity'].append(avg_ppl)
                summary['results_by_type'][sample_type]['avg_log_likelihood'].append(avg_llh)
            else:
                # Checkpoint doesn't have this type
                summary['results_by_type'][sample_type]['perplexity'].append(None)
                summary['results_by_type'][sample_type]['log_likelihood'].append(None)
                summary['results_by_type'][sample_type]['avg_perplexity'].append(None)
                summary['results_by_type'][sample_type]['avg_log_likelihood'].append(None)

    # Save results
    if output_path is None:
        model_name = Path(model_folder).name
        output_path = f"./prob_tracking/results/{model_name}_all_checkpoints_summary.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Summary saved to {output_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"{'='*80}")

    for sample_type in sorted(all_types):
        print(f"\nType: {sample_type}")
        avg_ppl_list = summary['results_by_type'][sample_type]['avg_perplexity']
        avg_llh_list = summary['results_by_type'][sample_type]['avg_log_likelihood']

        # Filter out None values
        valid_ppl = [x for x in avg_ppl_list if x is not None]
        valid_llh = [x for x in avg_llh_list if x is not None]

        if valid_ppl:
            print(f"  Perplexity: {valid_ppl[0]:.4f} → {valid_ppl[-1]:.4f}")
            print(f"  Log-Likelihood: {valid_llh[0]:.4f} → {valid_llh[-1]:.4f}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints in a folder and aggregate metrics"
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to model folder containing checkpoints (e.g., /projects/ai_safe/cuongdc/Qwen_Math_high/)"
    )
    parser.add_argument(
        "--pool_path",
        type=str,
        default="./prob_tracking/data/test_high.json",
        help="Path to pool.json file (default: ./prob_tracking/data/test_high.json)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON (default: ./prob_tracking/results/<model_name>_all_checkpoints_summary.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Path to base model before training (default: uses model_folder)"
    )
    parser.add_argument(
        "--include_instruction",
        action="store_true",
        help="If set, calculate log-likelihood on the whole sequence (instruction + input + output). "
             "By default, only calculates on output tokens."
    )

    args = parser.parse_args()

    evaluate_all_checkpoints(
        model_folder=args.model_folder,
        pool_path=args.pool_path,
        output_path=args.output_path,
        device=args.device,
        base_model_path=args.base_model_path,
        include_instruction=args.include_instruction
    )


if __name__ == "__main__":
    main()
