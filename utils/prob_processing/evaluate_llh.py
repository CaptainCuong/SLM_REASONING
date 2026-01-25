#!/usr/bin/env python3
"""
Evaluate log-likelihood metrics for a specified model on a pool of samples.

This script loads a model and calculates perplexity and log-likelihood
metrics for each sample in a pool file, aggregating results by sample type.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import calculate_llh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'prob_tracking', 'dynamics_utils'))

from calculate_llh import (
    load_model_and_tokenizer,
    load_pool_data,
    calculate_metrics_for_pool
)


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


def calculate_summary_stats(type_metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate summary statistics for each type.

    Args:
        type_metrics: Dictionary mapping types to their metric lists

    Returns:
        Dictionary with summary statistics (mean, min, max) for each type
    """
    summary = {}

    for sample_type, metrics in type_metrics.items():
        ppl_list = metrics['perplexity']
        llh_list = metrics['log_likelihood']

        summary[sample_type] = {
            'count': len(ppl_list),
            'avg_perplexity': sum(ppl_list) / len(ppl_list),
            'min_perplexity': min(ppl_list),
            'max_perplexity': max(ppl_list),
            'avg_log_likelihood': sum(llh_list) / len(llh_list),
            'min_log_likelihood': min(llh_list),
            'max_log_likelihood': max(llh_list)
        }

    return summary


def evaluate_model_llh(
    model_path: str,
    pool_path: str,
    output_path: str = None,
    device: str = None,
    include_instruction: bool = False
):
    """
    Evaluate a model on a pool of samples.

    Args:
        model_path: Path to model (can be base model or checkpoint)
        pool_path: Path to pool.json file
        output_path: Path to save results JSON
        device: Device to use (cuda/cpu)
        include_instruction: If True, calculate metrics on the whole sequence (prompt + output).
                           If False (default), calculate only on output tokens.
    """
    mode_str = "whole sequence (prompt + output)" if include_instruction else "output tokens only"

    print("=" * 80)
    print("MODEL LOG-LIKELIHOOD EVALUATION")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Pool file: {pool_path}")
    print(f"Calculation mode: {mode_str}")

    # Load pool data
    print(f"\nLoading pool data from {pool_path}...")
    pool_data = load_pool_data(pool_path)
    print(f"✓ Loaded {len(pool_data)} samples")

    # Count samples by type
    type_counts = {}
    for sample in pool_data:
        sample_type = sample.get('type', 'unknown')
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

    print("\nSample counts by type:")
    for sample_type, count in sorted(type_counts.items()):
        print(f"  {sample_type}: {count}")

    # Load model
    print(f"\n{'='*80}")
    print("Loading model...")
    print(f"{'='*80}")
    model, tokenizer, device_used = load_model_and_tokenizer(model_path, device)
    print(f"✓ Model loaded successfully on {device_used}")

    # Calculate metrics
    print(f"\n{'='*80}")
    print("Calculating metrics...")
    print(f"{'='*80}")
    results = calculate_metrics_for_pool(model, tokenizer, pool_data, device_used, include_instruction)
    print(f"✓ Calculated metrics for {len(results)} samples")

    # Clean up model to free memory
    del model
    del tokenizer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate by type
    print(f"\n{'='*80}")
    print("Aggregating metrics by type...")
    print(f"{'='*80}")
    type_metrics = aggregate_metrics_by_type(results)

    # Calculate summary statistics
    summary_stats = calculate_summary_stats(type_metrics)

    # Print summary
    print("\nSummary Statistics:")
    print("-" * 80)
    for sample_type in sorted(summary_stats.keys()):
        stats = summary_stats[sample_type]
        print(f"\nType: {sample_type}")
        print(f"  Count: {stats['count']}")
        print(f"  Perplexity: {stats['avg_perplexity']:.4f} (min: {stats['min_perplexity']:.4f}, max: {stats['max_perplexity']:.4f})")
        print(f"  Log-Likelihood: {stats['avg_log_likelihood']:.4f} (min: {stats['min_log_likelihood']:.4f}, max: {stats['max_log_likelihood']:.4f})")

    # Prepare output
    output_data = {
        'model_path': model_path,
        'pool_path': pool_path,
        'include_instruction': include_instruction,
        'total_samples': len(results),
        'metrics_by_type': type_metrics,
        'summary_stats': summary_stats,
        'individual_results': results
    }

    # Save results
    if output_path is None:
        model_name = Path(model_path).name
        output_path = f"temp/model_{model_name}_metrics.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Saving results to {output_path}...")
    print(f"{'='*80}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Results saved ({file_size:.2f} MB)")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model log-likelihood metrics on a pool of samples"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Path to model (default: Qwen/Qwen2.5-Math-7B)"
    )
    parser.add_argument(
        "--pool_path",
        type=str,
        required=True,
        help="Path to pool.json file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="temp/model_metrics.json",
        help="Path to save output JSON (default: temp/model_<model_name>_metrics.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--include_instruction",
        action="store_true",
        help="If set, calculate log-likelihood on the whole sequence (instruction + input + output). "
             "By default, only calculates on output tokens."
    )

    args = parser.parse_args()

    evaluate_model_llh(
        model_path=args.model_path,
        pool_path=args.pool_path,
        output_path=args.output_path,
        device=args.device,
        include_instruction=args.include_instruction
    )


if __name__ == "__main__":
    # Example usage (can be uncommented for testing):
    # evaluate_model_llh(
    #     model_path="Qwen/Qwen2.5-Math-7B",
    #     pool_path="/home/cuongdc/SLM_REASONING/prob_tracking/data/pool_high_1k.json",
    #     output_path="/home/cuongdc/SLM_REASONING/temp/model_metrics.json"
    # )

    main()
