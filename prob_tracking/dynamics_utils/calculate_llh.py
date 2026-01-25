#!/usr/bin/env python3
"""
Calculate log-likelihood and other metrics for samples in pool.json using a model checkpoint.

This script loads a model checkpoint and calculates comprehensive metrics including
log-likelihood, perplexity, entropy, and token probabilities for each sample in the pool.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys

# Add parent directory to path to import metrics module
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.prob_processing.metrics import calculate_response_metrics, ResponseMetrics


def load_model_and_tokenizer(checkpoint_path: str, device: str = None):
    """
    Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint directory
        device: Device to load model on (cuda/cpu)

    Returns:
        Tuple of (model, tokenizer, device)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )

    print(f"Loading tokenizer from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )

    if device != 'cuda':
        model = model.to(device)

    model.eval()
    print(f"Model loaded on {device}")

    return model, tokenizer, device


def load_pool_data(pool_path: str) -> List[Dict[str, Any]]:
    """
    Load samples from pool.json.

    Args:
        pool_path: Path to pool.json file

    Returns:
        List of sample dictionaries
    """
    with open(pool_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def calculate_metrics_for_pool(
    model,
    tokenizer,
    pool_data: List[Dict[str, Any]],
    device: str,
    include_instruction: bool = False
) -> List[Dict[str, Any]]:
    """
    Calculate metrics for all samples in the pool.

    Args:
        model: The language model
        tokenizer: The tokenizer
        pool_data: List of sample dictionaries with 'instruction', 'input', 'output', 'type'
        device: Device to run calculations on
        include_instruction: If True, calculate metrics on the whole sequence (prompt + output).
                           If False (default), calculate only on output tokens.

    Returns:
        List of dictionaries containing type and metrics for each sample
    """
    results = []

    for sample in tqdm(pool_data, desc="Calculating metrics"):
        # Extract fields
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')
        sample_type = sample.get('type', 'unknown')

        # Combine instruction and input as the prompt
        if instruction and input_text:
            prompt = f"{instruction}\n\n{input_text}"
        elif instruction:
            prompt = instruction
        elif input_text:
            prompt = input_text
        else:
            prompt = ""

        # Calculate metrics for the output given the prompt
        metrics = calculate_response_metrics(
            model=model,
            tokenizer=tokenizer,
            instruction=prompt,
            response=output_text,
            device=device,
            include_instruction=include_instruction
        )

        # Create result entry
        result = {
            'type': sample_type,
            'metrics': {
                'perplexity': metrics.perplexity,
                'log_likelihood': metrics.log_likelihood,
                'token_count': metrics.token_count
            }
        }

        results.append(result)

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate log-likelihood metrics for pool samples using a model checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., /helios-storage/helios3-data/cuong/model/Qwen_Math_high/checkpoint-555/)"
    )
    parser.add_argument(
        "--pool_path",
        type=str,
        default="prob_tracking/data/pool.json",
        help="Path to pool.json file (default: prob_tracking/data/pool.json)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON (default: prob_tracking/results/<checkpoint_name>_metrics.json)"
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

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint_path, args.device)

    # Load pool data
    print(f"\nLoading pool data from {args.pool_path}...")
    pool_data = load_pool_data(args.pool_path)
    print(f"Loaded {len(pool_data)} samples")

    # Calculate metrics
    mode_str = "whole sequence (prompt + output)" if args.include_instruction else "output tokens only"
    print(f"\nCalculating metrics for all samples (mode: {mode_str})...")
    results = calculate_metrics_for_pool(model, tokenizer, pool_data, device, args.include_instruction)

    # Determine output path
    if args.output_path is None:
        checkpoint_name = Path(args.checkpoint_path).parent.name if Path(args.checkpoint_path).name.startswith('checkpoint') else Path(args.checkpoint_path).name
        step_name = Path(args.checkpoint_path).name if Path(args.checkpoint_path).name.startswith('checkpoint') else 'final'
        output_path = f"prob_tracking/results/{checkpoint_name}_{step_name}_metrics.json"
    else:
        output_path = args.output_path

    # Save results
    save_results(results, output_path)

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)

    # Group by type
    type_groups = {}
    for result in results:
        sample_type = result['type']
        if sample_type not in type_groups:
            type_groups[sample_type] = []
        type_groups[sample_type].append(result['metrics'])

    for sample_type, metrics_list in type_groups.items():
        avg_log_likelihood = sum(m['log_likelihood'] for m in metrics_list) / len(metrics_list)
        avg_perplexity = sum(m['perplexity'] for m in metrics_list) / len(metrics_list)

        print(f"\nType: {sample_type} (n={len(metrics_list)})")
        print(f"  Average Log-Likelihood: {avg_log_likelihood:.4f}")
        print(f"  Average Perplexity: {avg_perplexity:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
