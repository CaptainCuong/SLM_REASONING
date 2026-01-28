#!/usr/bin/env python3
"""
Calculate loss of an LLM model on given samples.

This script loads a model and calculates the loss for each sample.
"""

import gc
import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


def prepare_prompt(instruction: str, input_text: str, tokenizer: Any) -> str:
    """
    Prepare a prompt from instruction and input using chat template.

    Args:
        instruction: The instruction text
        input_text: The input/question text
        tokenizer: The tokenizer to use

    Returns:
        Formatted prompt string with chat template applied
    """
    # Combine instruction and input as user content
    if instruction and input_text:
        user_content = f"{instruction}\n\n{input_text}"
    elif instruction:
        user_content = instruction
    elif input_text:
        user_content = input_text
    else:
        user_content = ""

    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": user_content}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = None,
    torch_dtype: torch.dtype = torch.bfloat16
) -> Tuple[nn.Module, Any, str]:
    """
    Load model and tokenizer from HuggingFace or local path.

    Args:
        model_name_or_path: Model name on HuggingFace or path to local checkpoint
        device: Device to load model on (cuda/cpu)
        torch_dtype: Data type for model weights

    Returns:
        Tuple of (model, tokenizer, device)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )

    print(f"Loading tokenizer from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device != 'cuda':
        model = model.to(device)

    # Set to eval mode (no gradients needed)
    model.eval()
    print(f"Model loaded on {device}")

    return model, tokenizer, device


def calculate_loss_for_sample(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: str,
    max_length: int = 32768
) -> Dict[str, Any]:
    """
    Calculate loss for a single sample (prompt + response).

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        response: The response/output text
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Dictionary containing loss statistics
    """
    # Tokenize the full sequence
    full_text = prompt + response
    inputs = tokenizer(
        full_text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding=False
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Tokenize just the prompt to know where response starts
    prompt_tokens = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding=False
    )
    prompt_length = prompt_tokens['input_ids'].shape[1]

    # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for response
    labels = input_ids.clone()
    labels[0, :prompt_length] = -100  # Mask prompt tokens

    # Forward pass with labels to get loss (no gradient computation needed)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    loss = outputs.loss

    result = {
        'loss': loss.item(),
        'sequence_length': input_ids.shape[1],
        'response_length': input_ids.shape[1] - prompt_length
    }

    # Clean up to prevent memory leak
    del inputs, input_ids, attention_mask, prompt_tokens, labels, outputs, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def calculate_loss_for_pool(
    model: nn.Module,
    tokenizer: Any,
    pool_data: List[Dict[str, Any]],
    device: str,
    max_length: int = 32768
) -> List[Dict[str, Any]]:
    """
    Calculate loss for all samples in a pool.

    Args:
        model: The language model
        tokenizer: The tokenizer
        pool_data: List of sample dictionaries with 'instruction', 'input', 'output', 'type'
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        List of dictionaries containing loss statistics for each sample
    """
    results = []

    for sample in tqdm(pool_data, desc="Calculating losses"):
        # Extract fields
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')
        sample_type = sample.get('type', 'unknown')

        # Prepare prompt using chat template
        prompt = prepare_prompt(instruction, input_text, tokenizer)

        # Calculate loss
        loss_result = calculate_loss_for_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            response=output_text,
            device=device,
            max_length=max_length
        )

        result = {
            'type': sample_type,
            'loss_stats': loss_result
        }

        results.append(result)

    return results


def load_pool_data(pool_path: str) -> List[Dict[str, Any]]:
    """Load samples from pool JSON file."""
    with open(pool_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def aggregate_results_by_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate loss statistics by sample type."""
    type_stats = {}

    for result in results:
        sample_type = result['type']
        if sample_type not in type_stats:
            type_stats[sample_type] = {
                'losses': [],
                'count': 0
            }

        type_stats[sample_type]['losses'].append(
            result['loss_stats']['loss']
        )
        type_stats[sample_type]['count'] += 1

    # Calculate summary statistics
    summary = {}
    for sample_type, stats in type_stats.items():
        losses = stats['losses']
        summary[sample_type] = {
            'count': stats['count'],
            'avg_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses)
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Calculate loss for an LLM on pool samples"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Model name on HuggingFace or path to checkpoint (default: Qwen/Qwen2.5-Math-7B)"
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
        default="ntk/results/losses.json",
        help="Path to save output JSON (default: ntk/results/losses.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Maximum sequence length (default: 32768)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LOSS CALCULATION")
    print("=" * 80)
    print(f"\nModel: {args.model_name_or_path}")

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        args.device
    )

    # Load and process pool data
    print(f"\nLoading pool data from {args.pool_path}...")
    pool_data = load_pool_data(args.pool_path)
    print(f"Loaded {len(pool_data)} samples")

    # Calculate loss for all samples
    print("\nCalculating losses...")
    results = calculate_loss_for_pool(
        model=model,
        tokenizer=tokenizer,
        pool_data=pool_data,
        device=device,
        max_length=args.max_length
    )

    # Aggregate results
    summary = aggregate_results_by_type(results)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics by Type:")
    print("=" * 80)
    for sample_type, stats in sorted(summary.items()):
        print(f"\nType: {sample_type} (n={stats['count']})")
        print(f"  Loss: {stats['avg_loss']:.6f} ± {stats['std_loss']:.6f}")

    # Prepare output
    output_data = {
        'model_path': args.model_name_or_path,
        'pool_path': args.pool_path,
        'total_samples': len(results),
        'summary_by_type': summary,
        'individual_results': results
    }

    # Save results
    save_results(output_data, args.output_path)

    # Clean up model and tokenizer to release memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
