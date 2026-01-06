#!/usr/bin/env python3
"""
Script to:
1. Load a sentence from algebra_generated.json
2. Calculate the loss
3. Take the gradient
4. Measure the size of the gradient
"""

import torch
import torch.nn.functional as F
import json
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


def load_sample_from_json(file_path: str, index: int = 0) -> dict:
    """
    Load a sample from a JSON file.

    Args:
        file_path: Path to the JSON file
        index: Index of the sample to load

    Returns:
        Dictionary containing the sample
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if index >= len(data):
        raise IndexError(f"Index {index} out of range. File has {len(data)} samples.")

    return data[index]


def calculate_loss_and_gradients(model, tokenizer, text, device="cuda"):
    """
    Calculate loss and gradients for a given text.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text string
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary containing loss, gradients, and gradient statistics
    """
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)

    # Enable gradient computation
    model.zero_grad()

    # Forward pass with gradient computation
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    # Backward pass to compute gradients
    loss.backward()

    # Collect gradient statistics
    gradient_stats = {}
    total_grad_norm = 0.0
    total_params = 0

    # Collect gradients for each parameter
    gradients_by_layer = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_norm = torch.norm(grad).item()
            grad_mean = torch.mean(grad).item()
            grad_std = torch.std(grad).item()
            grad_max = torch.max(torch.abs(grad)).item()

            gradients_by_layer[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max,
                'shape': list(grad.shape),
                'num_elements': grad.numel()
            }

            total_grad_norm += grad_norm ** 2
            total_params += grad.numel()

    # Calculate global gradient norm (L2 norm across all parameters)
    global_grad_norm = np.sqrt(total_grad_norm)

    gradient_stats['global_grad_norm'] = global_grad_norm
    gradient_stats['total_parameters'] = total_params
    gradient_stats['loss'] = loss.item()
    gradient_stats['perplexity'] = np.exp(loss.item())
    gradient_stats['num_tokens'] = input_ids.shape[1]
    gradient_stats['gradients_by_layer'] = gradients_by_layer

    return gradient_stats


def print_gradient_summary(gradient_stats: dict, top_k: int = 10):
    """
    Print a summary of gradient statistics.

    Args:
        gradient_stats: Dictionary containing gradient statistics
        top_k: Number of top layers to display
    """
    print("\n" + "="*70)
    print("GRADIENT ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nGlobal Statistics:")
    print(f"  Loss: {gradient_stats['loss']:.4f}")
    print(f"  Perplexity: {gradient_stats['perplexity']:.4f}")
    print(f"  Number of tokens: {gradient_stats['num_tokens']}")
    print(f"  Total parameters: {gradient_stats['total_parameters']:,}")
    print(f"  Global gradient norm (L2): {gradient_stats['global_grad_norm']:.4f}")

    # Sort layers by gradient norm
    layers = gradient_stats['gradients_by_layer']
    sorted_layers = sorted(layers.items(), key=lambda x: x[1]['norm'], reverse=True)

    print(f"\nTop {top_k} layers by gradient norm:")
    print(f"{'Layer Name':<50} {'Grad Norm':<12} {'Max':<12} {'Mean':<12}")
    print("-" * 90)

    for i, (name, stats) in enumerate(sorted_layers[:top_k]):
        print(f"{name:<50} {stats['norm']:<12.4f} {stats['max']:<12.6f} {stats['mean']:<12.6f}")

    print("\n" + "="*70)


def save_gradient_stats(gradient_stats: dict, output_path: str):
    """
    Save gradient statistics to a JSON file.

    Args:
        gradient_stats: Dictionary containing gradient statistics
        output_path: Path to save the JSON file
    """
    # Convert to JSON-serializable format
    output_data = {
        'loss': gradient_stats['loss'],
        'perplexity': gradient_stats['perplexity'],
        'num_tokens': gradient_stats['num_tokens'],
        'total_parameters': gradient_stats['total_parameters'],
        'global_grad_norm': gradient_stats['global_grad_norm'],
        'top_20_layers': []
    }

    # Get top 20 layers by gradient norm
    layers = gradient_stats['gradients_by_layer']
    sorted_layers = sorted(layers.items(), key=lambda x: x[1]['norm'], reverse=True)

    for name, stats in sorted_layers[:20]:
        output_data['top_20_layers'].append({
            'name': name,
            'norm': stats['norm'],
            'mean': stats['mean'],
            'std': stats['std'],
            'max': stats['max'],
            'shape': stats['shape'],
            'num_elements': stats['num_elements']
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nGradient statistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate loss and gradients for a sample from algebra_generated.json'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='generated_solutions/algebra_generated_stage2.json',
        help='Path to the JSON data file'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Index of the sample to analyze'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='Qwen/Qwen2.5-Math-7B',
        help='Model name or path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test/gradient_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--use_output_field',
        action='store_true',
        help='Use the output field instead of input field'
    )

    args = parser.parse_args()

    print("="*70)
    print("GRADIENT ANALYSIS")
    print("="*70)

    # Load sample
    print(f"\nLoading sample {args.index} from {args.data_file}...")
    sample = load_sample_from_json(args.data_file, args.index)

    # Extract text
    if args.use_output_field:
        text = sample.get('output', '')
        print(f"Using 'output' field (solution)")
    else:
        text = sample.get('input', '')
        print(f"Using 'input' field (problem)")

    print(f"\nText length: {len(text)} characters")
    print(f"Text preview: {text[:200]}...")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,  # Use float32 for gradient computation
        device_map=args.device,
        trust_remote_code=True
    )

    # Ensure model is in training mode for gradient computation
    model.train()

    print(f"Model loaded on {args.device}")

    # Calculate loss and gradients
    print("\nCalculating loss and gradients...")
    gradient_stats = calculate_loss_and_gradients(model, tokenizer, text, args.device)

    # Print summary
    print_gradient_summary(gradient_stats, top_k=15)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'gradient_stats_sample_{args.index}.json'
    save_gradient_stats(gradient_stats, str(output_file))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
