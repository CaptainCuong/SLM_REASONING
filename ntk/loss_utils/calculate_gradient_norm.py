#!/usr/bin/env python3
"""
Calculate gradient norm of loss for each answer in the dataset.

This script loads a model and calculates the gradient norm of the loss
for each answer, which can be used to analyze model sensitivity.
"""

import gc
import json
import os
import argparse
from typing import List, Dict, Any, Tuple
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = None,
    torch_dtype: torch.dtype = torch.bfloat16
) -> Tuple[nn.Module, Any, str]:
    """
    Load model and tokenizer from HuggingFace or local path.
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

    # Keep in train mode to compute gradients
    model.train()
    print(f"Model loaded on {device}")

    return model, tokenizer, device


def prepare_prompt(question: str, tokenizer: Any) -> str:
    """
    Prepare a prompt from question using chat template.
    """
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


def calculate_gradient_norm_for_sample(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: str,
    max_length: int = 8192
) -> Dict[str, Any]:
    """
    Calculate gradient norm for a single sample (prompt + response).

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        response: The response/output text
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Dictionary containing loss and gradient norm statistics
    """
    # Clear any existing gradients
    model.zero_grad()

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

    # Forward pass with labels to get loss
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss

    # Backward pass to compute gradients
    loss.backward()

    # Calculate gradient norm (L2 norm across all parameters)
    total_norm = 0.0
    num_params = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1

    total_norm = total_norm ** 0.5

    result = {
        'loss': loss.item(),
        'gradient_norm': total_norm,
        'num_params_with_grad': num_params,
        'sequence_length': input_ids.shape[1],
        'response_length': input_ids.shape[1] - prompt_length
    }

    # Clean up
    model.zero_grad()
    del inputs, input_ids, attention_mask, prompt_tokens, labels, outputs, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def process_dataset(
    model: nn.Module,
    tokenizer: Any,
    data: List[Dict[str, Any]],
    device: str,
    max_length: int = 8192,
    max_questions: int = None,
    max_answers_per_question: int = None
) -> List[Dict[str, Any]]:
    """
    Process all questions and answers in the dataset.
    Adds gradient_norm and loss directly into each answer object.
    """
    questions_to_process = data[:max_questions] if max_questions else data

    for question_item in tqdm(questions_to_process, desc="Processing questions"):
        question = question_item['question']
        answers = question_item['answers']

        # Prepare prompt
        prompt = prepare_prompt(question, tokenizer)

        # Limit answers if specified
        answers_to_process = answers[:max_answers_per_question] if max_answers_per_question else answers

        for answer_item in answers_to_process:
            answer_text = answer_item.get('answer', '')

            # Skip if answer is empty
            if not answer_text:
                continue

            grad_result = calculate_gradient_norm_for_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                response=answer_text,
                device=device,
                max_length=max_length
            )

            # Add gradient norm info directly into the answer object
            answer_item['loss'] = grad_result['loss']
            answer_item['gradient_norm'] = grad_result['gradient_norm']
            answer_item['response_length'] = grad_result['response_length']

    return data


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate gradient norm for each answer in the dataset"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Model name on HuggingFace or path to checkpoint"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/math12K_merged_answers_with_Qwen2.5_Math_7B_loglikelihood.json",
        help="Path to input JSON file with questions and answers"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ntk/results/gradient_norms.json",
        help="Path to save output JSON"
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
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all)"
    )
    parser.add_argument(
        "--max_answers",
        type=int,
        default=None,
        help="Maximum answers per question (default: all)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GRADIENT NORM CALCULATION")
    print("=" * 80)
    print(f"\nModel: {args.model_name_or_path}")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        args.device
    )

    # Load data
    print(f"\nLoading data from {args.input_path}...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions")

    # Process dataset - this modifies data in-place
    print("\nCalculating gradient norms...")
    updated_data = process_dataset(
        model=model,
        tokenizer=tokenizer,
        data=data,
        device=device,
        max_length=args.max_length,
        max_questions=args.max_questions,
        max_answers_per_question=args.max_answers
    )

    # Compute and print summary statistics
    all_losses = []
    all_grad_norms = []
    source_stats = {}

    for question_item in updated_data:
        for ans in question_item.get('answers', []):
            if 'gradient_norm' in ans:
                loss = ans['loss']
                grad_norm = ans['gradient_norm']
                source = ans.get('source', 'unknown')

                all_losses.append(loss)
                all_grad_norms.append(grad_norm)

                if source not in source_stats:
                    source_stats[source] = {'losses': [], 'grad_norms': []}
                source_stats[source]['losses'].append(loss)
                source_stats[source]['grad_norms'].append(grad_norm)

    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    print(f"Total questions: {len(updated_data)}")
    print(f"Total answers processed: {len(all_losses)}")
    if all_losses:
        print(f"\nOverall Loss: {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}")
        print(f"Overall Gradient Norm: {np.mean(all_grad_norms):.4f} ± {np.std(all_grad_norms):.4f}")

        print("\nBy Source:")
        for source, stats in source_stats.items():
            print(f"  {source} (n={len(stats['losses'])})")
            print(f"    Loss: {np.mean(stats['losses']):.4f} ± {np.std(stats['losses']):.4f}")
            print(f"    Gradient Norm: {np.mean(stats['grad_norms']):.4f} ± {np.std(stats['grad_norms']):.4f}")

    # Save updated data with gradient norms added to each answer
    save_results(updated_data, args.output_path)

    # Clean up
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
