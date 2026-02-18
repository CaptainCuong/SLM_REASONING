#!/usr/bin/env python3
"""
Calculate log likelihood for each answer using a specified model.

This script loads a model and calculates log likelihood for each answer
in a JSON file containing questions and their answers.
"""

import gc
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = None,
    torch_dtype: torch.dtype = torch.bfloat16
) -> Tuple[Any, Any, str]:
    """Load model and tokenizer."""
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

    model.eval()
    print(f"Model loaded on {device}")

    return model, tokenizer, device


def prepare_prompt(question: str, tokenizer: Any) -> str:
    """Prepare a prompt from question using chat template."""
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


def calculate_log_likelihood(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: str,
    max_length: int = 32768
) -> Dict[str, Any]:
    """
    Calculate log likelihood for a response given a prompt.

    Returns:
        Dictionary containing log_likelihood, perplexity, and token_count
    """
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

    # Create labels: -100 for prompt tokens (ignored in loss)
    labels = input_ids.clone()
    labels[0, :prompt_length] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    loss = outputs.loss.item()
    response_length = input_ids.shape[1] - prompt_length

    # Calculate log likelihood (negative loss * number of tokens)
    log_likelihood = -loss * response_length

    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(loss)).item()

    # Calculate average token probability
    avg_token_prob = torch.exp(torch.tensor(-loss)).item()

    result = {
        'log_likelihood': log_likelihood,
        'perplexity': perplexity,
        'average_token_prob': avg_token_prob,
        'token_count': response_length
    }

    # Clean up
    del inputs, input_ids, attention_mask, prompt_tokens, labels, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def get_model_short_name(model_path: str) -> str:
    """Extract a short name from model path for field naming."""
    # Get the last part of the path
    name = Path(model_path).name
    # Replace special characters with underscores
    name = name.replace('-', '_').replace('.', '_')
    return name


def process_data(
    data: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: str,
    field_name: str,
    max_length: int = 32768
) -> List[Dict[str, Any]]:
    """Process all questions and answers, adding log likelihood field."""
    total_answers = sum(len(q.get('answers', [])) for q in data)

    with tqdm(total=total_answers, desc="Calculating log likelihoods") as pbar:
        for question_data in data:
            question = question_data.get('question', '')
            prompt = prepare_prompt(question, tokenizer)

            for answer_entry in question_data.get('answers', []):
                answer_text = answer_entry.get('answer', '')

                try:
                    result = calculate_log_likelihood(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        response=answer_text,
                        device=device,
                        max_length=max_length
                    )

                    # Add the new field
                    answer_entry[field_name] = result['log_likelihood']
                    answer_entry[f"{field_name}_perplexity"] = result['perplexity']
                    answer_entry[f"{field_name}_avg_prob"] = result['average_token_prob']
                    answer_entry[f"{field_name}_tokens"] = result['token_count']

                except Exception as e:
                    print(f"\nError processing answer: {e}")
                    answer_entry[field_name] = None

                pbar.update(1)

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Calculate log likelihood for answers using a model"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Model name or path (default: Qwen/Qwen2.5-Math-7B)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/math12K_merged_answers_with_Qwen2.5_Math_7B_loglikelihood.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON file path (default: overwrites input)"
    )
    parser.add_argument(
        "--field_name",
        type=str,
        default=None,
        help="Field name for log likelihood (default: <model_name>_llh)"
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

    # Determine field name
    if args.field_name is None:
        model_short_name = get_model_short_name(args.model_name_or_path)
        field_name = f"{model_short_name}_llh"
    else:
        field_name = args.field_name

    # Determine output path
    if args.output_path is None:
        output_path = args.input_path
    else:
        output_path = args.output_path

    print("=" * 70)
    print("LOG LIKELIHOOD CALCULATION")
    print("=" * 70)
    print(f"Model: {args.model_name_or_path}")
    print(f"Input: {args.input_path}")
    print(f"Output: {output_path}")
    print(f"Field name: {field_name}")

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        args.device
    )

    # Load data
    print(f"\nLoading data from {args.input_path}...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions")
    total_answers = sum(len(q.get('answers', [])) for q in data)
    print(f"Total answers to process: {total_answers}")

    # Process data
    print("\nProcessing...")
    data = process_data(
        data=data,
        model=model,
        tokenizer=tokenizer,
        device=device,
        field_name=field_name,
        max_length=args.max_length
    )

    # Save results
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Clean up
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
