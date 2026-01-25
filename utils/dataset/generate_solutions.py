#!/usr/bin/env python3
"""
Script to generate solutions for questions in JSON files in the data folder.
Uses vLLM for efficient batch generation.
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_json_file(file_path):
    """Load JSON file and return data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_file(data, file_path):
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_prompts(data, tokenizer):
    """
    Prepare prompts from data entries.

    Args:
        data: List of entries with 'instruction' and 'input' fields
        tokenizer: The tokenizer to use

    Returns:
        List of formatted prompts
    """
    prompts = []

    for entry in data:
        instruction = entry["instruction"]
        question = entry["input"]

        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": f"{instruction}\n\n{question}"}
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    return prompts


def generate_solutions(json_file, model_name, output_dir, temperature=0.7,
                       max_tokens=2048, batch_size=None, n_sampling=1, p=0.9):
    """
    Generate solutions for questions in a JSON file.

    Args:
        json_file: Path to input JSON file
        model_name: Model name or path
        output_dir: Output directory for results
        temperature: Sampling temperature (0 for greedy)
        max_tokens: Maximum tokens to generate
        batch_size: Number of samples to process (None = all)
        n_sampling: Number of solutions per question
        p: Top-p (nucleus sampling) parameter (default: 0.9)
    """
    print(f"\n{'='*80}")
    print(f"Processing: {json_file}")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")

    # Load data
    data = load_json_file(json_file)
    print(f"Loaded {len(data)} entries from {json_file}")

    # Limit batch size if specified
    if batch_size is not None:
        data = data[:batch_size]
        print(f"Processing first {batch_size} entries")

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Prepare prompts
    print("Preparing prompts...")
    prompts = prepare_prompts(data, tokenizer)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_sampling,
        top_p=1.0 if temperature == 0 else p
    )

    print(f"\nSampling parameters:")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  N sampling: {n_sampling}")
    print(f"  Top-p: {sampling_params.top_p}")

    # Initialize vLLM
    print(f"\nLoading model with vLLM...")
    # Auto-detect number of available GPUs
    # num_gpus = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').count(',')) + 1 if 'CUDA_VISIBLE_DEVICES' in os.environ else len([f for f in os.listdir('/dev') if f.startswith('nvidia')]) if os.path.exists('/dev') else 1
    num_gpus = 2
    print(f"  Using {num_gpus} GPU(s)")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        gpu_memory_utilization=0.5
    )

    # Generate solutions
    print(f"\nGenerating solutions for {len(data)} questions...")
    completions = llm.generate(prompts, sampling_params)

    # Process results
    print("\nProcessing results...")
    results = []
    for i, (entry, completion) in enumerate(zip(data, completions)):
        generated_solutions = [output.text for output in completion.outputs]

        result = {
            "instruction": entry.get('instruction', ''),
            "input": entry.get('input', ''),
            "output": generated_solutions if n_sampling > 1 else generated_solutions[0]
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(data)} questions...")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    input_filename = Path(json_file).stem
    output_filename = f"{input_filename}_solutions.json"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\nSaving results to {output_path}...")
    save_json_file(results, output_path)

    # Print sample
    print("\n" + "="*80)
    print("Sample result (first entry):")
    print("="*80)
    # print(f"Question: {results[0]['input'][:200]}...")
    # print(f"\nGenerated solution: {str(results[0]['generated_solutions'])[:300]}...")

    print("\n" + "="*80)
    print(f"✓ Complete! Saved {len(results)} results to {output_path}")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate solutions for questions in JSON files"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSON file (e.g., data/algebra.json)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Model name or path (default: Qwen/Qwen2.5-Math-7B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./temp",
        help="Output directory (default: ./temp)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (default: 0.7, use 0 for greedy)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Maximum tokens to generate (default: 32768)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--n_sampling",
        type=int,
        default=1,
        help="Number of solutions per question (default: 1)"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1.0,
        help="Top-p (nucleus sampling) parameter (default: 0.9)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all JSON files in data folder (non-generated)"
    )

    args = parser.parse_args()

    if args.all:
        # Process all non-generated JSON files in data folder
        data_dir = "data"
        json_files = [
            f for f in Path(data_dir).glob("*.json")
            if not f.stem.endswith(("_generated", "_info", "_solutions"))
        ]

        print(f"\nFound {len(json_files)} JSON files to process:")
        for f in json_files:
            print(f"  - {f}")

        for json_file in json_files:
            try:
                generate_solutions(
                    json_file=str(json_file),
                    model_name=args.model_name,
                    output_dir=args.output_dir,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                    n_sampling=args.n_sampling,
                    p=args.p
                )
            except Exception as e:
                print(f"\n✗ Error processing {json_file}: {e}")
                continue
    else:
        # Process single file
        generate_solutions(
            json_file=args.input_file,
            model_name=args.model_name,
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            n_sampling=args.n_sampling,
            p=args.p
        )


if __name__ == "__main__":
    main()
