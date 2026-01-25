#!/usr/bin/env python3
"""
Script to generate solutions for questions in JSON files and evaluate them immediately.
Uses vLLM for efficient batch generation and evaluates correctness against ground truth.

This combines generation (like generate_solutions.py) with evaluation logic.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from math import comb

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'eval'))

from utils.parser import parse_question, parse_ground_truth, extract_answer
from utils.grader import check_is_correct


def load_json_file(file_path):
    """Load JSON file and return data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_file(data, file_path):
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl_file(data, file_path):
    """Save data to JSONL file (one JSON object per line)."""
    temp_file = file_path + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")
    os.rename(temp_file, file_path)


def prepare_prompts(data, tokenizer, instruction=None):
    """
    Prepare prompts from data entries.

    Args:
        data: List of entries with 'instruction' and 'input' fields
        tokenizer: The tokenizer to use
        instruction: Optional instruction to override entry instructions

    Returns:
        List of formatted prompts
    """
    prompts = []

    for entry in data:
        instr = instruction if instruction else entry.get("instruction", "")
        question = entry.get("input", entry.get("question", ""))

        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": f"{instr}\n\n{question}"}
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    return prompts


def calculate_pass_at_k(n_correct, n_total, k):
    """
    Calculate pass@k metric.

    Args:
        n_correct: Number of correct answers
        n_total: Total number of attempts
        k: k value for pass@k

    Returns:
        pass@k value
    """
    if n_correct == 0:
        return 0.0
    if n_total - n_correct < k:
        return 1.0
    return 1.0 - (comb(n_total - n_correct, k) / comb(n_total, k))


def generate_and_evaluate(
    json_file,
    model_name,
    output_dir,
    data_name="math",
    temperature=0.7,
    max_tokens=2048,
    batch_size=None,
    n_sampling=1,
    p=0.9,
    num_gpus=2,
    k=1,
    output_format="json"
):
    """
    Generate solutions for questions in a JSON file and evaluate them.

    Args:
        json_file: Path to input JSON file
        model_name: Model name or path
        output_dir: Output directory for results
        data_name: Dataset name for answer extraction (default: "math")
        temperature: Sampling temperature (0 for greedy)
        max_tokens: Maximum tokens to generate
        batch_size: Number of samples to process (None = all)
        n_sampling: Number of solutions per question
        p: Top-p (nucleus sampling) parameter
        num_gpus: Number of GPUs to use
        k: k value for pass@k calculation
        output_format: Output format ("json" or "jsonl")
    """
    print(f"\n{'='*80}")
    print(f"GENERATE AND EVALUATE")
    print(f"{'='*80}")
    print(f"Input file: {json_file}")
    print(f"Model: {model_name}")
    print(f"Data name: {data_name}")
    print(f"{'='*80}\n")

    # Load data
    data = load_json_file(json_file)
    print(f"✓ Loaded {len(data)} entries from {json_file}")

    # Limit batch size if specified
    if batch_size is not None:
        data = data[:batch_size]
        print(f"Processing first {batch_size} entries")

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded")

    # Prepare prompts
    print("\nPreparing prompts...")
    prompts = prepare_prompts(data, tokenizer)
    print(f"✓ Prepared {len(prompts)} prompts")

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
    print(f"  Using {num_gpus} GPU(s)")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        gpu_memory_utilization=0.5
    )
    print("✓ Model loaded")

    # Generate solutions
    print(f"\n{'='*80}")
    print(f"Generating solutions for {len(data)} questions...")
    print(f"{'='*80}")
    completions = llm.generate(prompts, sampling_params)
    print("✓ Generation complete")

    # Process results and evaluate
    print(f"\n{'='*80}")
    print("Processing and evaluating results...")
    print(f"{'='*80}")

    results = []
    correct_count = 0
    pass_at_k_list = []

    for i, (entry, completion) in enumerate(tqdm(zip(data, completions), total=len(data), desc="Evaluating")):
        # Extract generated solutions
        generated_solutions = [output.text for output in completion.outputs]

        # Get question and ground truth
        question = entry.get('input', entry.get('question', ''))

        # Parse ground truth - handle different data formats
        if 'answer' in entry:
            gt_answer = entry['answer']
        elif 'output' in entry:
            gt_answer = entry['output']
        else:
            # Try to use parser
            try:
                _, gt_answer = parse_ground_truth(entry, data_name)
            except:
                gt_answer = None
                print(f"Warning: Could not extract ground truth for question {i}")

        # Extract answers from generated solutions
        generated_answers = []
        for solution in generated_solutions:
            try:
                answer = extract_answer(solution, data_name)
                generated_answers.append(answer)
            except:
                generated_answers.append(None)

        # Check correctness
        is_correct_list = []
        for gen_answer in generated_answers:
            if gen_answer is not None and gt_answer is not None:
                try:
                    is_correct = check_is_correct(gen_answer, gt_answer, timeout=True)
                    is_correct_list.append(is_correct)
                except:
                    is_correct_list.append(False)
            else:
                is_correct_list.append(False)

        is_correct_any = any(is_correct_list)
        if is_correct_any:
            correct_count += 1

        # Calculate pass@k if multiple samples
        if n_sampling > 1:
            n_correct = sum(is_correct_list)
            pass_at_k_value = calculate_pass_at_k(n_correct, len(is_correct_list), k)
            pass_at_k_list.append(pass_at_k_value)

        # Prepare result entry
        result = {
            "question": question,
            "generated_responses": generated_solutions if n_sampling > 1 else generated_solutions[0],
            "generated_answers": generated_answers if n_sampling > 1 else generated_answers[0],
            "gold_answer": gt_answer,
            "is_correct": is_correct_any,
        }

        if n_sampling > 1:
            result["answers_correctness"] = is_correct_list

        # Preserve original fields
        if "instruction" in entry:
            result["instruction"] = entry["instruction"]
        if "id" in entry:
            result["id"] = entry["id"]
        if "source" in entry:
            result["source"] = entry["source"]

        results.append(result)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    input_filename = Path(json_file).stem
    if output_format == "jsonl":
        output_filename = f"{input_filename}_evaluated.jsonl"
        output_path = os.path.join(output_dir, output_filename)
        print(f"\nSaving results to {output_path}...")
        save_jsonl_file(results, output_path)
    else:
        output_filename = f"{input_filename}_evaluated.json"
        output_path = os.path.join(output_dir, output_filename)
        print(f"\nSaving results to {output_path}...")
        save_json_file(results, output_path)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved {len(results)} results ({file_size:.2f} MB)")

    # Print evaluation statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    accuracy = correct_count / len(results) if len(results) > 0 else 0
    print(f"\nCorrect: {correct_count}/{len(results)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    if pass_at_k_list:
        avg_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"\nPass@{k}: {avg_pass_at_k:.4f} ({avg_pass_at_k*100:.2f}%)")
        print(f"(Based on {len(pass_at_k_list)} samples with multiple generations)")

    # Show sample
    if results:
        print("\n" + "-"*80)
        print("SAMPLE RESULT (first entry)")
        print("-"*80)
        sample = results[0]
        print(f"Question: {sample['question'][:200]}...")
        if n_sampling == 1:
            print(f"\nGenerated solution: {str(sample['generated_responses'])[:200]}...")
            print(f"Generated answer: {sample['generated_answers']}")
        else:
            print(f"\nGenerated solutions (showing first): {str(sample['generated_responses'][0])[:200]}...")
            print(f"Generated answers: {sample['generated_answers'][:3]}...")  # Show first 3
        print(f"Ground truth: {sample['gold_answer']}")
        print(f"Correct: {sample['is_correct']}")

    print("\n" + "="*80)
    print(f"✓ COMPLETE! Results saved to {output_path}")
    print("="*80)

    return results, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Generate solutions and evaluate them immediately"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSON file with questions"
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
        "--data_name",
        type=str,
        default="math",
        help="Dataset name for answer extraction (default: math)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)"
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
        help="Top-p (nucleus sampling) parameter (default: 1.0)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="k value for pass@k calculation (default: 1)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output file format (default: json)"
    )

    args = parser.parse_args()

    # Generate and evaluate
    generate_and_evaluate(
        json_file=args.input_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        data_name=args.data_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n_sampling=args.n_sampling,
        p=args.p,
        num_gpus=args.num_gpus,
        k=args.k,
        output_format=args.output_format
    )


if __name__ == "__main__":
    # Example usage (can be uncommented for testing):
    # generate_and_evaluate(
    #     json_file="data/pool_high_1k.json",
    #     model_name="Qwen/Qwen2.5-Math-7B",
    #     output_dir="./temp",
    #     data_name="math",
    #     temperature=0.0,
    #     n_sampling=1
    # )

    main()
