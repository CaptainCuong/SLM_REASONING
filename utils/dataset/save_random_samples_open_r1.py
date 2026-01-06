#!/usr/bin/env python3
"""
Script to load OpenR1-Math-220k dataset and save 1,000 random samples in JSON format.
"""

from datasets import load_dataset
import json
import os
import random
import argparse


def main(args):
    print("Loading dataset: open-r1/OpenR1-Math-220k")
    dataset = load_dataset("open-r1/OpenR1-Math-220k")

    print("\nDataset downloaded successfully!")
    print(f"\nDataset structure:")
    print(dataset)

    # Access the train split
    train_data = dataset['train']
    total_samples = len(train_data)

    print(f"\nTotal number of examples: {total_samples}")

    # Determine number of samples to save
    num_samples = min(args.num_samples, total_samples)
    print(f"Saving {num_samples} random samples...")

    # Create random indices
    random.seed(args.seed)
    random_indices = random.sample(range(total_samples), num_samples)
    random_indices.sort()  # Sort for more efficient access

    # Select random samples
    sampled_data = train_data.select(random_indices)

    # Convert to the format: [{"instruction": ..., "input": ..., "output": ...}]
    formatted_data = []
    for example in sampled_data:
        if args.generated:
            formatted_example = {
                "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
                "input": example['problem'],
                "output": example['generations'][0]
            }
        else:
            formatted_example = {
                "instruction": "",
                "input": example['problem'],
                "output": example['solution']
            }

        # Optionally include problem_type
        if args.include_type:
            formatted_example['problem_type'] = example['problem_type']

        formatted_data.append(formatted_example)

    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON file - append '_generated' if using generated solutions
    if args.generated and not args.output_filename.endswith('_generated.json'):
        # Insert '_generated' before the .json extension
        base_name = args.output_filename.rsplit('.json', 1)[0]
        output_filename_final = f"{base_name}_generated.json"
    else:
        output_filename_final = args.output_filename

    output_filename = os.path.join(output_dir, output_filename_final)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {num_samples} random examples to {output_filename}")
    print(f"Random seed used: {args.seed}")
    print("\nProcessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save random samples from OpenR1-Math-220k dataset.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of random samples to save (default: 1000)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--generated",
        action="store_true",
        help="Use generated solutions instead of original solutions."
    )
    parser.add_argument(
        "--include_type",
        action="store_true",
        help="Include problem_type field in the output."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for the JSON file (default: data)."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="openr1_random_1k.json",
        help="Output filename (default: openr1_random_1k.json)."
    )
    args = parser.parse_args()
    main(args)
