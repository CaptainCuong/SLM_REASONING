#!/usr/bin/env python3
"""
Script to load OpenR1-Math-220k dataset and split it by problem_type,
saving 1000 samples per category in JSON format.
"""

from datasets import load_dataset
import json
import os
import argparse

def main(args):
    print("Loading dataset: open-r1/OpenR1-Math-220k")
    dataset = load_dataset("open-r1/OpenR1-Math-220k")

    print("\nDataset downloaded successfully!")
    print(f"\nDataset structure:")
    print(dataset)

    # Access the train split
    train_data = dataset['train']

    # Define target categories
    target_categories = [
        "Algebra",
        "Geometry",
        "Number Theory",
        "Combinatorics",
        "Logic and Puzzles",
        "Calculus",
        "Inequalities"
    ]

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Splitting dataset by problem_type")
    print("="*50)

    # Get the problem_type field from the train split
    problem_types = train_data['problem_type']
    print(f"\nTotal number of examples: {len(problem_types)}")
    print(f"Total number of unique problem types: {len(set(problem_types))}")

    # Filter and save each category
    for category in target_categories:
        print(f"\nProcessing category: {category}")

        # Filter examples for this category
        category_data = train_data.filter(lambda x: x['problem_type'] == category)

        if len(category_data) == 0:
            print(f"  Warning: No examples found for '{category}'")
            continue

        # Take first 1000 samples (or all if less than 1000)
        num_samples = min(1000, len(category_data))
        sampled_data = category_data.select(range(num_samples))

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
            formatted_data.append(formatted_example)

        # Save to JSON file
        if args.generated:
            output_filename = f"{output_dir}/{category.lower().replace(' ', '_')}_generated.json"
        else:
            output_filename = f"{output_dir}/{category.lower()}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved {num_samples} examples to {output_filename}")
        print(f"  Total available: {len(category_data)} examples")

    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split OpenR1-Math-220k dataset by problem type.")
    parser.add_argument(
        "--generated",
        action="store_true",
        help="Indicate if the dataset contains generated solutions."
    )
    args = parser.parse_args()
    main(args)
