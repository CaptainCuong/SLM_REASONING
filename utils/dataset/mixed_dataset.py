#!/usr/bin/env python3
"""
Script to load OpenR1-Math-220k dataset and create a mixed Algebra dataset
combining 1000 samples with solutions from 'generations' field and 1000 samples
with solutions from 'solution' field.
"""

from datasets import load_dataset
import json
import os

def main():
    print("Loading dataset: open-r1/OpenR1-Math-220k")
    dataset = load_dataset("open-r1/OpenR1-Math-220k")

    print("\nDataset downloaded successfully!")
    print(f"\nDataset structure:")
    print(dataset)

    # Access the train split
    train_data = dataset['train']

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Creating mixed Algebra dataset")
    print("="*50)

    # Filter Algebra examples
    print("\nFiltering Algebra examples...")
    algebra_data = train_data.filter(lambda x: x['problem_type'] == "Algebra")

    if len(algebra_data) == 0:
        print("Error: No Algebra examples found!")
        return

    print(f"Total Algebra examples available: {len(algebra_data)}")

    # Take first 1000 samples with 'generations' field
    num_samples_gen = min(1000, len(algebra_data))
    samples_generated = algebra_data.select(range(num_samples_gen))

    # Take next 1000 samples with 'solution' field (offset by 1000)
    start_idx = num_samples_gen
    end_idx = min(start_idx + 1000, len(algebra_data))
    num_samples_sol = end_idx - start_idx
    samples_solution = algebra_data.select(range(start_idx, end_idx))

    print(f"\nSamples with 'generations' field: {num_samples_gen}")
    print(f"Samples with 'solution' field: {num_samples_sol}")

    # Format data from 'generations' field
    formatted_data = []
    print("\nProcessing samples with 'generations' field...")
    for example in samples_generated:
        formatted_example = {
            "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "input": example['problem'],
            "output": example['generations'][0]
        }
        formatted_data.append(formatted_example)

    # Format data from 'solution' field
    print("Processing samples with 'solution' field...")
    for example in samples_solution:
        formatted_example = {
            "instruction": "",
            "input": example['problem'],
            "output": example['solution']
        }
        formatted_data.append(formatted_example)

    # Save to JSON file
    output_filename = f"{output_dir}/algebra_mixed.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'-'*50}")
    print(f"Total samples saved: {len(formatted_data)}")
    print(f"  - From 'generations': {num_samples_gen}")
    print(f"  - From 'solution': {num_samples_sol}")
    print(f"Saved to: {output_filename}")
    print("="*50)
    print("Processing complete!")
    print("="*50)

if __name__ == "__main__":
    main()