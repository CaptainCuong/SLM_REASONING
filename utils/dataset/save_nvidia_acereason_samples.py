#!/usr/bin/env python3
"""
Script to load nvidia/AceReason-1.1-SFT dataset and save all samples
in the same format as OpenR1-Math-220k.
"""

from datasets import load_dataset
import json
import os
import random

def main():
    print("Loading dataset: nvidia/AceReason-1.1-SFT")
    dataset = load_dataset("nvidia/AceReason-1.1-SFT")

    print("\nDataset downloaded successfully!")
    print(f"\nDataset structure:")
    print(dataset)

    # Access the train split
    train_data = dataset['train']

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Processing nvidia/AceReason-1.1-SFT dataset")
    print("="*50)

    print(f"\nTotal number of examples: {len(train_data)}")
    print(f"Features: {train_data.features}")

    # Use all samples
    total_samples = len(train_data)
    num_samples = total_samples
    sampled_data = train_data

    print(f"\nProcessing {num_samples} samples...")

    # Convert to the format: [{"instruction": ..., "input": ..., "output": ...}]
    formatted_data = []
    for example in sampled_data:
        # Adapt to the nvidia/AceReason format
        # The dataset structure may vary, adjust field names as needed
        formatted_example = {
            "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "input": example['input'],
            "output": example['output']
        }
        formatted_data.append(formatted_example)

    # Save to JSON file
    output_filename = f"{output_dir}/nvidia_acereason_all.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved {num_samples} examples to {output_filename}")
    print(f"  Total available: {len(train_data)} examples")

    json.dumps(formatted_data[0], indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)

if __name__ == "__main__":
    main()
