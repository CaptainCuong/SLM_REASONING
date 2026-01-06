#!/usr/bin/env python3
"""
Script to randomly subsample any JSON dataset file.

This utility loads a JSON file containing a list of samples and randomly
selects a specified number of samples, saving them to a new JSON file.
"""

import json
import random
import os
import argparse


def main(args):
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples in input file: {len(data)}")

    # Check if dataset has enough samples
    if len(data) <= args.num_samples:
        print(f"Dataset has {len(data)} samples, which is <= {args.num_samples}")
        print("Saving all samples without subsampling...")
        selected_data = data
    else:
        # Randomly select the specified number of samples
        print(f"Randomly selecting {args.num_samples} samples with seed={args.seed}...")
        random.seed(args.seed)
        selected_data = random.sample(data, args.num_samples)

    print(f"Selected {len(selected_data)} samples")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save selected samples to output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved to {args.output_file}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly subsample a JSON dataset file"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/openr1_random_20k_generated.json",
        help="Path to input JSON file (default: data/openr1_random_20k_generated.json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/openr1_random_5k_generated.json",
        help="Path to output JSON file (default: data/openr1_random_1k_generated.json)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to randomly select (default: 5000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    main(args)
