#!/usr/bin/env python3
"""
Merge JSON files matching a glob pattern into a single output file.
"""

import json
import glob
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merge JSON part files into a single file.")
    parser.add_argument("--pattern", type=str, required=True,
                        help="Glob pattern to match input files (e.g. 'data/math12K_lowest_likelihood_part*.json')")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output file path (e.g. 'data/math12K_lowest_likelihood.json')")
    args = parser.parse_args()

    part_files = sorted(glob.glob(args.pattern))

    print(f"Found {len(part_files)} part files:")
    for f in part_files:
        print(f"  - {f}")

    if not part_files:
        print("No part files found!")
        return

    merged_data = []
    for part_file in part_files:
        print(f"\nLoading {part_file}...")
        with open(part_file, 'r', encoding='utf-8') as f:
            part_data = json.load(f)
        print(f"  Loaded {len(part_data)} items")
        merged_data.extend(part_data)

    print(f"\nTotal merged items: {len(merged_data)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"\nSaving to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"Saved ({file_size:.2f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
