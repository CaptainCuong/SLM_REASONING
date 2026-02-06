#!/usr/bin/env python3
"""
Merge math12K_merged_loglikelihood_part*.json files into a single file.
"""

import json
import glob
import os

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all part files
    pattern = os.path.join(data_dir, "math12K_merged_loglikelihood_part*.json")
    part_files = sorted(glob.glob(pattern))

    print(f"Found {len(part_files)} part files:")
    for f in part_files:
        print(f"  - {os.path.basename(f)}")

    if not part_files:
        print("No part files found!")
        return

    # Merge all parts
    merged_data = []
    for part_file in part_files:
        print(f"\nLoading {os.path.basename(part_file)}...")
        with open(part_file, 'r', encoding='utf-8') as f:
            part_data = json.load(f)
        print(f"  Loaded {len(part_data)} items")
        merged_data.extend(part_data)

    print(f"\nTotal merged items: {len(merged_data)}")

    # Save merged file
    output_path = os.path.join(data_dir, "math12K_merged_loglikelihood.json")
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved ({file_size:.2f} MB)")
    print("Done!")

if __name__ == "__main__":
    main()
