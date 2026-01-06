"""
Parse .pt files in folder /projects/ai_safe/cuongdc/grad_proj_tulu_last_50k_qwen_7b_instruct_idx0_end/
Find the smallest index from files with format proj_iter_xxxxxx.pt
"""

import os
import re
from pathlib import Path

# Directory containing .pt files
directory = "/projects/ai_safe/cuongdc/grad_proj_tulu_last_50k_qwen_7b_instruct_idx0_end/"

print(f"Parsing .pt files in: {directory}\n")

# Check if directory exists
if not os.path.exists(directory):
    print(f"ERROR: Directory does not exist!")
    exit(1)

# Pattern to match proj_iter_xxxxxx.pt files
pattern = re.compile(r'^proj_iter_(\d+)\.pt$')

# Find all matching files and extract indices
indices = []
matching_files = []

for file in Path(directory).iterdir():
    if file.is_file() and file.name.endswith('.pt'):
        match = pattern.match(file.name)
        if match:
            index = int(match.group(1))
            indices.append(index)
            matching_files.append((index, file.name))

# Sort and display results
if indices:
    indices.sort()
    matching_files.sort()

    print(f"Total matching .pt files: {len(indices):,}")
    print(f"\nSmallest index: {min(indices):,}")
    print(f"Largest index: {max(indices):,}")
    print(f"Range: {max(indices) - min(indices):,}")

    # Show first 10 files
    print(f"\nFirst 10 files (by index):")
    for i, (idx, filename) in enumerate(matching_files[:10], 1):
        print(f"  {i:2d}. {filename:30s} (index: {idx:,})")

    # Show last 10 files
    if len(matching_files) > 10:
        print(f"\nLast 10 files (by index):")
        for i, (idx, filename) in enumerate(matching_files[-10:], 1):
            print(f"  {i:2d}. {filename:30s} (index: {idx:,})")

    # Check for gaps in indices
    print(f"\nChecking for gaps in indices...")
    expected_indices = set(range(min(indices), max(indices) + 1))
    actual_indices = set(indices)
    missing_indices = expected_indices - actual_indices

    if missing_indices:
        print(f"Found {len(missing_indices):,} missing indices")
        if len(missing_indices) <= 20:
            print(f"Missing indices: {sorted(missing_indices)}")
        else:
            print(f"First 20 missing indices: {sorted(missing_indices)[:20]}")
    else:
        print(f"No gaps found - all indices from {min(indices)} to {max(indices)} are present")

else:
    print("No matching .pt files found!")
    print("\nAll .pt files in directory:")
    pt_files = [f.name for f in Path(directory).iterdir() if f.is_file() and f.name.endswith('.pt')]
    for f in pt_files[:10]:
        print(f"  - {f}")