#!/usr/bin/env python3
"""
Script to rename proj_iter_*.pt files by adding 25000 to each index.
Example: proj_iter_000000.pt -> proj_iter_025000.pt
"""

import os
import re
from pathlib import Path

def rename_proj_files(directory, offset=25000):
    """
    Rename files matching proj_iter_XXXXXX.pt by adding offset to the index.

    Args:
        directory: Path to the directory containing the files
        offset: Number to add to each index (default: 25000)
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Pattern to match proj_iter_XXXXXX.pt files
    pattern = re.compile(r'^proj_iter_(\d{6})\.pt$')

    # Get all matching files
    files_to_rename = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            old_index = int(match.group(1))
            new_index = old_index + offset
            old_path = directory / filename
            new_filename = f"proj_iter_{new_index:06d}.pt"
            new_path = directory / new_filename
            files_to_rename.append((old_path, new_path, old_index, new_index))

    # Sort by index in descending order to avoid conflicts during renaming
    files_to_rename.sort(key=lambda x: x[2], reverse=True)

    print(f"Found {len(files_to_rename)} files to rename")
    print(f"Adding {offset} to each index\n")

    if files_to_rename:
        print(f"First file: {files_to_rename[-1][0].name} -> {files_to_rename[-1][1].name}")
        print(f"Last file: {files_to_rename[0][0].name} -> {files_to_rename[0][1].name}")

        # Ask for confirmation
        response = input(f"\nProceed with renaming {len(files_to_rename)} files? (yes/no): ")
        if response.lower() != 'yes':
            print("Renaming cancelled")
            return

        # Perform renaming
        success_count = 0
        for old_path, new_path, old_idx, new_idx in files_to_rename:
            try:
                old_path.rename(new_path)
                success_count += 1
                if success_count % 1000 == 0:
                    print(f"Renamed {success_count} files...")
            except Exception as e:
                print(f"Error renaming {old_path.name}: {e}")

        print(f"\nSuccessfully renamed {success_count} out of {len(files_to_rename)} files")
    else:
        print("No files found matching pattern proj_iter_XXXXXX.pt")

if __name__ == "__main__":
    directory = "/projects/ai_safe/cuongdc/grad_proj_tulu_last_50k_qwen_7b_instruct_idx0_end/"
    rename_proj_files(directory, offset=25000)
