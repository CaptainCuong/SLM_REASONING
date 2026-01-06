#!/usr/bin/env python3
"""
Script to convert JSONL files to JSON format.
JSONL format: one JSON object per line
JSON format: array of JSON objects
"""

import json
import argparse
import os
from pathlib import Path


def convert_jsonl_to_json(jsonl_path, json_path=None, indent=2, reformat=False):
    """
    Convert a JSONL file to JSON format.

    Args:
        jsonl_path: Path to input JSONL file
        json_path: Path to output JSON file (optional, auto-generated if not provided)
        indent: Indentation level for JSON output (default: 2)
        reformat: If True, reformat to instruction/input/output format (default: False)

    Returns:
        json_path: Path to the created JSON file
    """
    # Validate input file
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    # Auto-generate output path if not provided
    if json_path is None:
        input_path = Path(jsonl_path)
        json_path = input_path.parent / f"{input_path.stem}.json"

    print(f"Converting {jsonl_path} to {json_path}...")
    print("="*60)

    # Read JSONL file
    data = []
    line_count = 0
    error_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                original = json.loads(line)

                # Reformat if requested
                if reformat:
                    formatted = {
                        "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
                        "input": original.get("problem", original.get("input", "")),
                        # "output": original.get("solution", original.get("output", ""))
                    }
                    data.append(formatted)
                else:
                    data.append(original)

                line_count += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                error_count += 1

    print(f"Loaded {line_count} entries")
    if error_count > 0:
        print(f"Skipped {error_count} lines with errors")

    # Save as JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    # Calculate file size
    file_size = os.path.getsize(json_path) / (1024**2)  # MB

    print("="*60)
    print(f"✓ Conversion successful!")
    print(f"  Output file: {json_path}")
    print(f"  Entries: {len(data)}")
    print(f"  File size: {file_size:.2f} MB")
    print("="*60)

    return json_path


def analyze_json_structure(json_path):
    """
    Analyze and display the structure of the JSON file.

    Args:
        json_path: Path to JSON file
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("\nJSON Structure Analysis:")
    print("="*60)
    print(f"Total entries: {len(data)}")

    if len(data) > 0:
        print(f"\nFirst entry keys: {list(data[0].keys())}")

        # Analyze field types
        print("\nField types:")
        for key in data[0].keys():
            value_type = type(data[0][key]).__name__
            print(f"  {key}: {value_type}")

        # Show sample entry
        print("\nSample entry:")
        print(json.dumps(data[0], indent=2, ensure_ascii=False)[:300] + "...")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSONL file to JSON format'
    )

    parser.add_argument(
        'jsonl_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Path to output JSON file (default: same name as input with .json extension)'
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='Indentation level for JSON output (default: 2)'
    )
    parser.add_argument(
        '--no-indent',
        action='store_true',
        help='Output compact JSON (no indentation)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze and display JSON structure after conversion'
    )
    parser.add_argument(
        '--reformat',
        action='store_true',
        help='Reformat to instruction/input/output format (maps problem→input, solution→output, instruction="")'
    )

    args = parser.parse_args()

    # Set indent
    indent = None if args.no_indent else args.indent

    # Convert file
    json_path = convert_jsonl_to_json(
        args.jsonl_file,
        args.output,
        indent=indent,
        reformat=args.reformat
    )

    # Analyze if requested
    if args.analyze:
        analyze_json_structure(json_path)


if __name__ == "__main__":
    main()
