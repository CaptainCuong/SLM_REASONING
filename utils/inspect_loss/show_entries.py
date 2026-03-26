#!/usr/bin/env python3
"""
Print questions and answers matching a regex pattern on the 'type' field.

Input: a tracking JSON file (list of entries with 'instruction', 'input', 'output', 'type').

The --pattern argument is a regex applied to the 'type' field. Shell-style
wildcards are also supported (e.g. "id_4_*" is converted to "id_4_.*").

Usage:
    # All entries for question 4 (shell-style wildcard):
    python utils/inspect/show_entries.py prob_tracking/data/track_test_set/test_amc_low_3B.json -p "id_4_*"

    # All entries for question 4 (regex):
    python utils/inspect/show_entries.py prob_tracking/data/track_test_set/test_amc_low_3B.json -p "id_4_.*"

    # Only incorrect entries for question 10:
    python utils/inspect/show_entries.py prob_tracking/data/track_test_set/test_amc_low_3B.json -p "id_10_.*_incorrect"

    # Limit output length per answer:
    python utils/inspect/show_entries.py prob_tracking/data/track_test_set/test_amc_low_3B.json -p "id_4_*" -m 300

    # Write output to a file:
    python utils/inspect/show_entries.py prob_tracking/data/track_test_set/test_amc_low_3B.json -p "id_4_*" -o output.txt
"""

import json
import re
import sys
import argparse


def wildcard_to_regex(pattern: str) -> str:
    """Convert shell-style wildcard pattern to regex if needed.

    Replaces '*' with '.*' and '?' with '.', but only when the pattern
    does not already look like a regex (contains no regex-specific chars).
    """
    regex_special = set(r'\.+^${}()|[]')
    has_regex = any(c in regex_special for c in pattern)
    if not has_regex:
        # Treat as shell glob
        pattern = pattern.replace('*', '.*').replace('?', '.')
    return pattern


def show_entries(input_path: str, pattern: str, max_output_len: int, output_file: str = None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    regex_pattern = wildcard_to_regex(pattern)
    regex = re.compile(regex_pattern)

    matches = [(i, entry) for i, entry in enumerate(data) if regex.fullmatch(entry.get('type', ''))]

    out = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout

    print("=" * 80, file=out)
    print(f"File    : {input_path}", file=out)
    print(f"Pattern : {pattern}" + (f"  (regex: {regex_pattern})" if regex_pattern != pattern else ""), file=out)
    print(f"Matches : {len(matches)} / {len(data)}", file=out)
    print("=" * 80, file=out)

    for rank, (idx, entry) in enumerate(matches, 1):
        entry_type = entry.get('type', '')
        question = entry.get('input', '') or entry.get('instruction', '')
        answer = entry.get('output', '')
        truncated = len(answer) > max_output_len
        if truncated:
            answer = answer[:max_output_len] + "..."

        print(f"\n--- [{rank}] type: {entry_type}  (index: {idx}) ---", file=out)
        print(f"Q: {question}", file=out)
        print(f"A: {answer}", file=out)
        if truncated:
            print(f"   (output truncated to {max_output_len} chars)", file=out)

    if not matches:
        print(f"\nNo entries matched pattern '{pattern}'.", file=out)

    if output_file:
        out.close()
        print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Print questions and answers matching a regex on the 'type' field"
    )
    parser.add_argument(
        "input_path", type=str,
        help="Path to tracking JSON file"
    )
    parser.add_argument(
        "--pattern", "-p", type=str, required=True,
        help="Regex or wildcard pattern to match 'type' field (e.g. 'id_4_*', 'id_10_.*_incorrect')"
    )
    parser.add_argument(
        "--max_output_len", "-m", type=int, default=1000,
        help="Max characters of answer to display (default: 1000, use 0 for no limit)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output file (if not set, prints to stdout)"
    )
    args = parser.parse_args()

    max_len = args.max_output_len if args.max_output_len > 0 else float('inf')
    show_entries(args.input_path, args.pattern, max_len, args.output)


if __name__ == "__main__":
    main()
