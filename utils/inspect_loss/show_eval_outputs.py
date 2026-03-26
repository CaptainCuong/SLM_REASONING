#!/usr/bin/env python3
"""
Print questions and model-generated answers from an eval output JSONL file.

Each line of the JSONL is expected to have:
  question, generated_responses, generated_answers, gold_answer,
  is_correct, answers_correctness, id

Usage:
    # Print all questions and answers:
    python utils/inspect/show_eval_outputs.py eval/outputs/adaptive_3B_low/epoch_13/checkpoint-4400/amc/test_qwen-instruct_t0.7_k1_s0_e40.jsonl

    # Show only question with id=5:
    python utils/inspect/show_eval_outputs.py <file.jsonl> --id 5

    # Show only incorrect answers:
    python utils/inspect/show_eval_outputs.py <file.jsonl> --incorrect

    # Limit displayed response length:
    python utils/inspect/show_eval_outputs.py <file.jsonl> --max_len 500

    # Write output to a file:
    python utils/inspect/show_eval_outputs.py <file.jsonl> -o output.txt
"""

import json
import sys
import argparse


def load_jsonl(path: str):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_entry(entry: dict, rank: int, max_len: int, out) -> None:
    qid = entry.get('id', '?')
    question = entry.get('question', '')
    gold = entry.get('gold_answer', '')
    is_correct = entry.get('is_correct', None)
    responses = entry.get('generated_responses', [])
    gen_answers = entry.get('generated_answers', [])
    correctness = entry.get('answers_correctness', [])

    correct_str = "CORRECT" if is_correct else "WRONG"
    print(f"\n{'='*80}", file=out)
    print(f"[{rank}] id={qid}  |  {correct_str}  |  gold={gold}", file=out)
    print(f"{'='*80}", file=out)
    print(f"Q: {question}", file=out)

    for i, (resp, ans) in enumerate(zip(responses, gen_answers)):
        trial_correct = correctness[i] if i < len(correctness) else None
        trial_str = "correct" if trial_correct else "wrong"
        print(f"\n--- Response {i+1}/{len(responses)}  [{trial_str}]  extracted={ans} ---", file=out)
        truncated = max_len and len(resp) > max_len
        print(resp[:max_len] + ("..." if truncated else ""), file=out)
        if truncated:
            print(f"   (truncated to {max_len} chars)", file=out)


def main():
    parser = argparse.ArgumentParser(
        description="Print questions and answers from an eval output JSONL file"
    )
    parser.add_argument(
        "input_path", type=str,
        help="Path to eval output JSONL file"
    )
    parser.add_argument(
        "--id", type=int, default=None,
        help="Show only the entry with this id"
    )
    parser.add_argument(
        "--incorrect", action="store_true",
        help="Show only incorrect entries"
    )
    parser.add_argument(
        "--max_len", "-m", type=int, default=2000,
        help="Max characters of each response to display (default: 2000, use 0 for no limit)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write output to this file instead of stdout"
    )
    args = parser.parse_args()

    entries = load_jsonl(args.input_path)
    max_len = args.max_len if args.max_len > 0 else None

    # Filter
    filtered = entries
    if args.id is not None:
        filtered = [e for e in filtered if e.get('id') == args.id]
    if args.incorrect:
        filtered = [e for e in filtered if not e.get('is_correct', True)]

    out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout

    print(f"File    : {args.input_path}", file=out)
    print(f"Total   : {len(entries)}  |  Showing: {len(filtered)}", file=out)
    if args.id is not None:
        print(f"Filter  : id={args.id}", file=out)
    if args.incorrect:
        print(f"Filter  : incorrect only", file=out)

    for rank, entry in enumerate(filtered, 1):
        format_entry(entry, rank, max_len, out)

    if not filtered:
        print("\nNo entries matched the given filters.", file=out)

    if args.output:
        out.close()
        print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
