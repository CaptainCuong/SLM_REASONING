#!/usr/bin/env python3
"""
Add evaluation JSONL samples to an existing pool JSON file.

Reads a JSONL file with fields:
    question, generated_responses, is_correct / answers_correctness, ...

Converts each entry to pool format:
    {
        "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
        "input":  <question>,
        "output": <generated_response>,
        "type":   "id_{qid}_{tag}_{correct/incorrect}"
    }

and appends them to an existing pool JSON file.

Usage:
    python add_eval_samples_to_pool.py \
        --eval_file eval/outputs/Qwen/Qwen2.5-1.5B/math12k_paraphrased_qwen72b/test_qwen-instruct_t0.7_k1_s0_e50.jsonl \
        --pool_file prob_tracking/data/track_training_set/train_low_1.5B.json \
        --tag base_paraphrased

    python add_eval_samples_to_pool.py \
        --eval_file eval/outputs/Qwen/Qwen2.5-1.5B/math12k_paraphrased_qwen72b/test_qwen-instruct_t0.7_k1_s0_e50.jsonl \
        --pool_file prob_tracking/data/track_test_set/test_amc_low_1.5B.json \
        --tag base
"""

import json
import argparse

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add evaluation JSONL samples to an existing pool JSON file"
    )
    parser.add_argument(
        "--eval_file", type=str, required=True,
        help="Input JSONL file with evaluation results"
    )
    parser.add_argument(
        "--pool_file", type=str, required=True,
        help="Existing pool JSON file to append to"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output pool JSON file (default: overwrite pool_file)"
    )
    parser.add_argument(
        "--tag", type=str, required=True,
        help="Tag for the type field, e.g. 'base' -> id_1_base_correct"
    )
    parser.add_argument(
        "--start_id", type=int, default=1,
        help="Starting question ID (default: 1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load existing pool
    with open(args.pool_file, 'r', encoding='utf-8') as f:
        pool = json.load(f)

    print(f"Existing pool: {len(pool)} entries from {args.pool_file}")

    # Load eval results
    eval_entries = []
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_entries.append(json.loads(line.strip()))

    print(f"Eval file: {len(eval_entries)} entries from {args.eval_file}")

    # Convert and append
    new_count = 0
    for i, entry in enumerate(eval_entries):
        qid = args.start_id + i
        question = entry["question"]
        responses = entry["generated_responses"]
        correctness_list = entry.get("answers_correctness", [entry.get("is_correct", False)])

        for j, response in enumerate(responses):
            is_correct = correctness_list[j] if j < len(correctness_list) else False
            suffix = "correct" if is_correct else "incorrect"
            type_field = f"id_{qid}_{args.tag}_{suffix}"

            pool.append({
                "instruction": INSTRUCTION,
                "input": question,
                "output": response,
                "type": type_field,
            })
            new_count += 1

    # Save
    output_file = args.output_file or args.pool_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)

    print(f"Added {new_count} new entries")
    print(f"Total pool: {len(pool)} entries -> {output_file}")


if __name__ == "__main__":
    main()
