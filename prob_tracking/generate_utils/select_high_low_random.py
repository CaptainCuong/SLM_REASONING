"""
Select highest, lowest, and random answers from math12K_merged_loglikelihood.json
by a given log-likelihood field.

For each of the top N questions, selects:
  - The answer with the highest value of the given field
  - The answer with the lowest value of the given field
  - 2 random answers from gemma3_72b (excluding highest/lowest)
  - 2 random answers from qwen2.5_72b (excluding highest/lowest)

Output format matches prob_tracking pool format:
  { "instruction": ..., "input": ..., "output": ..., "type": ... }

Usage:
    python select_high_low_random.py --field Qwen2.5_Math_1.5B_llh [--seed 42]
"""

import argparse
import json
import math
import random
import os

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, required=True,
                        help="The log-likelihood field to rank by, e.g. Qwen2.5_Math_1.5B_llh")
    parser.add_argument("--input", type=str,
                        default="data/math12K_merged_loglikelihood.json",
                        help="Path to the input JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to the output JSON file (default: auto-generated)")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of questions to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def is_valid_value(v):
    """Check if the value is a valid finite number."""
    if v is None:
        return False
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return False
    return True


def select_answers(question_data, field, question_id):
    """Select highest, lowest, and random answers for a single question."""
    question = question_data["question"]
    answers = question_data["answers"]

    # Filter answers with valid values for the field
    valid_answers = [a for a in answers if field in a and is_valid_value(a[field])]

    if len(valid_answers) < 2:
        return []

    # Sort by the field value
    sorted_answers = sorted(valid_answers, key=lambda a: a[field])

    highest = sorted_answers[-1]
    lowest = sorted_answers[0]

    # Remaining answers (excluding highest and lowest)
    remaining = [a for a in valid_answers if a is not highest and a is not lowest]

    # Split remaining by source
    remaining_gemma = [a for a in remaining if a["source"] == "gemma3_72b"]
    remaining_qwen = [a for a in remaining if a["source"] == "qwen2.5_72b"]

    # Select 2 random from each source
    random_gemma = random.sample(remaining_gemma, min(2, len(remaining_gemma)))
    random_qwen = random.sample(remaining_qwen, min(2, len(remaining_qwen)))

    # Build results in pool format: instruction, input, output, type
    results = []

    def make_entry(answer, type_label):
        return {
            "instruction": INSTRUCTION,
            "input": question,
            "output": answer["answer"],
            "type": type_label,
        }

    results.append(make_entry(highest, f"id_{question_id}_highest_{field}"))
    results.append(make_entry(lowest, f"id_{question_id}_lowest_{field}"))

    for a in random_gemma:
        results.append(make_entry(a, f"id_{question_id}_random_gemma"))

    for a in random_qwen:
        results.append(make_entry(a, f"id_{question_id}_random_qwen"))

    return results


def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    top_n = min(args.top_n, len(data))
    all_selected = []

    for i in range(top_n):
        selected = select_answers(data[i], args.field, question_id=i + 1)
        all_selected.extend(selected)

    # Output path
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.input),
            f"selected_{args.field}.json"
        )

    with open(args.output, "w") as f:
        json.dump(all_selected, f, indent=2, ensure_ascii=False)

    print(f"Selected {len(all_selected)} answers from {top_n} questions")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
