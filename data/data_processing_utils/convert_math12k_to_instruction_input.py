"""
Convert data/math12K_merged_loglikelihood.json to a JSON file with just
"instruction" and "input" fields for each item.
"""

import json
import os

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def main():
    src = os.path.join(os.path.dirname(__file__), "math12K_merged_loglikelihood.json")
    dst = os.path.join(os.path.dirname(__file__), "math12K_instruction_input.json")

    with open(src, "r") as f:
        data = json.load(f)

    result = []
    for item in data:
        result.append({
            "instruction": INSTRUCTION,
            "input": item["question"],
        })

    with open(dst, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(result)} samples to {dst}")


if __name__ == "__main__":
    main()
