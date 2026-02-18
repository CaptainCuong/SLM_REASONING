"""
Convert chosen and rejected answer files into DPO format.

- Chosen answers from: data/math12K_highest_Qwen2_5_3B_Instruct_llh.json
- Rejected answers from: temp/math12K_instruction_input_solutions.json
- Output: data/math12K_dpo.json
"""

import json
import argparse
import os

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def convert_to_dpo(chosen_file: str, rejected_file: str, output_file: str):
    with open(chosen_file, "r", encoding="utf-8") as f:
        chosen_data = json.load(f)

    with open(rejected_file, "r", encoding="utf-8") as f:
        rejected_data = json.load(f)

    assert len(chosen_data) == len(rejected_data), (
        f"Mismatch: chosen has {len(chosen_data)}, rejected has {len(rejected_data)}"
    )

    dpo_data = []
    for chosen_item, rejected_item in zip(chosen_data, rejected_data):
        question = chosen_item["input"]

        dpo_entry = {
            "conversations": [
                {"from": "system", "value": INSTRUCTION},
                {"from": "human", "value": question},
            ],
            "chosen": {"from": "gpt", "value": chosen_item["output"]},
            "rejected": {"from": "gpt", "value": rejected_item["output"]},
        }
        dpo_data.append(dpo_entry)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Wrote {len(dpo_data)} DPO entries to {output_file} ({file_size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert chosen/rejected files to DPO format")
    parser.add_argument(
        "--chosen", type=str,
        default="/home/cuongdc/SLM_REASONING/data/math12K_highest_Qwen2_5_3B_Instruct_llh.json",
        help="Path to chosen answers file",
    )
    parser.add_argument(
        "--rejected", type=str,
        default="/home/cuongdc/SLM_REASONING/temp/math12K_instruction_input_solutions.json",
        help="Path to rejected answers file",
    )
    parser.add_argument(
        "--output", type=str,
        default="/home/cuongdc/SLM_REASONING/data/math12K_dpo.json",
        help="Output DPO file path",
    )
    args = parser.parse_args()

    convert_to_dpo(args.chosen, args.rejected, args.output)


if __name__ == "__main__":
    main()
