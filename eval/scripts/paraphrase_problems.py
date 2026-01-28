#!/usr/bin/env python3
"""
Paraphrase the 'problem' field in a JSONL file using Qwen model.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_name: str, device: str = "cuda"):
    """Load the Qwen model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer


def paraphrase_problem(model, tokenizer, problem: str, device: str = "cuda") -> str:
    """Paraphrase a single problem using the model."""
    prompt = f"""Paraphrase the following math problem. Keep all mathematical expressions, numbers, and LaTeX formatting exactly the same. Only rephrase the surrounding text to express the same question differently.

Original problem:
{problem}

Paraphrased problem:"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def process_jsonl(input_path: str, output_path: str, model_name: str, device: str = "cuda"):
    """Process the JSONL file and paraphrase all problems."""
    # Load model
    model, tokenizer = load_model(model_name, device)

    # Read input file
    print(f"Reading from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} entries")

    # Process each line
    results = []
    for line in tqdm(lines, desc="Paraphrasing"):
        data = json.loads(line.strip())
        original_problem = data.get('problem', '')

        if original_problem:
            paraphrased = paraphrase_problem(model, tokenizer, original_problem, device)
            data['problem_original'] = original_problem
            data['problem'] = paraphrased

        results.append(data)

    # Write output file
    print(f"Writing to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Paraphrase problems in a JSONL file")
    parser.add_argument(
        "--input",
        type=str,
        default="eval/data/amc/test.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: input_paraphrased.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Model name or path for paraphrasing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_paraphrased{input_path.suffix}")

    process_jsonl(args.input, args.output, args.model, args.device)


if __name__ == "__main__":
    main()
