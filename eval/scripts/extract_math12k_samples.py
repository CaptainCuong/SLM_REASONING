import json
import re
import argparse


def extract_boxed_answer(output_text):
    """Extract the answer from \\boxed{} in the output text."""
    # Handle nested braces by counting them
    pattern = r'\\boxed\{'
    match = re.search(pattern, output_text)
    if not match:
        return None

    start_idx = match.end()
    brace_count = 1
    idx = start_idx

    while idx < len(output_text) and brace_count > 0:
        if output_text[idx] == '{':
            brace_count += 1
        elif output_text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        return output_text[start_idx:idx-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract samples from math12K_highest_likelihood.json")
    parser.add_argument("--input", type=str, default="data/math12K_highest_likelihood.json",
                        help="Input JSON file path")
    parser.add_argument("--output", type=str, default="eval/data/math12k_top50.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to extract")
    args = parser.parse_args()

    # Load the input JSON file
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract first N samples
    samples = data[:args.num_samples]

    # Convert to new format
    converted = []
    for sample in samples:
        question = sample.get("input", "")
        output = sample.get("output", "")
        answer = extract_boxed_answer(output)

        converted.append({
            "question": question,
            "answer": answer
        })

    # Save in JSONL format
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Extracted {len(converted)} samples to {args.output}")


if __name__ == "__main__":
    main()
