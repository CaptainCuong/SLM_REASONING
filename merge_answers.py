"""
Merge answers from Gemma and Qwen models into a unified format.

This script merges answers from two JSON files:
- math35_gemma3_27bit_t06_all_4_answers.json (4 answers per question)
- math35_qwen25_72bin_16_answers_ques.json (16 answers per question)

Input format: {"questions": [...], "answers": [[...], [...]]}

Output format:
[
    {
        "question": "...",
        "answers": [
            {"answer": "...", "source": "gemma3_72b"},
            {"answer": "...", "source": "gemma3_72b"},
            ...
            {"answer": "...", "source": "qwen2.5_72b"},
            ...
        ]
    },
    ...
]
"""

import json
from typing import List, Dict


def merge_answer_files(
    gemma_file: str,
    qwen_file: str,
    output_file: str
):
    """
    Merge answers from Gemma and Qwen files.

    Args:
        gemma_file: Path to Gemma answers JSON file
        qwen_file: Path to Qwen answers JSON file
        output_file: Path to save merged output
    """
    print("=" * 80)
    print("Merging Answer Files")
    print("=" * 80)

    # Load Gemma file
    print(f"\nLoading {gemma_file}...")
    with open(gemma_file, 'r', encoding='utf-8') as f:
        gemma_data = json.load(f)
    print(f"  Gemma - Questions: {len(gemma_data['questions'])}")
    print(f"  Gemma - Answer sets: {len(gemma_data['answers'])}")
    if gemma_data['answers']:
        print(f"  Gemma - Answers per question: {len(gemma_data['answers'][0])}")

    # Load Qwen file
    print(f"\nLoading {qwen_file}...")
    with open(qwen_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    print(f"  Qwen - Questions: {len(qwen_data['questions'])}")
    print(f"  Qwen - Answer sets: {len(qwen_data['answers'])}")
    if qwen_data['answers']:
        print(f"  Qwen - Answers per question: {len(qwen_data['answers'][0])}")

    # Verify same number of questions
    gemma_questions = gemma_data['questions']
    gemma_answers = gemma_data['answers']
    qwen_questions = qwen_data['questions']
    qwen_answers = qwen_data['answers']

    if len(gemma_questions) != len(qwen_questions):
        print(f"\nWARNING: Different number of questions!")
        print(f"  Gemma: {len(gemma_questions)} questions")
        print(f"  Qwen:  {len(qwen_questions)} questions")
        num_questions = min(len(gemma_questions), len(qwen_questions))
        print(f"  Will process: {num_questions} questions")
    else:
        num_questions = len(gemma_questions)
        print(f"\n✓ Both files have {num_questions} questions")

    # Merge data
    merged_data = []

    print(f"\nMerging {num_questions} questions...")

    for i in range(num_questions):
        # Get question (use Gemma's question as primary)
        question = gemma_questions[i]

        # Verify questions match
        if qwen_questions[i] != question:
            print(f"\nWARNING: Questions differ at index {i}")
            print(f"  Gemma: {question[:80]}...")
            print(f"  Qwen:  {qwen_questions[i][:80]}...")

        # Get answers from both sources
        gemma_ans_list = gemma_answers[i]
        qwen_ans_list = qwen_answers[i]

        # Create answer list with source labels
        all_answers = []

        # Add Gemma answers
        for ans in gemma_ans_list:
            all_answers.append({
                "answer": ans,
                "source": "gemma3_72b"
            })

        # Add Qwen answers
        for ans in qwen_ans_list:
            all_answers.append({
                "answer": ans,
                "source": "qwen2.5_72b"
            })

        # Create merged entry
        merged_entry = {
            "question": question,
            "answers": all_answers
        }

        merged_data.append(merged_entry)

        # Print progress every 1000 questions
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{num_questions} questions...")

    # Save merged data
    print(f"\nSaving merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Total questions: {len(merged_data)}")

    if merged_data:
        total_answers = sum(len(entry['answers']) for entry in merged_data)
        avg_answers = total_answers / len(merged_data)
        print(f"Total answers: {total_answers}")
        print(f"Average answers per question: {avg_answers:.2f}")

        # Count answers by source
        gemma_count = sum(
            sum(1 for ans in entry['answers'] if ans['source'] == 'gemma3_72b')
            for entry in merged_data
        )
        qwen_count = sum(
            sum(1 for ans in entry['answers'] if ans['source'] == 'qwen2.5_72b')
            for entry in merged_data
        )

        print(f"\nAnswers by source:")
        print(f"  Gemma 3 72B:    {gemma_count}")
        print(f"  Qwen 2.5 72B:   {qwen_count}")

    print(f"\nOutput saved to: {output_file}")
    print("=" * 80)


def preview_merged_data(filepath: str, num_entries: int = 2):
    """Preview the merged data."""
    print("\n" + "=" * 80)
    print("PREVIEW OF MERGED DATA")
    print("=" * 80)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, entry in enumerate(data[:num_entries]):
        print(f"\nEntry {i + 1}:")
        print(f"  Question: {entry['question'][:100]}...")
        print(f"  Number of answers: {len(entry['answers'])}")
        print(f"  Answer sources:")

        for j, ans in enumerate(entry['answers'][:3]):  # Show first 3 answers
            print(f"    [{j+1}] Source: {ans['source']}")
            print(f"        Answer: {ans['answer'][:80]}...")

        if len(entry['answers']) > 3:
            print(f"    ... and {len(entry['answers']) - 3} more answers")


if __name__ == "__main__":
    # File paths
    gemma_file = "/workspace/SLM_REASONING/temp_str/math35_gemma3_27bit_t06_all_4_answers.json"
    qwen_file = "/workspace/SLM_REASONING/temp_str/math35_qwen25_72bin_16_answers_ques.json"
    output_file = "/workspace/SLM_REASONING/temp_str/math35_merged_answers.json"

    # Merge files
    merge_answer_files(gemma_file, qwen_file, output_file)

    # Preview results
    preview_merged_data(output_file, num_entries=2)
