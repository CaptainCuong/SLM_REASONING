#!/usr/bin/env python3
"""
Visualize average log-likelihood dynamics across training checkpoints.

This script loads the checkpoint summary JSON and creates plots showing
how average log-likelihood evolves for each sample type, and optionally
per-question log-likelihood dynamics.
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def load_summary(summary_path: str):
    """Load checkpoint summary JSON."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_avg_log_likelihood(summary: dict, output_path: str = None, show: bool = True):
    """
    Plot average log-likelihood for all sample types.

    Args:
        summary: Checkpoint summary dictionary
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    steps = summary['steps']
    results_by_type = summary['results_by_type']

    # Create figure
    plt.figure(figsize=(12, 8))

    # Generate distinct colors for each line
    num_types = len(results_by_type)
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_types, 10)))

    # If more than 10 types, use additional color maps
    if num_types > 10:
        colors = list(colors)
        colors.extend(plt.cm.Set3(np.linspace(0, 1, num_types - 10)))

    # Plot each sample type with a unique color
    for idx, (sample_type, metrics) in enumerate(results_by_type.items()):
        avg_llh = metrics['avg_log_likelihood']

        # Filter out None values
        valid_indices = [i for i, val in enumerate(avg_llh) if val is not None]
        valid_steps = [steps[i] for i in valid_indices]
        valid_llh = [avg_llh[i] for i in valid_indices]

        if valid_llh:
            plt.plot(valid_steps, valid_llh, marker='o', label=sample_type,
                    linewidth=2, color=colors[idx % len(colors)])

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Average Log-Likelihood', fontsize=12)
    plt.title('Average Log-Likelihood Dynamics Across Training', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_perplexity(summary: dict, output_path: str = None, show: bool = True):
    """
    Plot average perplexity for all sample types.

    Args:
        summary: Checkpoint summary dictionary
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    steps = summary['steps']
    results_by_type = summary['results_by_type']

    plt.figure(figsize=(12, 8))

    # Generate distinct colors for each line
    num_types = len(results_by_type)
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_types, 10)))

    # If more than 10 types, use additional color maps
    if num_types > 10:
        colors = list(colors)
        colors.extend(plt.cm.Set3(np.linspace(0, 1, num_types - 10)))

    # Plot each sample type with a unique color
    for idx, (sample_type, metrics) in enumerate(results_by_type.items()):
        avg_ppl = metrics['avg_perplexity']

        # Filter out None values
        valid_indices = [i for i, val in enumerate(avg_ppl) if val is not None]
        valid_steps = [steps[i] for i in valid_indices]
        valid_ppl = [avg_ppl[i] for i in valid_indices]

        if valid_ppl:
            plt.plot(valid_steps, valid_ppl, marker='o', label=sample_type,
                    linewidth=2, color=colors[idx % len(colors)])

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Average Perplexity', fontsize=12)
    plt.title('Average Perplexity Dynamics Across Training', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def create_summary_stats(summary: dict):
    """
    Print summary statistics for log-likelihood dynamics.

    Args:
        summary: Checkpoint summary dictionary
    """
    steps = summary['steps']
    results_by_type = summary['results_by_type']

    print("\n" + "="*80)
    print("Log-Likelihood Dynamics Summary")
    print("="*80)

    for sample_type, metrics in results_by_type.items():
        avg_llh = metrics['avg_log_likelihood']

        # Filter out None values
        valid_llh = [x for x in avg_llh if x is not None]

        if valid_llh:
            initial = valid_llh[0]
            final = valid_llh[-1]
            improvement = final - initial
            percent_change = (improvement / abs(initial)) * 100 if initial != 0 else 0

            print(f"\n{sample_type}:")
            print(f"  Initial LLH: {initial:.4f}")
            print(f"  Final LLH:   {final:.4f}")
            print(f"  Change:      {improvement:+.4f} ({percent_change:+.2f}%)")
            print(f"  Min LLH:     {min(valid_llh):.4f} (step {steps[avg_llh.index(min(valid_llh))]})")
            print(f"  Max LLH:     {max(valid_llh):.4f} (step {steps[avg_llh.index(max(valid_llh))]})")

    print("\n" + "="*80)


def filter_summary_by_questions(summary: dict, question_ids: list) -> dict:
    """
    Filter summary to only include types matching the given question IDs.
    E.g., question_ids=[1,2] keeps only types like "id_1_*" and "id_2_*".
    """
    prefixes = tuple(f'id_{q_id}_' for q_id in question_ids)
    filtered_results = {
        k: v for k, v in summary['results_by_type'].items()
        if k.startswith(prefixes)
    }
    return {**summary, 'results_by_type': filtered_results}


def parse_type_string(type_str: str):
    """
    Parse type string to extract question_id, source, and correctness.

    Args:
        type_str: Type string like "id_4_base_correct" or "id_4_cp555_incorrect"

    Returns:
        Tuple of (question_id, source, correctness)
    """
    match = re.match(r'id_(\d+)_(base|cp(\d+))_(correct|incorrect)', type_str)
    if match:
        question_id = int(match.group(1))
        source = 'base' if match.group(2) == 'base' else match.group(3)
        correctness = match.group(4)
        return question_id, source, correctness
    return None, None, None


def organize_by_question(summary: dict):
    """
    Organize results by question ID, averaging log-likelihood across types
    that belong to the same question at each checkpoint.

    Returns:
        Dictionary mapping question_id to dict with 'steps', 'avg_llh', 'correctness'
    """
    steps = summary['steps']
    results_by_type = summary['results_by_type']

    questions = defaultdict(lambda: {
        'llh_by_step': defaultdict(list),
        'correctness_by_step': defaultdict(list)
    })

    for type_str, metrics in results_by_type.items():
        question_id, source, correctness = parse_type_string(type_str)
        if question_id is None:
            continue

        avg_llh = metrics['avg_log_likelihood']
        for i, step in enumerate(steps):
            if i < len(avg_llh) and avg_llh[i] is not None:
                questions[question_id]['llh_by_step'][step].append(avg_llh[i])
                questions[question_id]['correctness_by_step'][step].append(correctness)

    # Compute average across types for each question at each step
    organized = {}
    for q_id, data in questions.items():
        sorted_steps = sorted(data['llh_by_step'].keys())
        avg_llhs = [np.mean(data['llh_by_step'][s]) for s in sorted_steps]

        # Correctness: "correct" if any type is correct at that step
        correctness_list = []
        for s in sorted_steps:
            corrs = data['correctness_by_step'][s]
            correctness_list.append('correct' if 'correct' in corrs else 'incorrect')

        organized[q_id] = {
            'steps': sorted_steps,
            'avg_llh': avg_llhs,
            'correctness': correctness_list
        }

    return organized


def plot_llh_by_question(
    questions: dict,
    output_path: str = None,
    show: bool = True,
    title: str = "Log-Likelihood Dynamics by Question"
):
    """
    Plot log-likelihood for each question across training steps.
    """
    plt.figure(figsize=(14, 8))

    num_questions = len(questions)
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_questions, 20)))

    for idx, (q_id, data) in enumerate(sorted(questions.items())):
        steps = data['steps']
        avg_llh = data['avg_llh']
        correctness = data['correctness']

        color = colors[idx % len(colors)]

        plt.plot(steps, avg_llh, marker='o', label=f'Q{q_id}',
                linewidth=2, color=color, markersize=6)

        # Mark incorrect points
        for s, l, c in zip(steps, avg_llh, correctness):
            if c == 'incorrect':
                plt.scatter([s], [l], marker='x', color='red', s=100, zorder=5)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Log-Likelihood', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.figtext(0.5, 0.02, 'Red X marks indicate incorrect predictions',
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def create_question_summary_stats(questions: dict):
    """Print summary statistics for per-question log-likelihood dynamics."""
    print("\n" + "=" * 80)
    print("Per-Question Log-Likelihood Summary")
    print("=" * 80)

    for q_id, data in sorted(questions.items()):
        llhs = data['avg_llh']
        steps = data['steps']
        correctness = data['correctness']

        initial = llhs[0]
        final = llhs[-1]
        change = final - initial
        percent_change = (change / abs(initial)) * 100 if initial != 0 else 0

        correct_count = sum(1 for c in correctness if c == 'correct')
        incorrect_count = len(correctness) - correct_count

        print(f"\nQuestion {q_id}:")
        print(f"  Initial LLH (step {steps[0]}): {initial:.4f}")
        print(f"  Final LLH (step {steps[-1]}):   {final:.4f}")
        print(f"  Change: {change:+.4f} ({percent_change:+.2f}%)")
        print(f"  Min: {min(llhs):.4f} | Max: {max(llhs):.4f}")
        print(f"  Correct: {correct_count} | Incorrect: {incorrect_count}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize average log-likelihood dynamics from checkpoint summary"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="prob_tracking/results/Qwen_Math_high_all_checkpoints_summary.json",
        help="Path to checkpoint summary JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prob_tracking/image",
        help="Directory to save plots (default: prob_tracking/image)"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plots (only save)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=['all', 'llh', 'perplexity', 'questions'],
        default='all',
        help="Type of plot to generate (default: all)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated list of question IDs to visualize (e.g., '1,2,3'). "
             "If not specified, visualize all questions."
    )

    args = parser.parse_args()

    # Load summary
    print(f"Loading summary from {args.summary_path}...")
    summary = load_summary(args.summary_path)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    show = not args.no_show

    # Filter by question IDs if specified
    if args.questions is not None:
        requested_ids = [int(q.strip()) for q in args.questions.split(',')]
        summary = filter_summary_by_questions(summary, requested_ids)
        print(f"Filtered to question(s): {requested_ids} ({len(summary['results_by_type'])} types)")

    # Generate plots
    if args.plot_type in ['all', 'llh']:
        print("\nGenerating log-likelihood plot...")
        create_summary_stats(summary)
        plot_avg_log_likelihood(
            summary,
            output_path=f"{args.output_dir}/avg_llh.png",
            show=show
        )

    if args.plot_type in ['all', 'perplexity']:
        print("\nGenerating perplexity plot...")
        plot_perplexity(
            summary,
            output_path=f"{args.output_dir}/avg_perplexity.png",
            show=show
        )

    if args.plot_type in ['all', 'questions']:
        print("\nOrganizing data by question...")
        questions = organize_by_question(summary)
        print(f"Found {len(questions)} unique questions")

        create_question_summary_stats(questions)

        print("\nGenerating per-question log-likelihood plot...")
        plot_llh_by_question(
            questions,
            output_path=f"{args.output_dir}/llh_by_question.png",
            show=show
        )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
