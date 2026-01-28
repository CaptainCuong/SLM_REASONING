#!/usr/bin/env python3
"""
Visualize loss dynamics across training checkpoints.

This script loads loss results and creates plots showing
how losses evolve for each question across checkpoints.
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict


def load_results(results_path: str) -> Dict[str, Any]:
    """Load loss results JSON."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_type_string(type_str: str) -> Tuple[int, str, str]:
    """
    Parse type string to extract question_id, source, and correctness.

    Args:
        type_str: Type string like "id_4_base_correct" or "id_4_cp555_incorrect"

    Returns:
        Tuple of (question_id, source, correctness)
        source is "base" or checkpoint step number as string
    """
    # Pattern: id_{question_id}_{source}_{correctness}
    match = re.match(r'id_(\d+)_(base|cp(\d+))_(correct|incorrect)', type_str)
    if match:
        question_id = int(match.group(1))
        if match.group(2) == 'base':
            source = 'base'
        else:
            source = match.group(3)  # checkpoint number
        correctness = match.group(4)
        return question_id, source, correctness
    return None, None, None


def organize_by_question(summary_by_type: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Organize results by question ID.

    Returns:
        Dictionary mapping question_id to dict with 'steps', 'losses', 'correctness'
    """
    questions = defaultdict(lambda: {
        'steps': [],
        'losses': [],
        'correctness': []
    })

    for type_str, stats in summary_by_type.items():
        question_id, source, correctness = parse_type_string(type_str)
        if question_id is None:
            continue

        # Convert source to step number (base = 0)
        step = 0 if source == 'base' else int(source)

        questions[question_id]['steps'].append(step)
        questions[question_id]['losses'].append(stats['avg_loss'])
        questions[question_id]['correctness'].append(correctness)

    # Sort by step
    for q_id in questions:
        sorted_indices = np.argsort(questions[q_id]['steps'])
        questions[q_id]['steps'] = [questions[q_id]['steps'][i] for i in sorted_indices]
        questions[q_id]['losses'] = [questions[q_id]['losses'][i] for i in sorted_indices]
        questions[q_id]['correctness'] = [questions[q_id]['correctness'][i] for i in sorted_indices]

    return dict(questions)


def plot_loss_by_question(
    questions: Dict[int, Dict[str, Any]],
    output_path: str = None,
    show: bool = True,
    title: str = "Loss Dynamics by Question"
):
    """
    Plot loss for each question across training steps.
    """
    plt.figure(figsize=(14, 8))

    num_questions = len(questions)
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_questions, 20)))

    for idx, (q_id, data) in enumerate(sorted(questions.items())):
        steps = data['steps']
        losses = data['losses']
        correctness = data['correctness']

        color = colors[idx % len(colors)]

        plt.plot(steps, losses, marker='o', label=f'Q{q_id}',
                linewidth=2, color=color, markersize=6)

        # Mark incorrect points
        for i, (s, l, c) in enumerate(zip(steps, losses, correctness)):
            if c == 'incorrect':
                plt.scatter([s], [l], marker='x', color='red', s=100, zorder=5)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
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


def create_summary_stats(questions: Dict[int, Dict[str, Any]]):
    """Print summary statistics for loss dynamics."""
    print("\n" + "=" * 80)
    print("Loss Dynamics Summary")
    print("=" * 80)

    for q_id, data in sorted(questions.items()):
        losses = data['losses']
        steps = data['steps']
        correctness = data['correctness']

        initial = losses[0]
        final = losses[-1]
        change = final - initial
        percent_change = (change / abs(initial)) * 100 if initial != 0 else 0

        correct_count = sum(1 for c in correctness if c == 'correct')
        incorrect_count = len(correctness) - correct_count

        print(f"\nQuestion {q_id}:")
        print(f"  Initial loss (step {steps[0]}): {initial:.4f}")
        print(f"  Final loss (step {steps[-1]}):   {final:.4f}")
        print(f"  Change: {change:+.4f} ({percent_change:+.2f}%)")
        print(f"  Min: {min(losses):.4f} | Max: {max(losses):.4f}")
        print(f"  Correct: {correct_count} | Incorrect: {incorrect_count}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize loss dynamics from results JSON"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="ntk/results/losses.json",
        help="Path to loss results JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ntk/image",
        help="Directory to save plots (default: ntk/image)"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plots (only save)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=['all', 'loss'],
        default='all',
        help="Type of plot to generate (default: all)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated list of question IDs to visualize (e.g., '1,2,3'). If not specified, visualize all questions."
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_path}...")
    results = load_results(args.results_path)

    print(f"Model: {results['model_path']}")
    print(f"Pool: {results['pool_path']}")
    print(f"Total samples: {results['total_samples']}")

    # Organize by question
    questions = organize_by_question(results['summary_by_type'])
    print(f"Found {len(questions)} unique questions")

    # Filter questions if specified
    if args.questions is not None:
        requested_ids = [int(q.strip()) for q in args.questions.split(',')]
        available_ids = set(questions.keys())

        # Check for non-existent questions
        missing_ids = [q_id for q_id in requested_ids if q_id not in available_ids]
        if missing_ids:
            raise ValueError(
                f"Question ID(s) {missing_ids} do not exist. "
                f"Available question IDs: {sorted(available_ids)}"
            )

        # Filter to only requested questions
        questions = {q_id: questions[q_id] for q_id in requested_ids}
        print(f"Visualizing {len(questions)} selected question(s): {requested_ids}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Print summary statistics
    create_summary_stats(questions)

    show = not args.no_show

    # Generate plots
    if args.plot_type in ['all', 'loss']:
        print("\nGenerating loss plot...")
        plot_loss_by_question(
            questions,
            output_path=f"{args.output_dir}/loss_by_question.png",
            show=show
        )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
