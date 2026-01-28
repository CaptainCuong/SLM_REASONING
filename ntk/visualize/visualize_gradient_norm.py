#!/usr/bin/env python3
"""
Visualize gradient norm dynamics across training checkpoints.

This script loads gradient norm results and creates plots showing
how gradient norms evolve for each question across checkpoints.
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
    """Load gradient norm results JSON."""
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
        Dictionary mapping question_id to dict with 'steps', 'gradient_norms', 'losses', 'correctness'
    """
    questions = defaultdict(lambda: {
        'steps': [],
        'gradient_norms': [],
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
        questions[question_id]['gradient_norms'].append(stats['avg_gradient_norm'])
        questions[question_id]['losses'].append(stats['avg_loss'])
        questions[question_id]['correctness'].append(correctness)

    # Sort by step
    for q_id in questions:
        sorted_indices = np.argsort(questions[q_id]['steps'])
        questions[q_id]['steps'] = [questions[q_id]['steps'][i] for i in sorted_indices]
        questions[q_id]['gradient_norms'] = [questions[q_id]['gradient_norms'][i] for i in sorted_indices]
        questions[q_id]['losses'] = [questions[q_id]['losses'][i] for i in sorted_indices]
        questions[q_id]['correctness'] = [questions[q_id]['correctness'][i] for i in sorted_indices]

    return dict(questions)


def plot_gradient_norms_by_question(
    questions: Dict[int, Dict[str, Any]],
    output_path: str = None,
    show: bool = True,
    title: str = "Gradient Norm Dynamics by Question"
):
    """
    Plot gradient norms for each question across training steps.

    Args:
        questions: Dictionary organized by question ID
        output_path: Path to save plot
        show: Whether to display plot
        title: Plot title
    """
    plt.figure(figsize=(14, 8))

    # Generate colors
    num_questions = len(questions)
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_questions, 20)))

    for idx, (q_id, data) in enumerate(sorted(questions.items())):
        steps = data['steps']
        grad_norms = data['gradient_norms']
        correctness = data['correctness']

        # Create color based on question
        color = colors[idx % len(colors)]

        # Plot line
        plt.plot(steps, grad_norms, marker='o', label=f'Q{q_id}',
                linewidth=2, color=color, markersize=6)

        # Mark incorrect points with 'x'
        for i, (s, g, c) in enumerate(zip(steps, grad_norms, correctness)):
            if c == 'incorrect':
                plt.scatter([s], [g], marker='x', color='red', s=100, zorder=5)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Add note about markers
    plt.figtext(0.5, 0.02, 'Red X marks indicate incorrect predictions',
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


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


def plot_correct_vs_incorrect(
    questions: Dict[int, Dict[str, Any]],
    output_path: str = None,
    show: bool = True
):
    """
    Plot comparison of gradient norms for correct vs incorrect predictions.
    """
    correct_norms = []
    incorrect_norms = []

    for q_id, data in questions.items():
        for g, c in zip(data['gradient_norms'], data['correctness']):
            if c == 'correct':
                correct_norms.append(g)
            else:
                incorrect_norms.append(g)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    axes[0].boxplot([correct_norms, incorrect_norms], tick_labels=['Correct', 'Incorrect'])
    axes[0].set_ylabel('Gradient Norm', fontsize=12)
    axes[0].set_title('Gradient Norm Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(correct_norms, bins=20, alpha=0.7, label=f'Correct (n={len(correct_norms)})', color='green')
    axes[1].hist(incorrect_norms, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect_norms)})', color='red')
    axes[1].set_xlabel('Gradient Norm', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Gradient Norm Histogram', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()

    # Print statistics
    print("\n" + "=" * 60)
    print("Correct vs Incorrect Statistics")
    print("=" * 60)
    print(f"\nCorrect predictions (n={len(correct_norms)}):")
    print(f"  Mean gradient norm: {np.mean(correct_norms):.4f}")
    print(f"  Std gradient norm:  {np.std(correct_norms):.4f}")
    print(f"  Min gradient norm:  {np.min(correct_norms):.4f}")
    print(f"  Max gradient norm:  {np.max(correct_norms):.4f}")

    print(f"\nIncorrect predictions (n={len(incorrect_norms)}):")
    print(f"  Mean gradient norm: {np.mean(incorrect_norms):.4f}")
    print(f"  Std gradient norm:  {np.std(incorrect_norms):.4f}")
    print(f"  Min gradient norm:  {np.min(incorrect_norms):.4f}")
    print(f"  Max gradient norm:  {np.max(incorrect_norms):.4f}")


def create_summary_stats(questions: Dict[int, Dict[str, Any]]):
    """Print summary statistics for gradient norm dynamics."""
    print("\n" + "=" * 80)
    print("Gradient Norm Dynamics Summary")
    print("=" * 80)

    for q_id, data in sorted(questions.items()):
        grad_norms = data['gradient_norms']
        steps = data['steps']
        correctness = data['correctness']

        initial = grad_norms[0]
        final = grad_norms[-1]
        change = final - initial
        percent_change = (change / abs(initial)) * 100 if initial != 0 else 0

        correct_count = sum(1 for c in correctness if c == 'correct')
        incorrect_count = len(correctness) - correct_count

        print(f"\nQuestion {q_id}:")
        print(f"  Initial (step {steps[0]}): {initial:.4f}")
        print(f"  Final (step {steps[-1]}):   {final:.4f}")
        print(f"  Change: {change:+.4f} ({percent_change:+.2f}%)")
        print(f"  Min: {min(grad_norms):.4f} | Max: {max(grad_norms):.4f}")
        print(f"  Correct: {correct_count} | Incorrect: {incorrect_count}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize gradient norm dynamics from results JSON"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="ntk/results/gradient_norms.json",
        help="Path to gradient norms results JSON"
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
        choices=['all', 'gradient_norm', 'loss', 'comparison'],
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
    if args.plot_type in ['all', 'gradient_norm']:
        print("\nGenerating gradient norm plot...")
        plot_gradient_norms_by_question(
            questions,
            output_path=f"{args.output_dir}/gradient_norms_by_question.png",
            show=show
        )

    if args.plot_type in ['all', 'loss']:
        print("\nGenerating loss plot...")
        plot_loss_by_question(
            questions,
            output_path=f"{args.output_dir}/loss_by_question.png",
            show=show
        )

    if args.plot_type in ['all', 'comparison']:
        print("\nGenerating correct vs incorrect comparison...")
        plot_correct_vs_incorrect(
            questions,
            output_path=f"{args.output_dir}/correct_vs_incorrect.png",
            show=show
        )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
