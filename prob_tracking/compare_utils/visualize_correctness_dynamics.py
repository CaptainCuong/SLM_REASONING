#!/usr/bin/env python3
"""
Visualize correct/incorrect dynamics of questions over training checkpoints.

This script reads evaluation results from a base model and multiple checkpoints,
then visualizes how the correctness of each question changes during training.
"""

import json
import re
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def find_checkpoints(model_dir: str) -> List[Tuple[int, str]]:
    """
    Find all checkpoint directories in the model folder.

    Args:
        model_dir: Path to model folder containing checkpoints

    Returns:
        List of (step_number, checkpoint_path) tuples, sorted by step number
    """
    model_path = Path(model_dir)
    checkpoints = []

    for item in model_path.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint'):
            match = re.search(r'checkpoint-(\d+)', item.name)
            if match:
                step_num = int(match.group(1))
                checkpoints.append((step_num, str(item)))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of entries."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_results_file(directory: str, dataset: str) -> Optional[str]:
    """Find the results JSONL file for a given dataset in a directory."""
    dataset_dir = Path(directory) / dataset
    if not dataset_dir.exists():
        return None

    # Look for JSONL files
    jsonl_files = list(dataset_dir.glob('*.jsonl'))
    if jsonl_files:
        return str(jsonl_files[0])

    return None


def load_correctness_data(
    base_model_dir: str,
    checkpoints_dir: str,
    dataset: str
) -> Tuple[Dict[str, bool], Dict[int, Dict[str, bool]], List[int]]:
    """
    Load correctness data from base model and all checkpoints.

    Returns:
        Tuple of:
        - base_correctness: {question_id: is_correct}
        - checkpoint_correctness: {step: {question_id: is_correct}}
        - steps: sorted list of checkpoint steps
    """
    # Load base model results
    base_file = find_results_file(base_model_dir, dataset)
    base_correctness = {}

    if base_file:
        base_entries = load_jsonl(base_file)
        for idx, entry in enumerate(base_entries, start=1):
            q_id = str(idx)  # Use reading order as ID, starting from 1
            base_correctness[q_id] = entry.get('is_correct', False)
        print(f"Loaded {len(base_correctness)} questions from base model")
    else:
        print(f"Warning: No results found for base model at {base_model_dir}/{dataset}")

    # Find and load checkpoint results
    checkpoints = find_checkpoints(checkpoints_dir)
    checkpoint_correctness = {}
    steps = []

    for step, cp_path in checkpoints:
        cp_file = find_results_file(cp_path, dataset)
        if cp_file:
            entries = load_jsonl(cp_file)
            cp_correct = {}
            for idx, entry in enumerate(entries, start=1):
                q_id = str(idx)  # Use reading order as ID, starting from 1
                cp_correct[q_id] = entry.get('is_correct', False)
            checkpoint_correctness[step] = cp_correct
            steps.append(step)
            print(f"Loaded checkpoint-{step}: {len(cp_correct)} questions")

    return base_correctness, checkpoint_correctness, steps


def compute_dynamics(
    base_correctness: Dict[str, bool],
    checkpoint_correctness: Dict[int, Dict[str, bool]],
    steps: List[int]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute correctness dynamics for each question.

    Returns:
        Dictionary mapping question_id to dynamics info:
        - 'base': base model correctness
        - 'trajectory': list of (step, is_correct) tuples
        - 'flips': number of times correctness changed
        - 'final': final correctness
        - 'improved': True if went from incorrect to correct
        - 'degraded': True if went from correct to incorrect
    """
    # Get all question IDs
    all_questions = set(base_correctness.keys())
    for cp_data in checkpoint_correctness.values():
        all_questions.update(cp_data.keys())

    dynamics = {}

    for q_id in all_questions:
        base_correct = base_correctness.get(q_id, None)

        trajectory = []
        for step in steps:
            if q_id in checkpoint_correctness.get(step, {}):
                trajectory.append((step, checkpoint_correctness[step][q_id]))

        # Count flips
        flips = 0
        prev = base_correct
        for step, correct in trajectory:
            if prev is not None and correct != prev:
                flips += 1
            prev = correct

        # Determine final state
        final = trajectory[-1][1] if trajectory else base_correct

        # Determine if improved or degraded
        improved = base_correct == False and final == True
        degraded = base_correct == True and final == False

        dynamics[q_id] = {
            'base': base_correct,
            'trajectory': trajectory,
            'flips': flips,
            'final': final,
            'improved': improved,
            'degraded': degraded
        }

    return dynamics


def plot_correctness_heatmap(
    dynamics: Dict[str, Dict[str, Any]],
    steps: List[int],
    output_path: str = None,
    show: bool = True,
    title: str = "Correctness Dynamics Heatmap"
):
    """
    Plot a heatmap showing correctness of each question across checkpoints.

    Green = correct, Red = incorrect, Gray = no data
    """
    # Sort questions by ID (numeric if possible)
    try:
        sorted_questions = sorted(dynamics.keys(), key=lambda x: int(x))
    except ValueError:
        sorted_questions = sorted(dynamics.keys())

    n_questions = len(sorted_questions)
    n_steps = len(steps) + 1  # +1 for base model

    # Create matrix: 1 = correct, 0 = incorrect, -1 = no data
    matrix = np.full((n_questions, n_steps), -1, dtype=float)

    for i, q_id in enumerate(sorted_questions):
        data = dynamics[q_id]

        # Base model
        if data['base'] is not None:
            matrix[i, 0] = 1 if data['base'] else 0

        # Checkpoints
        step_to_idx = {step: idx + 1 for idx, step in enumerate(steps)}
        for step, correct in data['trajectory']:
            if step in step_to_idx:
                matrix[i, step_to_idx[step]] = 1 if correct else 0

    # Create custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#d62728', '#2ca02c', '#cccccc'])  # red, green, gray

    fig, ax = plt.subplots(figsize=(max(14, n_steps * 0.5), max(8, n_questions * 0.3)))

    # Normalize matrix for colormap (0=red, 1=green, 2=gray)
    display_matrix = np.where(matrix == -1, 2, matrix)

    im = ax.imshow(display_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

    # Set labels
    x_labels = ['Base'] + [f'{s}' for s in steps]
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

    ax.set_yticks(range(n_questions))
    ax.set_yticklabels([f'Q{q}' for q in sorted_questions], fontsize=8)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Question ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ca02c', label='Correct'),
        mpatches.Patch(facecolor='#d62728', label='Incorrect'),
        mpatches.Patch(facecolor='#cccccc', label='No data')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_accuracy_curve(
    dynamics: Dict[str, Dict[str, Any]],
    steps: List[int],
    output_path: str = None,
    show: bool = True,
    title: str = "Accuracy Over Training"
):
    """Plot overall accuracy across training steps."""
    # Calculate accuracy at each step
    all_steps = [0] + steps  # 0 = base model
    accuracies = []

    for step_idx, step in enumerate(all_steps):
        correct = 0
        total = 0

        for q_id, data in dynamics.items():
            if step == 0:
                # Base model
                if data['base'] is not None:
                    total += 1
                    if data['base']:
                        correct += 1
            else:
                # Checkpoint
                for s, c in data['trajectory']:
                    if s == step:
                        total += 1
                        if c:
                            correct += 1
                        break

        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy * 100)

    plt.figure(figsize=(12, 6))
    plt.plot(all_steps, accuracies, marker='o', linewidth=2, markersize=8, color='#1f77b4')
    plt.fill_between(all_steps, accuracies, alpha=0.3)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Annotate base and final accuracy
    plt.annotate(f'Base: {accuracies[0]:.1f}%',
                xy=(all_steps[0], accuracies[0]),
                xytext=(10, 10), textcoords='offset points', fontsize=10)
    plt.annotate(f'Final: {accuracies[-1]:.1f}%',
                xy=(all_steps[-1], accuracies[-1]),
                xytext=(-50, 10), textcoords='offset points', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_flip_distribution(
    dynamics: Dict[str, Dict[str, Any]],
    output_path: str = None,
    show: bool = True
):
    """Plot distribution of correctness flips."""
    flips = [data['flips'] for data in dynamics.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of flips
    max_flips = max(flips) if flips else 0
    bins = range(0, max_flips + 2)
    axes[0].hist(flips, bins=bins, edgecolor='black', alpha=0.7, color='#1f77b4')
    axes[0].set_xlabel('Number of Flips', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Correctness Flips', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Pie chart: improved vs degraded vs stable
    improved = sum(1 for d in dynamics.values() if d['improved'])
    degraded = sum(1 for d in dynamics.values() if d['degraded'])
    stable_correct = sum(1 for d in dynamics.values()
                        if d['base'] == True and d['final'] == True)
    stable_incorrect = sum(1 for d in dynamics.values()
                          if d['base'] == False and d['final'] == False)

    labels = ['Improved', 'Degraded', 'Stable Correct', 'Stable Incorrect']
    sizes = [improved, degraded, stable_correct, stable_incorrect]
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e']
    explode = (0.05, 0.05, 0, 0)

    # Filter out zero values
    non_zero = [(l, s, c, e) for l, s, c, e in zip(labels, sizes, colors, explode) if s > 0]
    if non_zero:
        labels, sizes, colors, explode = zip(*non_zero)
        axes[1].pie(sizes, labels=labels, colors=colors, explode=explode,
                   autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Question Outcome Distribution', fontsize=12, fontweight='bold')

    plt.suptitle('Correctness Dynamics Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def print_summary(dynamics: Dict[str, Dict[str, Any]], steps: List[int]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("Correctness Dynamics Summary")
    print("=" * 80)

    total = len(dynamics)
    improved = sum(1 for d in dynamics.values() if d['improved'])
    degraded = sum(1 for d in dynamics.values() if d['degraded'])
    stable_correct = sum(1 for d in dynamics.values()
                        if d['base'] == True and d['final'] == True)
    stable_incorrect = sum(1 for d in dynamics.values()
                          if d['base'] == False and d['final'] == False)

    print(f"\nTotal questions: {total}")
    print(f"  Improved (incorrect → correct): {improved} ({improved/total*100:.1f}%)")
    print(f"  Degraded (correct → incorrect): {degraded} ({degraded/total*100:.1f}%)")
    print(f"  Stable correct: {stable_correct} ({stable_correct/total*100:.1f}%)")
    print(f"  Stable incorrect: {stable_incorrect} ({stable_incorrect/total*100:.1f}%)")

    # Questions with most flips
    flip_counts = [(q_id, d['flips']) for q_id, d in dynamics.items()]
    flip_counts.sort(key=lambda x: x[1], reverse=True)

    print("\nQuestions with most flips:")
    for q_id, flips in flip_counts[:5]:
        if flips > 0:
            print(f"  Q{q_id}: {flips} flips")

    # List improved questions
    if improved > 0:
        print("\nImproved questions:")
        for q_id, d in dynamics.items():
            if d['improved']:
                print(f"  Q{q_id}")

    # List degraded questions
    if degraded > 0:
        print("\nDegraded questions:")
        for q_id, d in dynamics.items():
            if d['degraded']:
                print(f"  Q{q_id}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize correct/incorrect dynamics over training checkpoints"
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="eval/outputs/Qwen/Qwen2.5-Math-7B",
        help="Directory containing base model evaluation results (e.g., eval/outputs/Qwen/Qwen2.5-Math-7B)"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint folders (e.g., eval/outputs/cuongdc/Qwen_Math_high)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (subfolder in checkpoint dirs, e.g., amc, math, amc_test)"
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
        choices=['all', 'heatmap', 'accuracy', 'flips'],
        default='all',
        help="Type of plot to generate (default: all)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CORRECTNESS DYNAMICS VISUALIZATION")
    print("=" * 80)
    print(f"\nBase model: {args.base_model_dir}")
    print(f"Checkpoints: {args.checkpoints_dir}")
    print(f"Dataset: {args.dataset}")

    # Load data
    print("\nLoading evaluation results...")
    base_correctness, checkpoint_correctness, steps = load_correctness_data(
        args.base_model_dir,
        args.checkpoints_dir,
        args.dataset
    )

    if not steps:
        print("Error: No checkpoint results found!")
        return

    # Compute dynamics
    print("\nComputing dynamics...")
    dynamics = compute_dynamics(base_correctness, checkpoint_correctness, steps)

    # Print summary
    print_summary(dynamics, steps)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    show = not args.no_show

    # Generate plots
    if args.plot_type in ['all', 'heatmap']:
        print("\nGenerating correctness heatmap...")
        plot_correctness_heatmap(
            dynamics, steps,
            output_path=f"{args.output_dir}/correctness_heatmap_{args.dataset}.png",
            show=show,
            title=f"Correctness Dynamics - {args.dataset}"
        )

    if args.plot_type in ['all', 'accuracy']:
        print("\nGenerating accuracy curve...")
        plot_accuracy_curve(
            dynamics, steps,
            output_path=f"{args.output_dir}/accuracy_curve_{args.dataset}.png",
            show=show,
            title=f"Accuracy Over Training - {args.dataset}"
        )

    if args.plot_type in ['all', 'flips']:
        print("\nGenerating flip distribution...")
        plot_flip_distribution(
            dynamics,
            output_path=f"{args.output_dir}/flip_distribution_{args.dataset}.png",
            show=show
        )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()