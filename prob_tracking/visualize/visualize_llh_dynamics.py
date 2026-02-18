#!/usr/bin/env python3
"""
Visualize average log-likelihood dynamics across training checkpoints.

This script loads the checkpoint summary JSON and creates plots showing
how average log-likelihood evolves for each sample type, and optionally
per-question log-likelihood dynamics.
"""

import json
import re
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

# Source detection and color mapping
SOURCE_COLORS = {
    'base': '#1f77b4',   # blue
    'qwen': '#ff7f0e',   # orange
    'highest': '#ff7f0e',  # orange
    'gemma': '#2ca02c',  # green
    'lowest': '#2ca02c',  # green
}


def detect_source(type_id: str) -> str:
    """Detect source from a sample type ID by looking for keywords."""
    type_id_lower = type_id.lower()
    for source in ('lowest', 'highest','gemma', 'qwen', 'base'):
        if source in type_id_lower:
            return source
    return 'unknown'


def load_summary(summary_path: str):
    """Load checkpoint summary JSON."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_by_id_patterns(summary: dict, patterns: list) -> dict:
    """
    Filter results_by_type to only include types matching any of the given
    glob/fnmatch patterns.

    E.g., patterns=["id_9*"] matches "id_9_base_problem_paraphrased_greedy_incorrect".
    """
    filtered = {
        k: v for k, v in summary['results_by_type'].items()
        if any(fnmatch.fnmatch(k, p) for p in patterns)
    }
    return {**summary, 'results_by_type': filtered}


def plot_avg_log_likelihood(summaries, output_path: str = None, show: bool = True,
                           color_by_correctness: bool = False,
                           color_by_source: bool = False):
    """
    Plot average log-likelihood for all sample types across one or more summaries.

    Args:
        summaries: A single summary dict (legacy) or a list of (label, summary) tuples
        output_path: Path to save plot (optional)
        show: Whether to display plot
        color_by_correctness: If True, color lines green/red by correctness
        color_by_source: If True, color lines by source (base/qwen/gemma)
    """
    # Normalize input: single summary -> list of (label, summary)
    if isinstance(summaries, dict):
        summaries = [('', summaries)]

    # Collect all (label_prefix, sample_type, steps, avg_llh) entries
    all_entries = []
    for _, summary in summaries:
        steps = summary['steps']
        for sample_type, metrics in summary['results_by_type'].items():
            all_entries.append((sample_type, sample_type, steps, metrics['avg_log_likelihood']))

    # Create figure
    plt.figure(figsize=(12, 8))

    num_entries = len(all_entries)
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_entries, 10)))
    if num_entries > 10:
        colors = list(colors)
        colors.extend(plt.cm.Set3(np.linspace(0, 1, num_entries - 10)))

    if color_by_correctness:
        correct_entries = [e for e in all_entries if '_correct' in e[1] and '_incorrect' not in e[1]]
        incorrect_entries = [e for e in all_entries if '_incorrect' in e[1]]
        green_shades = plt.cm.Greens(np.linspace(0.3, 0.9, max(len(correct_entries), 1)))
        red_shades = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(incorrect_entries), 1)))
        correct_idx = 0
        incorrect_idx = 0

    for idx, (display_name, sample_type, steps, avg_llh) in enumerate(all_entries):
        valid_indices = [i for i, val in enumerate(avg_llh) if val is not None]
        valid_steps = [steps[i] for i in valid_indices]
        valid_llh = [avg_llh[i] for i in valid_indices]

        if valid_llh:
            if color_by_source:
                source = detect_source(sample_type)
                color = SOURCE_COLORS.get(source, '#7f7f7f')
            elif color_by_correctness:
                if '_incorrect' in sample_type:
                    color = red_shades[incorrect_idx % len(red_shades)]
                    incorrect_idx += 1
                else:
                    color = green_shades[correct_idx % len(green_shades)]
                    correct_idx += 1
            else:
                color = colors[idx % len(colors)]
            plt.plot(valid_steps, valid_llh, marker='o', label=display_name,
                    linewidth=2, color=color)

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


def plot_perplexity(summaries, output_path: str = None, show: bool = True,
                    color_by_correctness: bool = False,
                    color_by_source: bool = False):
    """
    Plot average perplexity for all sample types across one or more summaries.

    Args:
        summaries: A single summary dict (legacy) or a list of (label, summary) tuples
        output_path: Path to save plot (optional)
        show: Whether to display plot
        color_by_correctness: If True, color lines green/red by correctness
        color_by_source: If True, color lines by source (base/qwen/gemma)
    """
    if isinstance(summaries, dict):
        summaries = [('', summaries)]

    all_entries = []
    for _, summary in summaries:
        steps = summary['steps']
        for sample_type, metrics in summary['results_by_type'].items():
            all_entries.append((sample_type, sample_type, steps, metrics['avg_perplexity']))

    plt.figure(figsize=(12, 8))

    num_entries = len(all_entries)
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_entries, 10)))
    if num_entries > 10:
        colors = list(colors)
        colors.extend(plt.cm.Set3(np.linspace(0, 1, num_entries - 10)))

    if color_by_correctness:
        correct_entries = [e for e in all_entries if '_correct' in e[1] and '_incorrect' not in e[1]]
        incorrect_entries = [e for e in all_entries if '_incorrect' in e[1]]
        green_shades = plt.cm.Greens(np.linspace(0.3, 0.9, max(len(correct_entries), 1)))
        red_shades = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(incorrect_entries), 1)))
        correct_idx = 0
        incorrect_idx = 0

    for idx, (display_name, sample_type, steps, avg_ppl) in enumerate(all_entries):
        valid_indices = [i for i, val in enumerate(avg_ppl) if val is not None]
        valid_steps = [steps[i] for i in valid_indices]
        valid_ppl = [avg_ppl[i] for i in valid_indices]

        if valid_ppl:
            if color_by_source:
                source = detect_source(sample_type)
                color = SOURCE_COLORS.get(source, '#7f7f7f')
            elif color_by_correctness:
                if '_incorrect' in sample_type:
                    color = red_shades[incorrect_idx % len(red_shades)]
                    incorrect_idx += 1
                else:
                    color = green_shades[correct_idx % len(green_shades)]
                    correct_idx += 1
            else:
                color = colors[idx % len(colors)]
            plt.plot(valid_steps, valid_ppl, marker='o', label=display_name,
                    linewidth=2, color=color)

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


def create_summary_stats(summaries):
    """
    Print summary statistics for log-likelihood dynamics.

    Args:
        summaries: A single summary dict (legacy) or a list of (label, summary) tuples
    """
    if isinstance(summaries, dict):
        summaries = [('', summaries)]

    print("\n" + "="*80)
    print("Log-Likelihood Dynamics Summary")
    print("="*80)

    for label, summary in summaries:
        steps = summary['steps']
        results_by_type = summary['results_by_type']

        if label:
            print(f"\n--- {label} ---")

        for sample_type, metrics in results_by_type.items():
            avg_llh = metrics['avg_log_likelihood']
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


def organize_by_question(summaries):
    """
    Organize results by question ID, averaging log-likelihood across types
    that belong to the same question at each checkpoint.

    Correctness at each step is determined by the type whose generation source
    matches that step (e.g., id_40_cp555_incorrect -> step 555 is incorrect).

    Args:
        summaries: A single summary dict (legacy) or a list of (label, summary) tuples

    Returns:
        Dictionary mapping display_key to dict with 'steps', 'avg_llh', 'correctness'
        When multiple summaries are provided, keys are prefixed with the label.
    """
    if isinstance(summaries, dict):
        summaries = [('', summaries)]

    # Map source to step: 'base' -> 0, '555' -> 555, etc.
    def source_to_step(source):
        if source == 'base':
            return 0
        return int(source)

    all_organized = {}

    for _, summary in summaries:
        steps = summary['steps']
        results_by_type = summary['results_by_type']

        questions = defaultdict(lambda: {
            'llh_by_step': defaultdict(list),
            'correctness_by_source': {}
        })

        for type_str, metrics in results_by_type.items():
            question_id, source, correctness = parse_type_string(type_str)
            if question_id is None:
                continue

            avg_llh = metrics['avg_log_likelihood']
            for i, step in enumerate(steps):
                if i < len(avg_llh) and avg_llh[i] is not None:
                    questions[question_id]['llh_by_step'][step].append(avg_llh[i])

            src_step = source_to_step(source)
            questions[question_id]['correctness_by_source'][src_step] = correctness

        for q_id, data in questions.items():
            sorted_steps = sorted(data['llh_by_step'].keys())
            avg_llhs = [np.mean(data['llh_by_step'][s]) for s in sorted_steps]

            correctness_by_source = data['correctness_by_source']
            correctness_list = []
            for s in sorted_steps:
                if s in correctness_by_source:
                    correctness_list.append(correctness_by_source[s])
                else:
                    earlier = [src for src in correctness_by_source if src <= s]
                    if earlier:
                        correctness_list.append(correctness_by_source[max(earlier)])
                    else:
                        correctness_list.append('unknown')

            all_organized[q_id] = {
                'steps': sorted_steps,
                'avg_llh': avg_llhs,
                'correctness': correctness_list
            }

    return all_organized


def plot_llh_by_question(
    questions: dict,
    output_path: str = None,
    show: bool = True,
    title: str = "Log-Likelihood Dynamics by Question",
    color_by_correctness: bool = False,
    color_by_source: bool = False
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
        print(f"Question {q_id}: Steps={steps}, LLH={avg_llh}, Correctness={correctness}")
        if color_by_source:
            source = detect_source(str(q_id))
            color = SOURCE_COLORS.get(source, '#7f7f7f')
        elif color_by_correctness:
            # Use final correctness to determine line color
            final_correct = correctness[-1] == 'correct' if correctness else True
            color = '#2ca02c' if final_correct else '#d62728'
        else:
            color = colors[idx % len(colors)]

        plt.plot(steps, avg_llh, marker='o', label=f'Q{q_id}',
                linewidth=2, color=color, markersize=6, alpha=0.7)

        if not color_by_correctness and not color_by_source:
            # Mark incorrect points only when not already colored by correctness
            for s, l, c in zip(steps, avg_llh, correctness):
                if c == 'incorrect':
                    plt.scatter([s], [l], marker='x', color='red', s=100, zorder=5)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Log-Likelihood', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)

    if color_by_source:
        legend_text = ' | '.join(f'{src} = {clr}' for src, clr in SOURCE_COLORS.items())
        plt.figtext(0.5, 0.02, f'Color by source: blue=base, orange=qwen, green=gemma',
                    ha='center', fontsize=10, style='italic')
    elif color_by_correctness:
        plt.figtext(0.5, 0.02, 'Green = final correct, Red = final incorrect',
                    ha='center', fontsize=10, style='italic')
    else:
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
        nargs='+',
        default=[],
        help="Path(s) to one or more checkpoint summary JSONs"
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
    parser.add_argument(
        "--id_patterns",
        type=str,
        nargs='+',
        default=None,
        help="Glob patterns to filter sample types by ID "
             "(e.g., 'id_9*' matches 'id_9_base_problem_paraphrased_greedy_incorrect'). "
             "Multiple patterns can be specified."
    )
    parser.add_argument(
        "--color_by_correctness",
        action="store_true",
        help="Color lines by correctness (green=correct, red=incorrect) instead of unique colors"
    )
    parser.add_argument(
        "--color_by_source",
        action="store_true",
        help="Color lines by source detected in ID (blue=base, orange=qwen, green=gemma)"
    )

    args = parser.parse_args()

    # Load all summaries
    summaries = []
    for path in args.summary_path:
        print(f"Loading summary from {path}...")
        summary = load_summary(path)
        label = Path(path).stem  # use filename (without extension) as label
        summaries.append((label, summary))

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    show = not args.no_show

    # Filter by question IDs if specified
    if args.questions is not None:
        requested_ids = [int(q.strip()) for q in args.questions.split(',')]
        summaries = [(label, filter_summary_by_questions(s, requested_ids))
                     for label, s in summaries]
        total_types = sum(len(s['results_by_type']) for _, s in summaries)
        print(f"Filtered to question(s): {requested_ids} ({total_types} types total)")

    # Filter by id patterns if specified
    if args.id_patterns is not None:
        summaries = [(label, filter_by_id_patterns(s, args.id_patterns))
                     for label, s in summaries]
        total_types = sum(len(s['results_by_type']) for _, s in summaries)
        print(f"Filtered by id patterns {args.id_patterns} ({total_types} types total)")

    # Generate plots
    if args.plot_type in ['all', 'llh']:
        print("\nGenerating log-likelihood plot...")
        create_summary_stats(summaries)
        plot_avg_log_likelihood(
            summaries,
            output_path=f"{args.output_dir}/avg_llh.png",
            show=show,
            color_by_correctness=args.color_by_correctness,
            color_by_source=args.color_by_source
        )

    if args.plot_type in ['all', 'perplexity']:
        print("\nGenerating perplexity plot...")
        plot_perplexity(
            summaries,
            output_path=f"{args.output_dir}/avg_perplexity.png",
            show=show,
            color_by_correctness=args.color_by_correctness,
            color_by_source=args.color_by_source
        )

    if args.plot_type in ['all', 'questions']:
        print("\nOrganizing data by question...")
        questions = organize_by_question(summaries)
        print(f"Found {len(questions)} unique questions")

        create_question_summary_stats(questions)

        print("\nGenerating per-question log-likelihood plot...")
        plot_llh_by_question(
            questions,
            output_path=f"{args.output_dir}/llh_by_question.png",
            show=show,
            color_by_correctness=args.color_by_correctness,
            color_by_source=args.color_by_source
        )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
