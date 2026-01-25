#!/usr/bin/env python3
"""
Visualize the distribution of highest and lowest likelihood answers across questions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_analyze_data(input_file: str):
    """
    Load data and extract the highest and lowest likelihood answer for each question.

    Returns:
        questions_stats: List of dictionaries with highest and lowest likelihood answers per question
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} questions")

    questions_stats = []

    for q_idx, question_item in enumerate(data):
        question = question_item['question']
        answers = question_item['answers']

        # Filter valid answers
        valid_answers = [
            ans for ans in answers
            if ans.get('log_likelihood') is not None
            and ans['log_likelihood'] != float('-inf')
            and ans['log_likelihood'] != float('inf')
        ]

        if not valid_answers:
            continue

        # Get the highest and lowest likelihood answers
        best_answer = max(valid_answers, key=lambda x: x['log_likelihood'])
        worst_answer = min(valid_answers, key=lambda x: x['log_likelihood'])

        # Store statistics
        stats = {
            'question_idx': q_idx,
            'question': question,
            'highest_ll': best_answer['log_likelihood'],
            'lowest_ll': worst_answer['log_likelihood'],
            'num_total_answers': len(valid_answers)
        }

        questions_stats.append(stats)

    print(f"Extracted likelihood answers for {len(questions_stats)} questions")

    return questions_stats

def create_visualization(questions_stats: List[Dict], output_dir: str):
    """
    Create histograms of highest and lowest likelihood distributions.
    """
    print("Creating visualizations...")

    # Extract data for plotting
    highest_lls = [q['highest_ll'] for q in questions_stats]
    lowest_lls = [q['lowest_ll'] for q in questions_stats]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Calculate quantiles
    quantiles = [25, 50, 75]
    highest_quantiles = np.percentile(highest_lls, quantiles)
    lowest_quantiles = np.percentile(lowest_lls, quantiles)

    # Histogram of highest likelihoods
    axes[0].hist(highest_lls, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Highest Log-Likelihood', fontsize=12)
    axes[0].set_ylabel('Number of Questions', fontsize=12)
    axes[0].set_title('Distribution of Highest Log-Likelihood Answers', fontsize=14, fontweight='bold')
    axes[0].axvline(highest_quantiles[0], color='orange', linestyle='--', linewidth=2, label=f'Q1 (25%): {highest_quantiles[0]:.2f}')
    axes[0].axvline(highest_quantiles[1], color='green', linestyle='--', linewidth=2, label=f'Q2 (50%): {highest_quantiles[1]:.2f}')
    axes[0].axvline(highest_quantiles[2], color='purple', linestyle='--', linewidth=2, label=f'Q3 (75%): {highest_quantiles[2]:.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of lowest likelihoods
    axes[1].hist(lowest_lls, bins=100, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Lowest Log-Likelihood', fontsize=12)
    axes[1].set_ylabel('Number of Questions', fontsize=12)
    axes[1].set_title('Distribution of Lowest Log-Likelihood Answers', fontsize=14, fontweight='bold')
    axes[1].axvline(lowest_quantiles[0], color='orange', linestyle='--', linewidth=2, label=f'Q1 (25%): {lowest_quantiles[0]:.2f}')
    axes[1].axvline(lowest_quantiles[1], color='green', linestyle='--', linewidth=2, label=f'Q2 (50%): {lowest_quantiles[1]:.2f}')
    axes[1].axvline(lowest_quantiles[2], color='purple', linestyle='--', linewidth=2, label=f'Q3 (75%): {lowest_quantiles[2]:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'likelihood_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Configuration
    input_file = "/home/cuongdc/SLM_REASONING/data/math12K_merged_answers_with_Qwen2.5_Math_7B_loglikelihood.json"
    output_dir = "/home/cuongdc/SLM_REASONING/data/visualizations"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Load and analyze data
    questions_stats = load_and_analyze_data(input_file)

    # Create visualization
    create_visualization(questions_stats, output_dir)

    print(f"\nVisualization saved to: {output_dir}/likelihood_distribution.png")
    print("Done!")
