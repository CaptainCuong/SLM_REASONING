"""
Utility functions for analyzing and visualizing response metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from metrics import ResponseMetrics
import json


def analyze_token_distribution(metrics: ResponseMetrics) -> Dict:
    """
    Analyze the distribution of token probabilities.

    Args:
        metrics: ResponseMetrics object

    Returns:
        Dictionary with distribution statistics
    """
    probs = np.array(metrics.token_probs)

    return {
        'mean_prob': float(np.mean(probs)),
        'std_prob': float(np.std(probs)),
        'min_prob': float(np.min(probs)),
        'max_prob': float(np.max(probs)),
        'median_prob': float(np.median(probs)),
        'q25_prob': float(np.percentile(probs, 25)),
        'q75_prob': float(np.percentile(probs, 75)),
    }


def identify_low_confidence_tokens(
    metrics: ResponseMetrics,
    threshold: float = 0.1
) -> List[Dict]:
    """
    Identify tokens with low probability (low model confidence).

    Args:
        metrics: ResponseMetrics object
        threshold: Probability threshold below which tokens are considered low confidence

    Returns:
        List of dictionaries with low confidence token information
    """
    low_conf_tokens = []

    for i, (token, prob, log_prob) in enumerate(zip(
        metrics.tokens,
        metrics.token_probs,
        metrics.token_log_probs
    )):
        if prob < threshold:
            low_conf_tokens.append({
                'position': i,
                'token': token,
                'probability': prob,
                'log_probability': log_prob
            })

    return low_conf_tokens


def identify_high_entropy_tokens(
    metrics: ResponseMetrics,
    percentile: float = 90
) -> List[Dict]:
    """
    Identify tokens with high entropy (high model uncertainty).

    Args:
        metrics: ResponseMetrics object
        percentile: Percentile threshold for high entropy

    Returns:
        List of dictionaries with high entropy token information
    """
    threshold = np.percentile(metrics.token_entropies, percentile)
    high_entropy_tokens = []

    for i, (token, entropy, prob) in enumerate(zip(
        metrics.tokens,
        metrics.token_entropies,
        metrics.token_probs
    )):
        if entropy >= threshold:
            high_entropy_tokens.append({
                'position': i,
                'token': token,
                'entropy': entropy,
                'probability': prob
            })

    return high_entropy_tokens


def plot_token_metrics(
    metrics: ResponseMetrics,
    save_path: Optional[str] = None,
    show_tokens: bool = True,
    max_tokens_display: int = 50
):
    """
    Plot token-level metrics.

    Args:
        metrics: ResponseMetrics object
        save_path: Path to save the plot (if None, display only)
        show_tokens: Whether to show token labels on x-axis
        max_tokens_display: Maximum number of tokens to display
    """
    n_tokens = min(len(metrics.tokens), max_tokens_display)
    positions = range(n_tokens)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Token Probabilities
    axes[0].bar(positions, metrics.token_probs[:n_tokens], color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Probability', fontsize=12)
    axes[0].set_title('Token Probabilities', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Plot 2: Log Probabilities
    axes[1].bar(positions, metrics.token_log_probs[:n_tokens], color='coral', alpha=0.7)
    axes[1].set_ylabel('Log Probability', fontsize=12)
    axes[1].set_title('Token Log Probabilities', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Entropy
    axes[2].bar(positions, metrics.token_entropies[:n_tokens], color='seagreen', alpha=0.7)
    axes[2].set_ylabel('Entropy', fontsize=12)
    axes[2].set_xlabel('Token Position', fontsize=12)
    axes[2].set_title('Token Entropy', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Add token labels if requested
    if show_tokens and metrics.tokens and n_tokens <= 30:
        token_labels = [t.replace('\n', '\\n')[:15] for t in metrics.tokens[:n_tokens]]
        axes[2].set_xticks(positions)
        axes[2].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
    else:
        axes[2].set_xlabel('Token Position', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def compare_metrics_table(
    metrics_list: List[ResponseMetrics],
    labels: Optional[List[str]] = None
) -> str:
    """
    Create a comparison table for multiple metrics.

    Args:
        metrics_list: List of ResponseMetrics objects
        labels: Optional labels for each metrics object

    Returns:
        Formatted table string
    """
    if labels is None:
        labels = [f"Response {i+1}" for i in range(len(metrics_list))]

    # Header
    table = f"{'Label':<20} {'Perplexity':<12} {'Log-Lik':<12} {'Avg Prob':<12} {'Entropy':<12} {'Tokens':<8}\n"
    table += "=" * 88 + "\n"

    # Rows
    for label, metrics in zip(labels, metrics_list):
        table += (f"{label:<20} {metrics.perplexity:<12.4f} {metrics.log_likelihood:<12.4f} "
                  f"{metrics.average_token_prob:<12.4f} {metrics.entropy:<12.4f} {metrics.token_count:<8}\n")

    return table


def generate_metrics_report(
    metrics: ResponseMetrics,
    instruction: str,
    response: str,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive metrics report.

    Args:
        metrics: ResponseMetrics object
        instruction: The instruction text
        response: The response text
        save_path: Optional path to save the report

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("RESPONSE METRICS REPORT")
    report.append("=" * 80)
    report.append("")

    # Input/Output
    report.append("INSTRUCTION:")
    report.append(f"  {instruction}")
    report.append("")
    report.append("RESPONSE:")
    report.append(f"  {response}")
    report.append("")

    # Overall Metrics
    report.append("-" * 80)
    report.append("OVERALL METRICS:")
    report.append("-" * 80)
    report.append(f"  Perplexity:              {metrics.perplexity:.4f}")
    report.append(f"  Log-Likelihood:          {metrics.log_likelihood:.4f}")
    report.append(f"  Average Token Prob:      {metrics.average_token_prob:.4f}")
    report.append(f"  Entropy:                 {metrics.entropy:.4f}")
    report.append(f"  Token Count:             {metrics.token_count}")
    report.append("")

    # Distribution Analysis
    report.append("-" * 80)
    report.append("PROBABILITY DISTRIBUTION:")
    report.append("-" * 80)
    dist_stats = analyze_token_distribution(metrics)
    report.append(f"  Mean:                    {dist_stats['mean_prob']:.4f}")
    report.append(f"  Std Dev:                 {dist_stats['std_prob']:.4f}")
    report.append(f"  Min:                     {dist_stats['min_prob']:.4f}")
    report.append(f"  25th Percentile:         {dist_stats['q25_prob']:.4f}")
    report.append(f"  Median:                  {dist_stats['median_prob']:.4f}")
    report.append(f"  75th Percentile:         {dist_stats['q75_prob']:.4f}")
    report.append(f"  Max:                     {dist_stats['max_prob']:.4f}")
    report.append("")

    # Low Confidence Tokens
    report.append("-" * 80)
    report.append("LOW CONFIDENCE TOKENS (prob < 0.1):")
    report.append("-" * 80)
    low_conf = identify_low_confidence_tokens(metrics, threshold=0.1)
    if low_conf:
        for token_info in low_conf[:10]:  # Show top 10
            report.append(f"  Position {token_info['position']}: '{token_info['token']}' "
                          f"(prob={token_info['probability']:.4f})")
        if len(low_conf) > 10:
            report.append(f"  ... and {len(low_conf) - 10} more")
    else:
        report.append("  None found")
    report.append("")

    # High Entropy Tokens
    report.append("-" * 80)
    report.append("HIGH ENTROPY TOKENS (top 10%):")
    report.append("-" * 80)
    high_entropy = identify_high_entropy_tokens(metrics, percentile=90)
    if high_entropy:
        for token_info in high_entropy[:10]:  # Show top 10
            report.append(f"  Position {token_info['position']}: '{token_info['token']}' "
                          f"(entropy={token_info['entropy']:.4f}, prob={token_info['probability']:.4f})")
        if len(high_entropy) > 10:
            report.append(f"  ... and {len(high_entropy) - 10} more")
    else:
        report.append("  None found")
    report.append("")

    report.append("=" * 80)

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {save_path}")

    return report_text


def export_metrics_csv(
    metrics_list: List[ResponseMetrics],
    output_path: str,
    labels: Optional[List[str]] = None
):
    """
    Export metrics to CSV file.

    Args:
        metrics_list: List of ResponseMetrics objects
        output_path: Path to save CSV file
        labels: Optional labels for each metrics object
    """
    import csv

    if labels is None:
        labels = [f"response_{i}" for i in range(len(metrics_list))]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'label', 'perplexity', 'log_likelihood', 'average_token_prob',
            'entropy', 'token_count'
        ])

        # Data rows
        for label, metrics in zip(labels, metrics_list):
            writer.writerow([
                label,
                metrics.perplexity,
                metrics.log_likelihood,
                metrics.average_token_prob,
                metrics.entropy,
                metrics.token_count
            ])

    print(f"Metrics exported to: {output_path}")
