"""
Select 2500 samples with highest absolute score and 2500 samples with lowest absolute score.
Mimics the pattern from select_by_score_magnitude.py
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_data(json_path: str, npy_path: str) -> Tuple[List[Dict], np.ndarray]:
    """
    Load JSON dataset and numpy scores.

    Args:
        json_path: Path to JSON file containing dataset samples
        npy_path: Path to numpy file containing scores

    Returns:
        Tuple of (samples list, scores array)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    scores = np.load(npy_path)

    # Validate lengths match
    if len(samples) != len(scores):
        raise ValueError(
            f"Mismatch: {len(samples)} samples but {len(scores)} scores"
        )

    return samples, scores


def select_by_absolute_score(
    samples: List[Dict],
    scores: np.ndarray,
    n_samples: int = 2500
) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray]:
    """
    Select samples with highest and lowest absolute scores.

    Args:
        samples: List of sample dictionaries
        scores: Array of scores corresponding to samples
        n_samples: Number of samples to select for each category (default: 2500)

    Returns:
        Tuple of (highest_abs_samples, lowest_abs_samples, highest_abs_scores, lowest_abs_scores)
    """
    # Calculate absolute magnitudes
    magnitudes = np.abs(scores)

    # Get indices sorted by magnitude (descending)
    sorted_indices = np.argsort(magnitudes)[::-1]

    # Select top n_samples with highest absolute score
    highest_indices = sorted_indices[:n_samples]
    highest_samples = [samples[i] for i in highest_indices]
    highest_scores = scores[highest_indices]

    # Select top n_samples with lowest absolute score
    lowest_indices = sorted_indices[-n_samples:]
    lowest_samples = [samples[i] for i in lowest_indices]
    lowest_scores = scores[lowest_indices]

    return highest_samples, lowest_samples, highest_scores, lowest_scores


def save_results(
    highest_samples: List[Dict],
    lowest_samples: List[Dict],
    highest_scores: np.ndarray,
    lowest_scores: np.ndarray,
    output_dir: str,
    prefix: str = "selected"
) -> None:
    """
    Save selected samples and scores to files.
    Combines highest and lowest samples into a single JSON file with 5000 samples.

    Args:
        highest_samples: Samples with highest absolute score
        lowest_samples: Samples with lowest absolute score
        highest_scores: Scores for highest absolute score samples
        lowest_scores: Scores for lowest absolute score samples
        output_dir: Directory to save output files
        prefix: Prefix for output filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Combine highest and lowest samples into one list
    combined_samples = highest_samples + lowest_samples
    combined_scores = np.concatenate([highest_scores, lowest_scores])

    # Save combined samples to single JSON file
    with open(output_path / f"{prefix}_highest_abs_lowest_abs_combined.json", 'w', encoding='utf-8') as f:
        json.dump(combined_samples, f, indent=2, ensure_ascii=False)

    # Save combined scores
    np.save(output_path / f"{prefix}_highest_abs_lowest_abs_combined_scores.npy", combined_scores)
    print(f"\nResults saved to {output_dir}/")
    print(f"  - {prefix}_highest_abs_lowest_abs_combined.json ({len(combined_samples)} samples)")
    print(f"    - First {len(highest_samples)} samples: highest absolute scores")
    print(f"    - Last {len(lowest_samples)} samples: lowest absolute scores")
    print(f"  - {prefix}_highest_abs_lowest_abs_combined_scores.npy")


def print_statistics(
    highest_scores: np.ndarray,
    lowest_scores: np.ndarray
) -> None:
    """Print statistics about selected samples."""
    print("\n=== Statistics ===")
    print(f"\nHighest Absolute Score Samples:")
    print(f"  Mean score: {np.mean(highest_scores):.6f}")
    print(f"  Mean absolute score: {np.mean(np.abs(highest_scores)):.6f}")
    print(f"  Min score: {np.min(highest_scores):.6f}")
    print(f"  Max score: {np.max(highest_scores):.6f}")
    print(f"  Std dev: {np.std(highest_scores):.6f}")

    print(f"\nLowest Absolute Score Samples:")
    print(f"  Mean score: {np.mean(lowest_scores):.6f}")
    print(f"  Mean absolute score: {np.mean(np.abs(lowest_scores)):.6f}")
    print(f"  Min score: {np.min(lowest_scores):.6f}")
    print(f"  Max score: {np.max(lowest_scores):.6f}")
    print(f"  Std dev: {np.std(lowest_scores):.6f}")


def main():
    # Configuration
    json_path = "data/openr1_random_20k_generated.json"
    npy_path = "temp_str/grad_dot_products_openr1_random_20k_generated_vs_olympiadbench_7b_greedy.npy"
    n_samples = 2500
    output_dir = "data/selected_samples"
    prefix = "selected"

    print(f"Loading data from:")
    print(f"  JSON: {json_path}")
    print(f"  NPY: {npy_path}")

    # Load data
    samples, scores = load_data(json_path, npy_path)
    print(f"\nLoaded {len(samples)} samples with {len(scores)} scores")

    # Select samples by absolute score
    print(f"\nSelecting {n_samples} samples with highest absolute score and {n_samples} with lowest absolute score...")
    highest_samples, lowest_samples, highest_scores, lowest_scores = select_by_absolute_score(
        samples, scores, n_samples
    )

    # Print statistics
    print_statistics(highest_scores, lowest_scores)

    # Save results
    save_results(
        highest_samples,
        lowest_samples,
        highest_scores,
        lowest_scores,
        output_dir,
        prefix
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
