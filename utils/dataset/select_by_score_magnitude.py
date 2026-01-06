"""
Select samples with highest and lowest magnitude scores from a dataset.

This utility loads a JSON dataset and corresponding numpy score file,
then selects the top N samples with highest and lowest score magnitudes.
"""

import json
import numpy as np
import argparse
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


def select_by_magnitude(
    samples: List[Dict],
    scores: np.ndarray,
    n_samples: int = 1000
) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray]:
    """
    Select samples with highest and lowest score magnitudes.

    Args:
        samples: List of sample dictionaries
        scores: Array of scores corresponding to samples
        n_samples: Number of samples to select for each category (highest/lowest)

    Returns:
        Tuple of (highest_samples, lowest_samples, highest_scores, lowest_scores)
    """
    # Calculate absolute magnitudes
    magnitudes = np.abs(scores)

    # Get indices sorted by magnitude (descending)
    sorted_indices = np.argsort(magnitudes)[::-1]

    # Select top n_samples with highest magnitude
    highest_indices = sorted_indices[:n_samples]
    highest_samples = [samples[i] for i in highest_indices]
    highest_scores = scores[highest_indices]

    # Select top n_samples with lowest magnitude
    lowest_indices = sorted_indices[-n_samples:]
    lowest_samples = [samples[i] for i in lowest_indices]
    lowest_scores = scores[lowest_indices]

    return highest_samples, lowest_samples, highest_scores, lowest_scores


def select_by_score(
    samples: List[Dict],
    scores: np.ndarray,
    n_samples: int = 1000
) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray]:
    """
    Select samples with highest and lowest scores (without taking absolute values).

    Args:
        samples: List of sample dictionaries
        scores: Array of scores corresponding to samples
        n_samples: Number of samples to select for each category (highest/lowest)

    Returns:
        Tuple of (highest_samples, lowest_samples, highest_scores, lowest_scores)
    """
    # Get indices sorted by score (descending)
    sorted_indices = np.argsort(scores)[::-1]

    # Select top n_samples with highest scores
    highest_indices = sorted_indices[:n_samples]
    highest_samples = [samples[i] for i in highest_indices]
    highest_scores = scores[highest_indices]

    # Select top n_samples with lowest scores
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
    prefix: str = "selected",
    suffix_high: str = "highest_magnitude",
    suffix_low: str = "lowest_magnitude"
) -> None:
    """
    Save selected samples and scores to files.

    Args:
        highest_samples: Samples with highest magnitude/score
        lowest_samples: Samples with lowest magnitude/score
        highest_scores: Scores for highest magnitude/score samples
        lowest_scores: Scores for lowest magnitude/score samples
        output_dir: Directory to save output files
        prefix: Prefix for output filenames
        suffix_high: Suffix for highest samples filename
        suffix_low: Suffix for lowest samples filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save highest samples
    with open(output_path / f"{prefix}_{suffix_high}.json", 'w', encoding='utf-8') as f:
        json.dump(highest_samples, f, indent=2, ensure_ascii=False)

    # Save lowest samples
    with open(output_path / f"{prefix}_{suffix_low}.json", 'w', encoding='utf-8') as f:
        json.dump(lowest_samples, f, indent=2, ensure_ascii=False)

    # Save scores
    np.save(output_path / f"{prefix}_{suffix_high}_scores.npy", highest_scores)
    np.save(output_path / f"{prefix}_{suffix_low}_scores.npy", lowest_scores)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - {prefix}_{suffix_high}.json ({len(highest_samples)} samples)")
    print(f"  - {prefix}_{suffix_low}.json ({len(lowest_samples)} samples)")
    print(f"  - {prefix}_{suffix_high}_scores.npy")
    print(f"  - {prefix}_{suffix_low}_scores.npy")
    

def print_statistics(
    highest_scores: np.ndarray,
    lowest_scores: np.ndarray,
    label_high: str = "Highest Magnitude",
    label_low: str = "Lowest Magnitude"
) -> None:
    """Print statistics about selected samples."""
    print("\n=== Statistics ===")
    print(f"\n{label_high} Samples:")
    print(f"  Mean score: {np.mean(highest_scores):.6f}")
    print(f"  Mean magnitude: {np.mean(np.abs(highest_scores)):.6f}")
    print(f"  Min score: {np.min(highest_scores):.6f}")
    print(f"  Max score: {np.max(highest_scores):.6f}")
    print(f"  Std dev: {np.std(highest_scores):.6f}")

    print(f"\n{label_low} Samples:")
    print(f"  Mean score: {np.mean(lowest_scores):.6f}")
    print(f"  Mean magnitude: {np.mean(np.abs(lowest_scores)):.6f}")
    print(f"  Min score: {np.min(lowest_scores):.6f}")
    print(f"  Max score: {np.max(lowest_scores):.6f}")
    print(f"  Std dev: {np.std(lowest_scores):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Select samples with highest and lowest magnitude scores or raw scores"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="data/openr1_random_20k_generated.json",
        help="Path to JSON dataset file"
    )
    parser.add_argument(
        "--npy_path",
        type=str,
        default="temp_str/grad_dot_products_openr1_random_20k_generated_vs_olympiadbench_7b_greedy.npy",
        help="Path to numpy score file"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of samples to select for each category (default: 1000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/selected_samples",
        help="Output directory for selected samples"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="selected",
        help="Prefix for output filenames"
    )
    parser.add_argument(
        "--use_raw_scores",
        action="store_true",
        help="Select by raw scores instead of magnitude (absolute values)"
    )

    args = parser.parse_args()

    print(f"Loading data from:")
    print(f"  JSON: {args.json_path}")
    print(f"  NPY: {args.npy_path}")

    # Load data
    samples, scores = load_data(args.json_path, args.npy_path)
    print(f"\nLoaded {len(samples)} samples with {len(scores)} scores")

    # Select samples based on mode
    if args.use_raw_scores:
        print(f"\nSelecting top {args.n_samples} samples by raw scores...")
        highest_samples, lowest_samples, highest_scores, lowest_scores = select_by_score(
            samples, scores, args.n_samples
        )
        suffix_high = "highest_score"
        suffix_low = "lowest_score"
        label_high = "Highest Score"
        label_low = "Lowest Score"
    else:
        print(f"\nSelecting top {args.n_samples} samples by magnitude...")
        highest_samples, lowest_samples, highest_scores, lowest_scores = select_by_magnitude(
            samples, scores, args.n_samples
        )
        suffix_high = "highest_magnitude"
        suffix_low = "lowest_magnitude"
        label_high = "Highest Magnitude"
        label_low = "Lowest Magnitude"

    # Print statistics
    print_statistics(highest_scores, lowest_scores, label_high, label_low)

    # Save results
    save_results(
        highest_samples,
        lowest_samples,
        highest_scores,
        lowest_scores,
        args.output_dir,
        args.prefix,
        suffix_high,
        suffix_low
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
