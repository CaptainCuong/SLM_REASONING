#!/usr/bin/env python3
"""
Calculate similarity (dot product or cosine) between gradient projections.
Uses mean mode: computes mean of folder2, then similarity with each file in folder1.
"""

import sys
import argparse
sys.path.append('influence/utils')

from calculate_grad_similarity import calculate_dots_with_mean, calculate_cosine_with_mean, print_mean_dot_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate gradient similarity (dot product or cosine) with mean'
    )
    parser.add_argument(
        '--folder1',
        type=str,
        default="/projects/ai_safe/cuongdc/grad_proj_openr1_random_20k_generated_idx0_end",
        help='Path to folder1 (training data)'
    )
    parser.add_argument(
        '--folder2',
        type=str,
        default="/projects/ai_safe/cuongdc/grad_proj_olympiadbench_7b_greedy_idx0_end",
        help='Path to folder2 (test data - will compute mean)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="temp_str/openr1_random_20k_generated_vs_olympiadbench_7b_greedy.npy",
        help='Output .npy file path'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='dot',
        choices=['dot', 'cosine'],
        help='Similarity metric: dot (dot product) or cosine (cosine similarity)'
    )

    args = parser.parse_args()

    metric_name = "Cosine Similarity" if args.metric == 'cosine' else "Dot Product"

    # Add metric prefix to output filename if not already present
    import os
    output_dir = os.path.dirname(args.output)
    output_basename = os.path.basename(args.output)

    if args.metric == 'cosine':
        if not output_basename.startswith('cosine_'):
            output_basename = f"cosine_{output_basename}"
    else:  # dot
        if not output_basename.startswith('grad_dot_products_'):
            output_basename = f"grad_dot_products_{output_basename}"

    args.output = os.path.join(output_dir, output_basename)

    print("="*80)
    print(f"Gradient {metric_name} Calculation (Mean Mode)")
    print("="*80)
    print(f"\nDataset 1: {args.folder1}")
    print(f"Dataset 2: {args.folder2} (will compute mean)")
    print(f"\nMetric: {metric_name}")
    print(f"Mode: Calculate {args.metric} of each individual example with mean of folder2")
    print(f"Note: Handles batch dimension - each file contains multiple examples")
    print(f"Note: Does NOT flatten - maintains gradient structure")
    print(f"Note: Mean of folder2 has first dimension = 1")
    print("="*80)

    # Calculate similarity with mean based on metric
    if args.metric == 'cosine':
        results = calculate_cosine_with_mean(
            args.folder1,
            args.folder2,
            save_path=args.output
        )
    else:
        results = calculate_dots_with_mean(
            args.folder1,
            args.folder2,
            save_path=args.output
        )

    # Print statistics
    print_mean_dot_results(results)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output}")
    print("\nInterpretation:")
    print("  - Each projection file contains multiple examples (batch_size)")
    print(f"  - {metric_name} computed for each individual example vs mean of folder2")
    if args.metric == 'cosine':
        print("  - Cosine similarity ranges from -1 (opposite) to 1 (aligned)")
        print("  - Higher values mean stronger directional alignment with mean test gradient")
    else:
        print("  - Higher dot product means stronger alignment with mean test gradient")
    print("  - This indicates training examples similar to the test set")
    print("  - Can help identify influential training examples")
    print("="*80)
