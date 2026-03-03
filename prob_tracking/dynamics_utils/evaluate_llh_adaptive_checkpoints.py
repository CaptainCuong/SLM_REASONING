#!/usr/bin/env python3
"""
Evaluate all checkpoints across adaptive training epochs.

Epochs are stored in separate folders (epoch_1, epoch_2, ...). Checkpoint steps
within each epoch are local (e.g. checkpoint-1100), but are converted to
cumulative global steps for the summary (e.g. epoch_2/checkpoint-1100 → step 5500
if epoch_1 ended at step 4400).
"""

import gc
import json
import os
import argparse
import re
from pathlib import Path
from typing import List, Tuple
import torch
from tqdm import tqdm

from calculate_llh import (
    load_model_and_tokenizer,
    load_pool_data,
    calculate_metrics_for_pool,
)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_epoch_folders(model_folder: str) -> List[Tuple[int, Path]]:
    """Return (epoch_num, path) pairs sorted by epoch number."""
    model_path = Path(model_folder)
    epochs = []
    for item in model_path.iterdir():
        if item.is_dir():
            match = re.fullmatch(r'epoch[_-]?(\d+)', item.name)
            if match:
                epochs.append((int(match.group(1)), item))
    epochs.sort(key=lambda x: x[0])
    
    # Exclude the last epoch
    if epochs:
        print(f"Found {len(epochs)} epoch folders. Excluding the last one: {epochs[-1][1].name}")
        epochs = epochs[:-1]
    else:
        print("No epoch folders found.")
    return epochs


def get_epoch_total_steps(epoch_path: Path) -> int:
    """Read trainer_state.json and return the epoch's final global_step."""
    state_file = epoch_path / 'trainer_state.json'
    if not state_file.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {epoch_path}")
    with open(state_file) as f:
        state = json.load(f)
    return int(state['global_step'])


def find_checkpoints_in_epoch(epoch_path: Path) -> List[Tuple[int, Path]]:
    """Return (local_step, path) pairs for all checkpoints in an epoch folder."""
    checkpoints = []
    for item in epoch_path.iterdir():
        if item.is_dir():
            match = re.search(r'checkpoint-(\d+)', item.name)
            if match:
                checkpoints.append((int(match.group(1)), item))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def collect_all_checkpoints(model_folder: str) -> List[Tuple[int, int, int, Path]]:
    """
    Collect all checkpoints across epochs with cumulative global steps.

    Returns list of (global_step, epoch_num, local_step, path) sorted by global_step.
    """
    epoch_folders = find_epoch_folders(model_folder)
    if not epoch_folders:
        raise ValueError(f"No epoch_* folders found in {model_folder}")

    all_checkpoints = []
    cumulative_offset = 0

    for epoch_num, epoch_path in epoch_folders:
        checkpoints = find_checkpoints_in_epoch(epoch_path)
        epoch_total = get_epoch_total_steps(epoch_path)

        for local_step, ckpt_path in checkpoints:
            global_step = cumulative_offset + local_step
            all_checkpoints.append((global_step, epoch_num, local_step, ckpt_path))

        cumulative_offset += epoch_total

    all_checkpoints.sort(key=lambda x: x[0])
    return all_checkpoints


# ---------------------------------------------------------------------------
# Metrics aggregation (same as original)
# ---------------------------------------------------------------------------

def aggregate_metrics_by_type(results):
    type_metrics = {}
    for result in results:
        sample_type = result['type']
        if sample_type not in type_metrics:
            type_metrics[sample_type] = {'perplexity': [], 'log_likelihood': [], 'token_count': []}
        m = result['metrics']
        type_metrics[sample_type]['perplexity'].append(m['perplexity'])
        type_metrics[sample_type]['log_likelihood'].append(m['log_likelihood'])
        type_metrics[sample_type]['token_count'].append(m['token_count'])
    return type_metrics


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_adaptive_checkpoints(
    model_folder: str,
    pool_path: str,
    output_path: str = None,
    device: str = None,
    base_model_path: str = None,
    include_instruction: bool = False,
):
    """
    Evaluate all checkpoints across adaptive training epochs.

    Checkpoint steps are made cumulative across epochs so the timeline is
    continuous (e.g. epoch_2/checkpoint-1100 appears as step 5500 if epoch_1
    ended at step 4400).
    """
    # Load pool data once
    print(f"\nLoading pool data from {pool_path}...")
    pool_data = load_pool_data(pool_path)
    print(f"Loaded {len(pool_data)} samples")

    mode_str = "whole sequence" if include_instruction else "output tokens only"
    all_checkpoint_results = []

    # ------------------------------------------------------------------
    # Evaluate base model (step 0)
    # ------------------------------------------------------------------
    base_path = base_model_path if base_model_path else model_folder
    print(f"\n{'='*80}")
    print(f"Processing BASE MODEL (step 0) — mode: {mode_str}")
    print(f"Path: {base_path}")
    print(f"{'='*80}")

    model, tokenizer, device_used = load_model_and_tokenizer(base_path, device)
    results = calculate_metrics_for_pool(model, tokenizer, pool_data, device_used, include_instruction)
    type_metrics = aggregate_metrics_by_type(results)

    all_checkpoint_results.append({
        'checkpoint': 'base_model',
        'global_step': 0,
        'epoch': 0,
        'local_step': 0,
        'metrics_by_type': type_metrics,
    })

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("✓ Completed base model evaluation")

    # ------------------------------------------------------------------
    # Discover and report all checkpoints
    # ------------------------------------------------------------------
    print(f"\nScanning for checkpoints under {model_folder}...")
    all_checkpoints = collect_all_checkpoints(model_folder)

    if not all_checkpoints:
        print("No checkpoints found. Only base model results will be saved.")
    else:
        print(f"Found {len(all_checkpoints)} checkpoints:")
        for global_step, epoch_num, local_step, path in all_checkpoints:
            print(f"  epoch_{epoch_num}/checkpoint-{local_step}  →  global step {global_step}")

    # ------------------------------------------------------------------
    # Process each checkpoint
    # ------------------------------------------------------------------
    for global_step, epoch_num, local_step, ckpt_path in tqdm(all_checkpoints, desc="Processing checkpoints"):
        label = f"epoch_{epoch_num}/checkpoint-{local_step} (global step {global_step})"
        print(f"\n{'='*80}")
        print(f"Processing {label}")
        print(f"{'='*80}")

        model, tokenizer, device_used = load_model_and_tokenizer(str(ckpt_path), device)
        results = calculate_metrics_for_pool(model, tokenizer, pool_data, device_used, include_instruction)
        type_metrics = aggregate_metrics_by_type(results)

        all_checkpoint_results.append({
            'checkpoint': f'epoch_{epoch_num}/checkpoint-{local_step}',
            'global_step': global_step,
            'epoch': epoch_num,
            'local_step': local_step,
            'metrics_by_type': type_metrics,
        })

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"✓ Completed {label}")

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Creating summary...")

    all_steps = [r['global_step'] for r in all_checkpoint_results]
    all_ckpt_names = [r['checkpoint'] for r in all_checkpoint_results]

    all_types = set()
    for r in all_checkpoint_results:
        all_types.update(r['metrics_by_type'].keys())

    results_by_type = {}
    for sample_type in all_types:
        results_by_type[sample_type] = {
            'perplexity': [],
            'log_likelihood': [],
            'avg_perplexity': [],
            'avg_log_likelihood': [],
        }
        for r in all_checkpoint_results:
            if sample_type in r['metrics_by_type']:
                td = r['metrics_by_type'][sample_type]
                results_by_type[sample_type]['perplexity'].append(td['perplexity'])
                results_by_type[sample_type]['log_likelihood'].append(td['log_likelihood'])
                results_by_type[sample_type]['avg_perplexity'].append(
                    sum(td['perplexity']) / len(td['perplexity'])
                )
                results_by_type[sample_type]['avg_log_likelihood'].append(
                    sum(td['log_likelihood']) / len(td['log_likelihood'])
                )
            else:
                results_by_type[sample_type]['perplexity'].append(None)
                results_by_type[sample_type]['log_likelihood'].append(None)
                results_by_type[sample_type]['avg_perplexity'].append(None)
                results_by_type[sample_type]['avg_log_likelihood'].append(None)

    summary = {
        'model_folder': model_folder,
        'num_checkpoints': len(all_checkpoint_results),
        'checkpoints': all_ckpt_names,
        'steps': all_steps,
        'results_by_type': results_by_type,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if output_path is None:
        pool_name = Path(pool_path).stem
        model_name = Path(model_folder).name
        output_path = f"./prob_tracking/results/{model_name}_{pool_name}_adaptive_summary.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ Summary saved to {output_path}")

    # ------------------------------------------------------------------
    # Print summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"{'='*80}")
    for sample_type in sorted(all_types):
        print(f"\nType: {sample_type}")
        valid_ppl = [x for x in results_by_type[sample_type]['avg_perplexity'] if x is not None]
        valid_llh = [x for x in results_by_type[sample_type]['avg_log_likelihood'] if x is not None]
        if valid_ppl:
            print(f"  Perplexity:      {valid_ppl[0]:.4f} → {valid_ppl[-1]:.4f}")
            print(f"  Log-Likelihood:  {valid_llh[0]:.4f} → {valid_llh[-1]:.4f}")
    print(f"\n{'='*80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints across adaptive training epochs with cumulative global steps"
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Root folder containing epoch_* subdirectories (e.g. /helios-storage/.../adaptive_3B_low)"
    )
    parser.add_argument(
        "--pool_path",
        type=str,
        default="./prob_tracking/data/test_high.json",
        help="Path to pool JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON (default: ./prob_tracking/results/<model>_<pool>_adaptive_summary.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Path or HF name of the base model before training"
    )
    parser.add_argument(
        "--include_instruction",
        action="store_true",
        help="Calculate log-likelihood on the whole sequence (instruction + output). "
             "Default: output tokens only."
    )

    args = parser.parse_args()

    evaluate_adaptive_checkpoints(
        model_folder=args.model_folder,
        pool_path=args.pool_path,
        output_path=args.output_path,
        device=args.device,
        base_model_path=args.base_model_path,
        include_instruction=args.include_instruction,
    )


if __name__ == "__main__":
    main()
