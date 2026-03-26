#!/bin/bash

# List of phase 1 transition epochs to evaluate
phase1_epochs=(3 4 5 6 7)

# Base directory pattern for phase 2 models
base_dir_prefix="/projects/ai_safe/cuongdc/adaptive_3B_low_phase2_from_phase1_epoch_"

# Array of datasets to evaluate
datasets=("math12k" "cn_math_2024" "gaokao" "grade_school_math" "kaoyan" "olympiadbench" "aime" "amc" "gpqa" "math" "minerva")

for phase1_epoch in "${phase1_epochs[@]}"; do
    base_model_dir="${base_dir_prefix}${phase1_epoch}"

    if [ ! -d "$base_model_dir" ]; then
        echo "Skipping phase1 epoch ${phase1_epoch}: directory not found ($base_model_dir)"
        continue
    fi

    echo "=========================================="
    echo "Phase 1 epoch ${phase1_epoch}: $base_model_dir"
    echo "=========================================="

    # Collect all epoch dirs (and their checkpoints) under this phase 2 dir
    model_paths=()
    for epoch_dir in "$base_model_dir"/epoch*; do
        if [ -d "$epoch_dir" ]; then
            for checkpoint in "$epoch_dir"/checkpoint-*; do
                if [ -d "$checkpoint" ]; then
                    model_paths+=("$checkpoint")
                fi
            done
        fi
    done

    if [ ${#model_paths[@]} -eq 0 ]; then
        echo "No checkpoints found under $base_model_dir, skipping."
        continue
    fi

    echo "model_paths: ${model_paths[@]}"

    for model_path in "${model_paths[@]}"; do
        echo "=========================================="
        echo "Evaluating model: $model_path"
        echo "=========================================="

        for data_name in "${datasets[@]}"; do
            echo "Running evaluation for dataset: $data_name"

            CUDA_VISIBLE_DEVICES='0,1' \
            python eval.py \
            --model_name_or_path "$model_path" \
            --data_name "$data_name" \
            --prompt_type "qwen-instruct" \
            --temperature 0.6 \
            --start_idx 0 \
            --end_idx -1 \
            --n_sampling 8 \
            --k 8 \
            --split "test" \
            --max_tokens 32768 \
            --seed 0 \
            --top_p 0.9 \
            --surround_with_messages

            echo "Completed evaluation for $data_name on $model_path"
            echo "------------------------------------------"
        done

        echo "Completed all evaluations for model: $model_path"
        echo "=========================================="
        echo ""
    done
done

echo "All phase 2 evaluations completed!"
