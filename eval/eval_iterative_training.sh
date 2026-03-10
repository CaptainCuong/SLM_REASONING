#!/bin/bash

# Base model directory
base_model_dir="/projects/ai_safe/cuongdc/adaptive_high2low_3B" # No slash at the end

# Array of datasets to evaluate
datasets=("math12k" "cn_math_2024" "gaokao" "grade_school_math" "kaoyan" "olympiadbench" "aime" "amc" "gpqa" "math" "minerva")
# Create array of all model paths to evaluate (base model + all checkpoints)

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

echo "model_paths: ${model_paths[@]}"

# Loop through all model paths
for model_path in "${model_paths[@]}"; do
    echo "=========================================="
    echo "Evaluating model: $model_path"
    echo "=========================================="

    # Loop through all datasets
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

echo "All evaluations completed!"