#!/bin/bash

# Base model directory
# base_model_dir="/projects/ai_safe/cuongdc/Qwen_Math_high"
base_model_dir="Qwen/Qwen2.5-Math-7B"

# Array of datasets to evaluate
# datasets=("cn_math_2024" "gaokao" "grade_school_math" "kaoyan" "olympiadbench" "aime" "amc" "gpqa" "math" "minerva")
datasets=("test")

# Create array of all model paths to evaluate (base model + all checkpoints)
model_paths=("$base_model_dir")
for checkpoint in "$base_model_dir"/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
        model_paths+=("$checkpoint")
    fi
done

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
        --temperature 0.0 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 1 \
        --k 1 \
        --split "test" \
        --max_tokens 32768 \
        --seed 0 \
        --top_p 1 \
        --surround_with_messages

        echo "Completed evaluation for $data_name on $model_path"
        echo "------------------------------------------"
    done

    echo "Completed all evaluations for model: $model_path"
    echo "=========================================="
    echo ""
done

echo "All evaluations completed!"
