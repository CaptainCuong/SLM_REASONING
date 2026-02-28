#!/bin/bash

# Base model directory
base_model_dir="/projects/ai_safe/cuongdc/adaptive_7B_low/epoch_2" # No slash at the end

# Array of datasets to evaluate
datasets=("math12k" "cn_math_2024" "gaokao" "grade_school_math" "kaoyan" "olympiadbench" "aime" "amc" "gpqa" "math" "minerva")
datasets=("cn_math_2024" "gaokao" "grade_school_math" "kaoyan" "olympiadbench" "aime" "gpqa" "math" "minerva")
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
        --temperature 0.7 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 1 \
        --k 1 \
        --split "test" \
        --max_tokens 32768 \
        --seed 0 \
        --top_p 0.9 \
        --surround_with_messages \
        --output_dir "./outputs/adaptive_7B_low" \

        echo "Completed evaluation for $data_name on $model_path"
        echo "------------------------------------------"
    done

    echo "Completed all evaluations for model: $model_path"
    echo "=========================================="
    echo ""
done

echo "All evaluations completed!"