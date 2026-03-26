#!/bin/bash

# Base model directory
base_model_dir="/projects/ai_safe/cuongdc/Qwen3-8B_low" # No slash at the end # tc-gpu004

# Array of datasets to evaluate
datasets=("olympiadbench")

# Checkpoints to exclude (step numbers only, e.g. 100 500 1000)
exclude_checkpoints=(1100 11000 12100 13200 14300)

# Create array of all model paths to evaluate (base model + all checkpoints)
model_paths=()
for checkpoint in "$base_model_dir"/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
        step="${checkpoint##*checkpoint-}"
        skip=false
        for excl in "${exclude_checkpoints[@]}"; do
            if [ "$step" = "$excl" ]; then
                skip=true
                break
            fi
        done
        $skip || model_paths+=("$checkpoint")
    fi
done

# Loop through all datasets
for data_name in "${datasets[@]}"; do
    echo "Running evaluation for dataset: $data_name"

    # Loop through all model paths
    for model_path in "${model_paths[@]}"; do
        echo "=========================================="
        echo "Evaluating model: $model_path"
        echo "=========================================="

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