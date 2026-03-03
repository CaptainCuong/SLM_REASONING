#!/bin/bash

# Base model directory
base_model_dir="/projects/ai_safe/cuongdc/Qwen_mix_high_self_0.75_7B" # No slash at the end
# GPU management: 8 GPUs, 1 GPU per job
NUM_GPUS=8

# Array of datasets to evaluate
datasets=("cn_math_2024" "gaokao" "grade_school_math" "kaoyan" "olympiadbench" "aime" "gpqa" "math" "minerva")

# Create array of all model paths to evaluate (base model + all checkpoints)
model_paths=("$base_model_dir")
for checkpoint in "$base_model_dir"/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
        model_paths+=("$checkpoint")
    fi
done

declare -a gpu_pids  # PID of the job running on each GPU slot

# Initialize all GPU slots as free
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_pids[$i]=""
done

# Find a free GPU slot; blocks until one is available
get_free_gpu() {
    while true; do
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            pid=${gpu_pids[$gpu_id]}
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$gpu_id"
                return
            fi
        done
        sleep 5
    done
}

# Loop through all model paths and datasets, dispatching 1 job per GPU
for model_path in "${model_paths[@]}"; do
    echo "=========================================="
    echo "Scheduling jobs for model: $model_path"
    echo "=========================================="

    for data_name in "${datasets[@]}"; do
        gpu_id=$(get_free_gpu)

        echo "GPU $gpu_id | Model: $(basename "$model_path") | Dataset: $data_name"

        CUDA_VISIBLE_DEVICES="$gpu_id" \
        python eval.py \
        --model_name_or_path "$model_path" \
        --data_name "$data_name" \
        --prompt_type "qwen-instruct" \
        --temperature 0.6 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 8 \
        --k 1 \
        --split "test" \
        --max_tokens 32768 \
        --seed 0 \
        --top_p 0.9 \
        --surround_with_messages \
        > "./logs/Qwen_mix_high_self_0.75_7B_$(basename "$model_path")_${data_name}.log" 2>&1 &

        gpu_pids[$gpu_id]=$!
    done
done

# Wait for all remaining background jobs
echo "=========================================="
echo "Waiting for all jobs to complete..."
echo "=========================================="
wait

echo "All evaluations completed!"
