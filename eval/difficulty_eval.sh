#!/bin/bash

# Array of different n_sampling values to test
N_SAMPLING_VALUES=(5 10 15 20)

# Base model path - modify as needed
MODEL_PATH="/helios-storage/helios3-data/cuong/model/Qwen_Math_high/checkpoint-555/"

# Loop through each n_sampling value
for n_sampling in "${N_SAMPLING_VALUES[@]}"
do
    echo "=========================================="
    echo "Running evaluation with n_sampling=${n_sampling}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES='0,1,2,3' \
    python eval.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_name "cn_math_2024" \
    --prompt_type "qwen-instruct" \
    --temperature 0.7 \
    --start_idx 0 \
    --end_idx -1 \
    --n_sampling ${n_sampling} \
    --k 1 \
    --split "test" \
    --max_tokens 32768 \
    --seed 0 \
    --top_p 0.9 \
    --surround_with_messages

    echo ""
    echo "Completed evaluation with n_sampling=${n_sampling}"
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
