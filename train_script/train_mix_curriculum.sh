#!/bin/bash
# Multi-round curriculum training: gradually reduce RATIO_A (high-likelihood ratio) over rounds
# Round 1: 0.75 high / 0.25 low  (base model → round1 checkpoint)
# Round 2: 0.50 high / 0.50 low  (round1 checkpoint → round2 checkpoint)
# Round 3: 0.25 high / 0.75 low  (round2 checkpoint → round3 checkpoint)

RATIOS=(1 0.9 0.8 0.7)
NUM_ROUNDS=${#RATIOS[@]}

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
PREV_MODEL=$BASE_MODEL

for i in "${!RATIOS[@]}"; do
    ROUND=$((i + 1))
    RATIO_A=${RATIOS[$i]}

    echo "========================================"
    echo "Round $ROUND / $NUM_ROUNDS  |  RATIO_A=$RATIO_A"
    echo "Starting from: $PREV_MODEL"
    echo "========================================"

    # Mix data with current ratio
    python data/data_processing_utils/mix_training_data.py \
        --file_a data/math12K_highest_Qwen2_5_3B_Instruct_llh.json \
        --file_b data/math12K_lowest_Qwen2_5_3B_Instruct_llh.json \
        --ratio_a $RATIO_A \
        --output data/math12K_high_low_mixed.json

    OUTPUT_DIR=/helios-storage/helios4-data/cuong/model/Qwen_curriculum_round${ROUND}_high_ratio${RATIO_A}_3B

    CUDA_VISIBLE_DEVICES='0,1' \
    llamafactory-cli train \
        --model_name_or_path $PREV_MODEL \
        --stage sft \
        --do_train true \
        --finetuning_type full \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --flash_attn fa2 \
        --gradient_checkpointing true \
        --dataset math12K_high_low_mixed \
        --cutoff_len 8192 \
        --overwrite_cache true \
        --preprocessing_num_workers 64 \
        --template qwen \
        --output_dir $OUTPUT_DIR \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 1100 \
        --plot_loss true \
        --overwrite_output_dir true \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-6 \
        --num_train_epochs 2 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.0 \
        --bf16 true \
        --ddp_timeout 180000000 \
        --save_only_model true

    if [ $? -ne 0 ]; then
        echo "Training failed at round $ROUND. Exiting."
        exit 1
    fi

    PREV_MODEL=$OUTPUT_DIR
    echo "Round $ROUND done. Model saved to $OUTPUT_DIR"
done

echo "All $NUM_ROUNDS rounds completed."
