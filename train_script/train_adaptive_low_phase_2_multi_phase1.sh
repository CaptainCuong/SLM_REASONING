GPUS=0,1
MODEL_SIZE=3B
NUM_EPOCHS=4
PHASE1_EPOCHS=(10 12 3 4)   # List of phase 1 epochs to run phase 2 from
PHASE1_OUTPUT_DIR=/projects/ai_safe/cuongdc/adaptive_${MODEL_SIZE}_low
mkdir -p data/$MODEL_SIZE

# Check there are 2 GPUs in GPUS
if [ $(echo $GPUS | tr "," "\n" | wc -l) -ne 2 ]; then
    echo "Please specify 2 GPUs in the GPUS variable."
    exit 1
fi

GPU_0=$(echo $GPUS | cut -d',' -f1)
GPU_1=$(echo $GPUS | cut -d',' -f2)

for PHASE1_EPOCH in "${PHASE1_EPOCHS[@]}"; do
    OUTPUT_DIR=/projects/ai_safe/cuongdc/adaptive_${MODEL_SIZE}_low_phase2_from_phase1_epoch_${PHASE1_EPOCH}
    mkdir -p $OUTPUT_DIR

    PHASE1_MODEL=$PHASE1_OUTPUT_DIR/epoch_${PHASE1_EPOCH}
    echo "========================================"
    echo "Starting phase 2 from phase 1 epoch ${PHASE1_EPOCH}: $PHASE1_MODEL"
    echo "========================================"

    CURRENT_MODEL=$PHASE1_MODEL

    # --- Data selection (done once for all phase 2 epochs) ---
    # Parts 1 and 2 in parallel, 1 GPU each
    CUDA_VISIBLE_DEVICES=$GPU_0 python data/data_processing_utils/calculate_log_likelihood.py \
        --model_name_or_path $CURRENT_MODEL \
        --input_path data/math12K_phase2_teacher_answers_part1.json \
        --output_path data/$MODEL_SIZE/math12K_phase2_likelihood_part1.json \
        --field_name llh &
    CUDA_VISIBLE_DEVICES=$GPU_1 python data/data_processing_utils/calculate_log_likelihood.py \
        --model_name_or_path $CURRENT_MODEL \
        --input_path data/math12K_phase2_teacher_answers_part2.json \
        --output_path data/$MODEL_SIZE/math12K_phase2_likelihood_part2.json \
        --field_name llh &
    wait

    python data/data_processing_utils/merge_loglikelihood_parts.py \
        --pattern "data/$MODEL_SIZE/math12K_phase2_likelihood_part*.json" \
        --output_path data/$MODEL_SIZE/math12K_likelihood.json

    python data/data_processing_utils/select_math12k_answers_by_field.py \
        --input_file data/$MODEL_SIZE/math12K_likelihood.json \
        --output_dir data/$MODEL_SIZE/ \
        --field llh

    rm data/$MODEL_SIZE/math12K_phase2_likelihood_part*.json

    for epoch in $(seq 1 $NUM_EPOCHS); do
        echo "========================================"
        echo "Phase 1 epoch ${PHASE1_EPOCH} -> Phase 2 epoch $epoch / $NUM_EPOCHS"
        echo "Model: $CURRENT_MODEL"
        echo "========================================"

        EPOCH_OUTPUT_DIR=$OUTPUT_DIR/epoch_${epoch}
        CUDA_VISIBLE_DEVICES=$GPUS llamafactory-cli train \
            --model_name_or_path $CURRENT_MODEL \
            --stage sft \
            --do_train true \
            --finetuning_type full \
            --deepspeed examples/deepspeed/ds_z3_config.json \
            --flash_attn fa2 \
            --gradient_checkpointing true \
            --dataset 3B_high \
            --cutoff_len 8192 \
            --overwrite_cache true \
            --preprocessing_num_workers 64 \
            --template qwen \
            --output_dir $EPOCH_OUTPUT_DIR \
            --logging_steps 1 \
            --save_strategy steps \
            --save_steps 50 \
            --plot_loss true \
            --overwrite_output_dir true \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1.25e-6 \
            --num_train_epochs 1 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.0 \
            --bf16 true \
            --ddp_timeout 180000000 \
            --save_only_model true

        CURRENT_MODEL=$EPOCH_OUTPUT_DIR/
    done
done
