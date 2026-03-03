GPUS=0,1
MODEL_SIZE=3B
NUM_EPOCHS=4
START_EPOCH=1
PHASE1_EPOCH=14
PHASE1_OUTPUT_DIR=/helios-storage/helios4-data/cuong/model/adaptive_${MODEL_SIZE}_low
OUTPUT_DIR=/helios-storage/helios4-data/cuong/model/adaptive_${MODEL_SIZE}_low_phase2_from_phase1_epoch_${PHASE1_EPOCH}
mkdir -p $OUTPUT_DIR
mkdir -p data/$MODEL_SIZE

# Check there are 2 GPUs in GPUS
if [ $(echo $GPUS | tr "," "\n" | wc -l) -ne 2 ]; then
    echo "Please specify 2 GPUs in the GPUS variable."
    exit 1
fi

GPU_0=$(echo $GPUS | cut -d',' -f1)
GPU_1=$(echo $GPUS | cut -d',' -f2)

# Load from specified phase 1 epoch
PHASE1_MODEL=$PHASE1_OUTPUT_DIR/epoch_${PHASE1_EPOCH}
echo "Starting phase 2 from phase 1 epoch ${PHASE1_EPOCH}: $PHASE1_MODEL"

# Resume from last completed phase 2 epoch if START_EPOCH > 1
if [ "$START_EPOCH" -gt 1 ]; then
    CURRENT_MODEL=$OUTPUT_DIR/epoch_$((START_EPOCH - 1))
    echo "Resuming phase 2 from epoch $((START_EPOCH - 1)): $CURRENT_MODEL"
else
    CURRENT_MODEL=$PHASE1_MODEL
fi

for epoch in $(seq $START_EPOCH $NUM_EPOCHS); do
    echo "========================================"
    echo "Phase 2 - Epoch $epoch / $NUM_EPOCHS"
    echo "Model: $CURRENT_MODEL"
    echo "========================================"

    # --- Data selection ---
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

    # --- Training on 3B_high ---
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
        --save_steps 1100 \
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

    # Use this epoch's checkpoint as the model for the next epoch
    CURRENT_MODEL=$EPOCH_OUTPUT_DIR/
done
