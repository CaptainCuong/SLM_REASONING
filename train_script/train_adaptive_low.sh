GPUS=0,1
MODEL_SIZE=3B
NUM_EPOCHS=15
START_EPOCH=1
OUTPUT_DIR=/helios-storage/helios4-data/cuong/model/adaptive_${MODEL_SIZE}_low
mkdir -p $OUTPUT_DIR
mkdir -p data/$MODEL_SIZE

if [ "$MODEL_SIZE" == "7B" ]; then
    MODEL_NAME=Qwen/Qwen2.5-Math-7B
    TRAIN_DATA=7B_low
elif [ "$MODEL_SIZE" == "3B" ]; then
    MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
    TRAIN_DATA=3B_low
elif [ "$MODEL_SIZE" == "1.5B" ]; then
    MODEL_NAME=Qwen/Qwen2.5-1.5B
    TRAIN_DATA=1.5B_low
else
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE. Please set it to 1.5B, 3B or 7B."
    exit 1
fi

# Check there are 2 GPUs in GPU
if [ $(echo $GPUS | tr "," "\n" | wc -l) -ne 2 ]; then
    echo "Please specify 2 GPUs in the GPUS variable."
    exit 1
fi

GPU_0=$(echo $GPUS | cut -d',' -f1)
GPU_1=$(echo $GPUS | cut -d',' -f2)

# Adaptive training loop: recompute data selection at the start of every epoch
# Resume from the last completed epoch if START_EPOCH > 1
if [ "$START_EPOCH" -gt 1 ]; then
    CURRENT_MODEL=$OUTPUT_DIR/epoch_$((START_EPOCH - 1))
    echo "Resuming from epoch $((START_EPOCH - 1)): $CURRENT_MODEL"
else
    CURRENT_MODEL=$MODEL_NAME
fi
for epoch in $(seq $START_EPOCH $NUM_EPOCHS); do
    echo "========================================"
    echo "Epoch $epoch / $NUM_EPOCHS"
    echo "Model: $CURRENT_MODEL"
    echo "========================================"

    # --- Data selection ---
    # Round 1: parts 1 and 2 in parallel, 1 GPU each
    CUDA_VISIBLE_DEVICES=$GPU_0 python data/data_processing_utils/calculate_log_likelihood.py \
        --model_name_or_path $CURRENT_MODEL \
        --input_path data/math12K_teacher_answers_part1.json \
        --output_path data/$MODEL_SIZE/math12K_likelihood_part1.json \
        --field_name llh &
    CUDA_VISIBLE_DEVICES=$GPU_1 python data/data_processing_utils/calculate_log_likelihood.py \
        --model_name_or_path $CURRENT_MODEL \
        --input_path data/math12K_teacher_answers_part2.json \
        --output_path data/$MODEL_SIZE/math12K_likelihood_part2.json \
        --field_name llh &
    wait

    # Round 2: parts 3 and 4 in parallel, 1 GPU each
    CUDA_VISIBLE_DEVICES=$GPU_0 python data/data_processing_utils/calculate_log_likelihood.py \
        --model_name_or_path $CURRENT_MODEL \
        --input_path data/math12K_teacher_answers_part3.json \
        --output_path data/$MODEL_SIZE/math12K_likelihood_part3.json \
        --field_name llh &
    CUDA_VISIBLE_DEVICES=$GPU_1 python data/data_processing_utils/calculate_log_likelihood.py \
        --model_name_or_path $CURRENT_MODEL \
        --input_path data/math12K_teacher_answers_part4.json \
        --output_path data/$MODEL_SIZE/math12K_likelihood_part4.json \
        --field_name llh &
    wait

    python data/data_processing_utils/merge_loglikelihood_parts.py \
        --pattern "data/$MODEL_SIZE/math12K_likelihood_part*.json" \
        --output_path data/$MODEL_SIZE/math12K_likelihood.json

    python data/data_processing_utils/select_math12k_answers_by_field.py \
        --input_file data/$MODEL_SIZE/math12K_likelihood.json \
        --output_dir data/$MODEL_SIZE/ \
        --field llh

    rm data/$MODEL_SIZE/math12K_likelihood_part*.json

    # --- Training ---
    EPOCH_OUTPUT_DIR=$OUTPUT_DIR/epoch_${epoch}
    CUDA_VISIBLE_DEVICES=$GPUS llamafactory-cli train \
        --model_name_or_path $CURRENT_MODEL \
        --stage sft \
        --do_train true \
        --finetuning_type full \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --flash_attn fa2 \
        --gradient_checkpointing true \
        --dataset $TRAIN_DATA \
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