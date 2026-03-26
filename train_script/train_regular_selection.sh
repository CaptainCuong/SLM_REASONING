GPUS=0,1
NUM_EPOCHS=5
MODEL_NAME="Qwen/Qwen3.5-4B"
LR=1.25e-6
MODE="both" # "high" or "low" or "both"
DATA_FOLDER=data/$(basename $MODEL_NAME)
mkdir -p $DATA_FOLDER

# Check there are 2 GPUs in GPUS
if [ $(echo $GPUS | tr "," "\n" | wc -l) -ne 2 ]; then
    echo "Please specify 2 GPUs in the GPUS variable."
    exit 1
fi

GPU_0=$(echo $GPUS | cut -d',' -f1)
GPU_1=$(echo $GPUS | cut -d',' -f2)


echo "MODEL_NAME: $MODEL_NAME"
echo "LR: $LR"


# --- Data selection (done once for all epochs) ---
CUDA_VISIBLE_DEVICES=$GPU_0 python data/data_processing_utils/calculate_log_likelihood.py \
    --model_name_or_path $MODEL_NAME \
    --input_path data/math12K_merged_loglikelihood_part1.json \
    --output_path $DATA_FOLDER/math12K_likelihood_part1.json \
    --field_name llh &
CUDA_VISIBLE_DEVICES=$GPU_1 python data/data_processing_utils/calculate_log_likelihood.py \
    --model_name_or_path $MODEL_NAME \
    --input_path data/math12K_merged_loglikelihood_part2.json \
    --output_path $DATA_FOLDER/math12K_likelihood_part2.json \
    --field_name llh &
wait

CUDA_VISIBLE_DEVICES=$GPU_0 python data/data_processing_utils/calculate_log_likelihood.py \
    --model_name_or_path $MODEL_NAME \
    --input_path data/math12K_merged_loglikelihood_part3.json \
    --output_path $DATA_FOLDER/math12K_likelihood_part3.json \
    --field_name llh &
CUDA_VISIBLE_DEVICES=$GPU_1 python data/data_processing_utils/calculate_log_likelihood.py \
    --model_name_or_path $MODEL_NAME \
    --input_path data/math12K_merged_loglikelihood_part4.json \
    --output_path $DATA_FOLDER/math12K_likelihood_part4.json \
    --field_name llh &
wait

python data/data_processing_utils/merge_loglikelihood_parts.py \
    --pattern "$DATA_FOLDER/math12K_likelihood_part*.json" \
    --output_path $DATA_FOLDER/math12K_likelihood.json

python data/data_processing_utils/select_math12k_answers_by_field.py \
    --input_file $DATA_FOLDER/math12K_likelihood.json \
    --output_dir $DATA_FOLDER/ \
    --field llh

rm $DATA_FOLDER/math12K_likelihood_part*.json

# Register datasets in dataset_info.json
MODEL_BASE=$(basename $MODEL_NAME)
DATASET_HIGH="${MODEL_BASE}_high"
DATASET_LOW="${MODEL_BASE}_low"
python3 -c "
import json
with open('data/dataset_info.json', 'r') as f:
    info = json.load(f)
if '${DATASET_HIGH}' not in info:
    info['${DATASET_HIGH}'] = {'file_name': '${MODEL_BASE}/math12K_highest_llh.json'}
if '${DATASET_LOW}' not in info:
    info['${DATASET_LOW}'] = {'file_name': '${MODEL_BASE}/math12K_lowest_llh.json'}
with open('data/dataset_info.json', 'w') as f:
    json.dump(info, f, indent=2)
"

train() {
    local DATASET=$1
    local OUT=$2
    CUDA_VISIBLE_DEVICES=$GPUS \
    llamafactory-cli train \
        --model_name_or_path $MODEL_NAME \
        --stage sft \
        --do_train true \
        --finetuning_type full \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --flash_attn fa2 \
        --gradient_checkpointing true \
        --dataset $DATASET \
        --cutoff_len 8192 \
        --overwrite_cache true \
        --preprocessing_num_workers 64 \
        --template qwen \
        --output_dir $OUT \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 1100 \
        --plot_loss true \
        --overwrite_output_dir true \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate $LR \
        --num_train_epochs $NUM_EPOCHS \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.0 \
        --bf16 true \
        --ddp_timeout 180000000 \
        --save_only_model true
}

OUTPUT_DIR_LOW=/projects/ai_safe/cuongdc/$(basename $MODEL_NAME)_low
OUTPUT_DIR_HIGH=/projects/ai_safe/cuongdc/$(basename $MODEL_NAME)_high

if [ "$MODE" == "high" ]; then
    train $DATASET_HIGH $OUTPUT_DIR_HIGH
elif [ "$MODE" == "low" ]; then
    train $DATASET_LOW $OUTPUT_DIR_LOW
elif [ "$MODE" == "both" ]; then
    train $DATASET_HIGH $OUTPUT_DIR_HIGH
    train $DATASET_LOW $OUTPUT_DIR_LOW
else
    echo "Invalid MODE: $MODE. Must be 'high', 'low', or 'both'."
    exit 1
fi