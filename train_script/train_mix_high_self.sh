RATIO_A=0.75
python data/data_processing_utils/mix_training_data.py \
    --file_a data/math12K_highest_likelihood.json \
    --file_b data/math12K_7B_solutions.json \
    --ratio_a $RATIO_A \
    --output data/math12K_high_self_mixed.json

OUTPUT_DIR=/projects/ai_safe/cuongdc/Qwen_mix_high_self_${RATIO_A}_7B
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-Math-7B \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --flash_attn fa2 \
    --gradient_checkpointing true \
    --dataset math12K_high_self_mixed \
    --cutoff_len 8192 \
    --overwrite_cache true \
    --preprocessing_num_workers 64 \
    --template qwen \
    --output_dir $OUTPUT_DIR \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 1110 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.25e-6 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --save_only_model true