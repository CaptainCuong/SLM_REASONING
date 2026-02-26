# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_high_7B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset math12k --output_file prob_tracking/data/track_training_set/train_high_7B.json
# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_low_7B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset math12k --output_file prob_tracking/data/track_training_set/train_low_7B.json

# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_high_3B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-3B-Instruct --dataset math12k --output_file prob_tracking/data/track_training_set/train_high_3B.json
# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_low_3B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-3B-Instruct --dataset math12k --output_file prob_tracking/data/track_training_set/train_low_3B.json

# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_high_1.5B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-1.5B --dataset math12k --output_file prob_tracking/data/track_training_set/train_high_1.5B.json
# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_low_1.5B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-1.5B --dataset math12k --output_file prob_tracking/data/track_training_set/train_low_1.5B.json

# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_high_grad_norm_7B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset math12k --output_file prob_tracking/data/track_training_set/train_high_grad_norm_7B.json
# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_low_grad_norm_7B/ --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset math12k --output_file prob_tracking/data/track_training_set/train_low_grad_norm_7B.json


###### Decoding 7B t=0.7 ######
CHECKPOINTS="555 1100 1650 2200 2750 3300 3850 4400 4995 5550 6105 6660 7215 7770 8325 8880 9435 9990 10545 10990"
OUTPUT_FILE="prob_tracking/data/track_training_set/train_low_7B_t0.7.json"
touch "$OUTPUT_FILE"
printf '[\n]\n' > "$OUTPUT_FILE"
OUTPUT_FILE="prob_tracking/data/track_training_set/train_high_7B_t0.7.json"
touch "$OUTPUT_FILE"
printf '[\n]\n' > "$OUTPUT_FILE"
python prob_tracking/generate_utils/add_eval_samples_to_pool.py --eval_file eval/outputs/Qwen/Qwen2.5-Math-7B/math12k/test_qwen-instruct_t0.7_k1_s0_e50.jsonl --pool_file prob_tracking/data/track_training_set/train_low_7B_t0.7.json --output_file prob_tracking/data/track_training_set/train_low_7B_t0.7.json --tag base_t0.7
python prob_tracking/generate_utils/add_eval_samples_to_pool.py --eval_file eval/outputs/Qwen/Qwen2.5-Math-7B/math12k/test_qwen-instruct_t0.7_k1_s0_e50.jsonl --pool_file prob_tracking/data/track_training_set/train_high_7B_t0.7.json --output_file prob_tracking/data/track_training_set/train_high_7B_t0.7.json --tag base_t0.7
# for CP in $CHECKPOINTS; do
#     python prob_tracking/generate_utils/add_eval_samples_to_pool.py --eval_file eval/outputs/model/Math12K_low_lr5e-6_bs2_gas_1_2H200/checkpoint-$CP/math12k/test_qwen-instruct_t0.7_k1_s0_e50.jsonl --pool_file prob_tracking/data/track_training_set/train_low_7B_t0.7.json --output_file prob_tracking/data/track_training_set/train_low_7B_t0.7.json --tag base_t0.7_cp_$CP
#     python prob_tracking/generate_utils/add_eval_samples_to_pool.py --eval_file eval/outputs/model/Math12K_high_lr5e-6_bs2_gas_1_2H200/checkpoint-$CP/math12k/test_qwen-instruct_t0.7_k1_s0_e50.jsonl --pool_file prob_tracking/data/track_training_set/train_high_7B_t0.7.json --output_file prob_tracking/data/track_training_set/train_high_7B_t0.7.json --tag base_t0.7_cp_$CP
# done
