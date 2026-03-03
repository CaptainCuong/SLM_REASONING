DATASETS="amc olympiadbench"

# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_high_1.5B --base_model_dir eval/outputs/Qwen/Qwen2.5-1.5B --dataset amc --output_file prob_tracking/data/track_test_set/test_amc_high_1.5B.json
# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_low_1.5B --base_model_dir eval/outputs/Qwen/Qwen2.5-1.5B --dataset amc --output_file prob_tracking/data/track_test_set/test_amc_low_1.5B.json

# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_high_3B --base_model_dir eval/outputs/Qwen/Qwen2.5-3B-Instruct --dataset amc --output_file prob_tracking/data/track_test_set/test_amc_high_3B.json
# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_low_3B --base_model_dir eval/outputs/Qwen/Qwen2.5-3B-Instruct --dataset amc --output_file prob_tracking/data/track_test_set/test_amc_low_3B.json

# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_high_3B_dpo --base_model_dir eval/outputs/Qwen/Qwen2.5-3B-Instruct --dataset amc --output_file prob_tracking/data/track_test_set/test_amc_high_3B_dpo.json

# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_high_7B --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset amc --output_file prob_tracking/data/track_test_set/test_amc_high_7B.json
# python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_low_7B --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset amc --output_file prob_tracking/data/track_test_set/test_amc_low_7B.json
python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_high_7B --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset olympiadbench --output_file prob_tracking/data/track_test_set/test_olympiadbench_high_7B.json
python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py --model_dir eval/outputs/cuongdc/Qwen_Math_low_7B --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B --dataset olympiadbench --output_file prob_tracking/data/track_test_set/test_olympiadbench_low_7B.json

# OUTPUT_FILE="prob_tracking/data/track_test_set/test_amc_teacher.json"
# touch "$OUTPUT_FILE"
# printf '[\n]\n' > "$OUTPUT_FILE"
# python prob_tracking/generate_utils/add_eval_samples_to_pool.py --eval_file eval/outputs/google/gemma-3-27b-it/amc/test_qwen-instruct_t0.0_k1_s0_e40.jsonl --pool_file prob_tracking/data/track_test_set/test_amc_teacher.json --output_file prob_tracking/data/track_test_set/test_amc_teacher.json --tag gemma
# python prob_tracking/generate_utils/add_eval_samples_to_pool.py --eval_file eval/outputs/Qwen/Qwen2.5-72B/amc/test_qwen-instruct_t0.0_k1_s0_e40.jsonl --pool_file prob_tracking/data/track_test_set/test_amc_teacher.json --output_file prob_tracking/data/track_test_set/test_amc_teacher.json --tag qwen72

