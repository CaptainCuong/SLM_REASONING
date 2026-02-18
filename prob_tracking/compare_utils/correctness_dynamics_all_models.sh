python prob_tracking/compare_utils/visualize_correctness_dynamics.py --base_model_dir eval/outputs/Qwen/Qwen2.5-1.5B/ --checkpoints_dir eval/outputs/cuongdc/Qwen_Math_high_1.5B/ --dataset amc
python prob_tracking/compare_utils/visualize_correctness_dynamics.py --base_model_dir eval/outputs/Qwen/Qwen2.5-1.5B/ --checkpoints_dir eval/outputs/cuongdc/Qwen_Math_low_1.5B/ --dataset amc

python prob_tracking/compare_utils/visualize_correctness_dynamics.py --base_model_dir eval/outputs/Qwen/Qwen2.5-3B-Instruct/ --checkpoints_dir eval/outputs/cuongdc/Qwen_high_3B/ --dataset amc
python prob_tracking/compare_utils/visualize_correctness_dynamics.py --base_model_dir eval/outputs/Qwen/Qwen2.5-3B-Instruct/ --checkpoints_dir eval/outputs/cuongdc/Qwen_low_3B/ --dataset amc

python prob_tracking/compare_utils/visualize_correctness_dynamics.py --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B/ --checkpoints_dir eval/outputs/cuongdc/Qwen_Math_high_7B/ --dataset amc
python prob_tracking/compare_utils/visualize_correctness_dynamics.py --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B/ --checkpoints_dir eval/outputs/cuongdc/Qwen_Math_low_7B/ --dataset amc