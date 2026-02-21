python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --model_folder /projects/ai_safe/cuongdc/Qwen_Math_high_1.5B --base_model_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_test_set/test_amc_high_1.5B.json
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --model_folder /projects/ai_safe/cuongdc/Qwen_Math_low_1.5B --base_model_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_test_set/test_amc_low_1.5B.json

python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --model_folder /projects/ai_safe/cuongdc/Qwen_high_3B --base_model_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_test_set/test_amc_high_3B.json
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --model_folder /projects/ai_safe/cuongdc/Qwen_low_3B --base_model_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_test_set/test_amc_low_3B.json

python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --model_folder /projects/ai_safe/cuongdc/Qwen_Math_high --base_model_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_test_set/test_amc_high_7B.json
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --model_folder /projects/ai_safe/cuongdc/Qwen_Math_low --base_model_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_test_set/test_amc_low_7B.json