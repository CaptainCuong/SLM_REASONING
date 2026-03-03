# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_training_set/train_high_1.5B.json --base_model_path Qwen/Qwen2.5-1.5B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_high_1.5B/
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_training_set/train_low_1.5B.json --base_model_path Qwen/Qwen2.5-1.5B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_low_1.5B/

# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_training_set/train_high_3B.json --base_model_path Qwen/Qwen2.5-3B-Instruct --model_folder /projects/ai_safe/cuongdc/Qwen_high_3B/
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_training_set/train_low_3B.json --base_model_path Qwen/Qwen2.5-3B-Instruct --model_folder /projects/ai_safe/cuongdc/Qwen_low_3B/

# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_training_set/train_high_7B.json --base_model_path Qwen/Qwen2.5-Math-7B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_high/
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_training_set/train_low_7B.json --base_model_path Qwen/Qwen2.5-Math-7B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_low/

python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_training_set/train_low_7B_t0.7.json --base_model_path Qwen/Qwen2.5-Math-7B --model_folder /helios-storage/helios4-data/cuong/model/Math12K_low_lr5e-6_bs2_gas_1_2H200