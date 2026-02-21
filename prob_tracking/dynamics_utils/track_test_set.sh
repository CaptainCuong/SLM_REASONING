# ############### Track 1.5B ###############
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_high_1.5B.json --base_model_path Qwen/Qwen2.5-1.5B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_high_1.5B/
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_low_1.5B.json --base_model_path Qwen/Qwen2.5-1.5B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_low_1.5B/

# ############### Track 3B ###############
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_high_3B.json --base_model_path Qwen/Qwen2.5-3B-Instruct --model_folder /projects/ai_safe/cuongdc/Qwen_high_3B/
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_low_3B.json --base_model_path Qwen/Qwen2.5-3B-Instruct --model_folder /projects/ai_safe/cuongdc/Qwen_low_3B/

# ############### Track 7B ###############
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_high_7B.json --base_model_path Qwen/Qwen2.5-Math-7B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_high/
# python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_low_7B.json --base_model_path Qwen/Qwen2.5-Math-7B --model_folder /projects/ai_safe/cuongdc/Qwen_Math_low/

############### Track Teacher ###############
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_teacher.json --base_model_path Qwen/Qwen2.5-1.5B --model_folder /helios-storage/helios4-data/cuong/model/Qwen_Math_high_1.5B/ --output_path prob_tracking/results/test_amc_teacher_high_1.5B.json
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_teacher.json --base_model_path Qwen/Qwen2.5-1.5B --model_folder /helios-storage/helios4-data/cuong/model/Qwen_Math_low_1.5B/ --output_path prob_tracking/results/test_amc_teacher_low_1.5B.json

python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_teacher.json --base_model_path Qwen/Qwen2.5-3B-Instruct --model_folder /helios-storage/helios4-data/cuong/model/Qwen_high_3B/ --output_path prob_tracking/results/test_amc_teacher_high_3B.json
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_teacher.json --base_model_path Qwen/Qwen2.5-3B-Instruct --model_folder /helios-storage/helios4-data/cuong/model/Qwen_low_3B/ --output_path prob_tracking/results/test_amc_teacher_low_3B.json

python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_teacher.json --base_model_path Qwen/Qwen2.5-Math-7B --model_folder /helios-storage/helios4-data/cuong/model/Math12K_high_lr5e-6_bs2_gas_1_2H200/ --output_path prob_tracking/results/test_amc_teacher_high_7B.json
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py --pool_path prob_tracking/data/track_test_set/test_amc_teacher.json --base_model_path Qwen/Qwen2.5-Math-7B --model_folder /helios-storage/helios4-data/cuong/model/Math12K_low_lr5e-6_bs2_gas_1_2H200/ --output_path prob_tracking/results/test_amc_teacher_low_7B.json