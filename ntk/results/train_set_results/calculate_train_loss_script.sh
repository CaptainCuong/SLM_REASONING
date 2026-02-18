# python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_training_set/train_high_7B.json --output_path ntk/results/train_set_results/losses_math12k_high_7B.json
# python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_training_set/train_low_7B.json --output_path ntk/results/train_set_results/losses_math12k_low_7B.json

# python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_training_set/train_high_3B.json --output_path ntk/results/train_set_results/losses_math12k_high_3B.json
# python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_training_set/train_low_3B.json --output_path ntk/results/train_set_results/losses_math12k_low_3B.json

# python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_training_set/train_high_1.5B.json --output_path ntk/results/train_set_results/losses_math12k_high_1.5B.json
# python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_training_set/train_low_1.5B.json --output_path ntk/results/train_set_results/losses_math12k_low_1.5B.json

##############################################

python ntk/loss_utils/calculate_loss.py --model_name_or_path google/gemma-3-27b-it --pool_path prob_tracking/data/track_training_set/train_high_7B.json --output_path ntk/results/train_set_results/losses_gemma_math12k_high_7B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path google/gemma-3-27b-it --pool_path prob_tracking/data/track_training_set/train_low_7B.json --output_path ntk/results/train_set_results/losses_gemma_math12k_low_7B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path google/gemma-3-27b-it --pool_path prob_tracking/data/track_training_set/train_high_3B.json --output_path ntk/results/train_set_results/losses_gemma_math12k_high_3B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path google/gemma-3-27b-it --pool_path prob_tracking/data/track_training_set/train_low_3B.json --output_path ntk/results/train_set_results/losses_gemma_math12k_low_3B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path google/gemma-3-27b-it --pool_path prob_tracking/data/track_training_set/train_high_1.5B.json --output_path ntk/results/train_set_results/losses_gemma_math12k_high_1.5B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path google/gemma-3-27b-it --pool_path prob_tracking/data/track_training_set/train_low_1.5B.json --output_path ntk/results/train_set_results/losses_gemma_math12k_low_1.5B.json

#################### LOSS UNDER FINAL CHECKPOINT ##########################

python ntk/loss_utils/calculate_loss.py --model_name_or_path /projects/ai_safe/cuongdc/Qwen_Math_high_1.5B/checkpoint-22000/ --pool_path prob_tracking/data/track_training_set/train_high_1.5B.json --output_path ntk/results/train_set_results/losses_final_cp_high_1.5B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path /projects/ai_safe/cuongdc/Qwen_Math_low_1.5B/checkpoint-22000/ --pool_path prob_tracking/data/track_training_set/train_low_1.5B.json --output_path ntk/results/train_set_results/losses_final_cp_low_1.5B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path /projects/ai_safe/cuongdc/Qwen_high_3B/checkpoint-22000/ --pool_path prob_tracking/data/track_training_set/train_high_3B.json --output_path ntk/results/train_set_results/losses_final_cp_high_3B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path /projects/ai_safe/cuongdc/Qwen_low_3B/checkpoint-22000/ --pool_path prob_tracking/data/track_training_set/train_low_3B.json --output_path ntk/results/train_set_results/losses_final_cp_low_3B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path /projects/ai_safe/cuongdc/Qwen_Math_high/checkpoint-10990/ --pool_path prob_tracking/data/track_training_set/train_high_7B.json --output_path ntk/results/train_set_results/losses_final_cp_high_7B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path /projects/ai_safe/cuongdc/Qwen_Math_low/checkpoint-10990/ --pool_path prob_tracking/data/track_training_set/train_low_7B.json --output_path ntk/results/train_set_results/losses_final_cp_low_7B.json