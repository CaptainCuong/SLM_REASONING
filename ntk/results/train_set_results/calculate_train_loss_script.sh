python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_training_set/train_high_7B.json --output_path ntk/results/train_set_results/losses_math12k_high_7B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_training_set/train_low_7B.json --output_path ntk/results/train_set_results/losses_math12k_low_7B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_training_set/train_high_3B.json --output_path ntk/results/train_set_results/losses_math12k_high_3B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_training_set/train_low_3B.json --output_path ntk/results/train_set_results/losses_math12k_low_3B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_training_set/train_high_1.5B.json --output_path ntk/results/train_set_results/losses_math12k_high_1.5B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_training_set/train_low_1.5B.json --output_path ntk/results/train_set_results/losses_math12k_low_1.5B.json