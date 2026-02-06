python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_test_set/test_amc_high_7B.json --output_path ntk/results/test_set_results/losses_math12k_high_7B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-Math-7B --pool_path prob_tracking/data/track_test_set/test_amc_low_7B.json --output_path ntk/results/test_set_results/losses_math12k_low_7B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_test_set/test_amc_high_3B.json --output_path ntk/results/test_set_results/losses_math12k_high_3B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-3B-Instruct --pool_path prob_tracking/data/track_test_set/test_amc_low_3B.json --output_path ntk/results/test_set_results/losses_math12k_low_3B.json

python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_test_set/test_amc_high_1.5B.json --output_path ntk/results/test_set_results/losses_math12k_high_1.5B.json
python ntk/loss_utils/calculate_loss.py --model_name_or_path Qwen/Qwen2.5-1.5B --pool_path prob_tracking/data/track_test_set/test_amc_low_1.5B.json --output_path ntk/results/test_set_results/losses_math12k_low_1.5B.json