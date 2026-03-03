CUDA_VISIBLE_DEVICES='2,3' \
python eval.py \
--model_name_or_path "/helios-storage/helios4-data/cuong/model/adaptive_7B_high/epoch_1/checkpoint-3330" \
--data_name "amc" \
--prompt_type "qwen-instruct" \
--temperature 0.0 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 0.9 \
--surround_with_messages \
--output_dir /helios-storage/helios4-data/cuong/data_storage/learning_dynamics/eval