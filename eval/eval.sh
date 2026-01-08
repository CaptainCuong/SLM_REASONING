CUDA_VISIBLE_DEVICES='0,1' \
python eval.py \
--model_name_or_path "/helios-storage/helios3-data/cuong/model/Qwen_Math_high/checkpoint-555/" \
--data_name "olympiadbench" \
--prompt_type "qwen-instruct" \
--temperature 0.7 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 0.9 \
--surround_with_messages \