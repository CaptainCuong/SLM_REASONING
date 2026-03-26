# mistralai/Mistral-8x7B-Instruct-v0.1
# mistralai/Codestral-22B-v0.1
# meta-llama/Llama-3.1-70B-Instruct
CUDA_VISIBLE_DEVICES='0,1' \
python eval.py \
--model_name_or_path "/projects/ai_safe/cuongdc/gemma-3-4b-it_high/checkpoint-1100" \
--data_name "aime" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 8 \
--k 8 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 0.9 \
--surround_with_messages