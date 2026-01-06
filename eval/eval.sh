CUDA_VISIBLE_DEVICES='0,1' \
python eval.py \
--model_name_or_path "/projects/ai_safe/cuongdc/olympiad_bench_val/random_5k/checkpoint-1875" \
--data_name "olympiadbench" \
--prompt_type "qwen-instruct" \
--temperature 0.7 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 5 \
--k 5 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 0.9 \
--surround_with_messages \

