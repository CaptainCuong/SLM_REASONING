python data/data_processing_utils/convert_to_dpo.py \
    --chosen /home/cuongdc/SLM_REASONING/data/math12K_highest_Qwen2_5_3B_Instruct_llh.json \
    --rejected /home/cuongdc/SLM_REASONING/temp/math12K_instruction_input_solutions.json \
    --output /home/cuongdc/SLM_REASONING/data/math12K_dpo_high_3B.json

python data/data_processing_utils/convert_to_dpo.py \
    --chosen /home/cuongdc/SLM_REASONING/data/math12K_lowest_Qwen2_5_3B_Instruct_llh.json \
    --rejected /home/cuongdc/SLM_REASONING/temp/math12K_instruction_input_solutions.json \
    --output /home/cuongdc/SLM_REASONING/data/math12K_dpo_low_3B.json