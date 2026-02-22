MODEL_SIZES=(1.5B 3B 7B)
EXPS=(high low)
for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
    ### Visualize base answer on paraphrased problem with different decoding strategies ###
    for EXP in "${EXPS[@]}"; do
        python prob_tracking/visualize/visualize_avg_llh_by_pattern.py \
            --files prob_tracking/results/train_${EXP}_${MODEL_SIZE}_all_checkpoints_summary.json \
                    prob_tracking/results/train_${EXP}_${MODEL_SIZE}_track_related_all_checkpoints_summary.json \
            --patterns "*base_problem_paraphrased_greedy*" "*base_problem_paraphrased_t*" \
            --notations "Greedy Base Answer Paraphrased Problem" "Base Answer (t=0.7) Paraphrased Problem" \
            --output  prob_tracking/image/decoding_comparison/paraphrased_problem_${EXP}_${MODEL_SIZE}.png
    done
done