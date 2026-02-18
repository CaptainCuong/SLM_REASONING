# mkdir -p ntk/image/train_test_gap
# MODEL_SIZE="1.5 3 7"
# EXPS="high low"
# for MODEL in $MODEL_SIZE; do
#     for EXP in $EXPS; do
#         python ntk/visualize/visualize_n_calculate_avg_loss_multiple.py \
#                 --results_paths ntk/results/train_set_results/losses_math12k_${EXP}_${MODEL}B.json \
#                                 ntk/results/test_set_results/losses_amc_${EXP}_${MODEL}B.json \
#                 --labels "${MODEL} ${EXP} TRAIN" "${MODEL} ${EXP} TEST" \
#                 --visualize --output_plot ntk/image/train_test_gap/${EXP}_${MODEL}B.png
#     done
# done


# mkdir -p ntk/image/base_teacher_gap
# MODEL_SIZE="1.5 3 7"
# EXPS="high low"
# for MODEL in $MODEL_SIZE; do
#     for EXP in $EXPS; do
#         python ntk/visualize/visualize_n_calculate_avg_loss_multiple.py \
#                 --results_paths ntk/results/train_set_results/losses_math12k_${EXP}_${MODEL}B.json \
#                                 ntk/results/train_set_results/losses_math12k_${EXP}_${MODEL}B.json \
#                 --labels "${MODEL} ${EXP} TRAIN" "${MODEL} ${EXP} TEST" \
#                 --visualize --output_plot ntk/image/base_teacher_gap/${EXP}_${MODEL}B.png
#     done
# done

mkdir -p ntk/image/base_high_low_gap
MODEL_SIZE="1.5 3 7"
for MODEL in $MODEL_SIZE; do
    # python ntk/visualize/visualize_n_calculate_avg_loss_multiple.py \
    #         --results_paths ntk/results/train_set_results/losses_math12k_high_${MODEL}B.json \
    #                         ntk/results/train_set_results/losses_math12k_low_${MODEL}B.json \
    #         --labels "${MODEL} HIGH TRAIN" "${MODEL} LOW TRAIN" \
    #         --visualize --output_plot ntk/image/base_high_low_gap/${MODEL}B_base.png
    python ntk/visualize/visualize_loss_distribution.py \
        --results_path ntk/results/train_set_results/losses_final_cp_high_${MODEL}B.json \
                       ntk/results/train_set_results/losses_final_cp_low_${MODEL}B.json \
        --patterns "*base*" \
        --labels "${MODEL}B HIGH" "${MODEL}B LOW" \
        --plot_type histogram \
        --output_plot ntk/image/base_high_low_gap/${MODEL}B_final_loss_gap.png
done