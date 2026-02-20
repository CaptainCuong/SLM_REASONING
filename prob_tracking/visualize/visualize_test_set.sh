QUESTIONS="1 2 3 4 5"

mkdir -p prob_tracking/image/track_test_set

############### 1.5B ###############
for Q in $QUESTIONS; do
mkdir -p prob_tracking/image/track_test_set/high_1.5B/Q${Q}
mkdir -p prob_tracking/image/track_test_set/low_1.5B/Q${Q}
mkdir -p prob_tracking/image/track_test_set/high_1.5B/Q${Q}/raw
mkdir -p prob_tracking/image/track_test_set/low_1.5B/Q${Q}/raw
mkdir -p prob_tracking/image/track_test_set/high_1.5B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_test_set/low_1.5B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_test_set/high_1.5B/Q${Q}/source
mkdir -p prob_tracking/image/track_test_set/low_1.5B/Q${Q}/source
mkdir -p prob_tracking/image/track_test_set/high_1.5B/Q${Q}/multiple_attributes
mkdir -p prob_tracking/image/track_test_set/low_1.5B/Q${Q}/multiple_attributes

# ### Raw visualization ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_1.5B/Q${Q}/raw --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_1.5B/Q${Q}/raw --no_show

# ### Color by correctness ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_1.5B/Q${Q}/correctness --color_by_correctness --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_1.5B/Q${Q}/correctness --color_by_correctness --no_show

# ### Color by source ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_1.5B/Q${Q}/source --color_by_source --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_1.5B/Q${Q}/source --color_by_source --no_show

### Color by multiple attributes ###
python prob_tracking/visualize/visualize_multi_factor.py --summary_path prob_tracking/results/test_amc_high_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_1.5B/Q${Q}/multiple_attributes
python prob_tracking/visualize/visualize_multi_factor.py --summary_path prob_tracking/results/test_amc_low_1.5B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_1.5B/Q${Q}/multiple_attributes

done


############### 3B ###############
for Q in $QUESTIONS; do
mkdir -p prob_tracking/image/track_test_set/high_3B/Q${Q}
mkdir -p prob_tracking/image/track_test_set/low_3B/Q${Q}
mkdir -p prob_tracking/image/track_test_set/high_3B/Q${Q}/raw
mkdir -p prob_tracking/image/track_test_set/low_3B/Q${Q}/raw
mkdir -p prob_tracking/image/track_test_set/high_3B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_test_set/low_3B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_test_set/high_3B/Q${Q}/source
mkdir -p prob_tracking/image/track_test_set/low_3B/Q${Q}/source
mkdir -p prob_tracking/image/track_test_set/high_3B/Q${Q}/multiple_attributes
mkdir -p prob_tracking/image/track_test_set/low_3B/Q${Q}/multiple_attributes

# ### Raw visualization ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_3B/Q${Q}/raw --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_3B/Q${Q}/raw --no_show

# #### Color by correctness ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_3B/Q${Q}/correctness --color_by_correctness --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_3B/Q${Q}/correctness --color_by_correctness --no_show

# ### Color by source ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_3B/Q${Q}/source --color_by_source --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_3B/Q${Q}/source --color_by_source --no_show

### Color by multiple attributes ###
python prob_tracking/visualize/visualize_multi_factor.py --summary_path prob_tracking/results/test_amc_high_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_3B/Q${Q}/multiple_attributes
python prob_tracking/visualize/visualize_multi_factor.py --summary_path prob_tracking/results/test_amc_low_3B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_3B/Q${Q}/multiple_attributes

done


############### 7B ###############
for Q in $QUESTIONS; do
mkdir -p prob_tracking/image/track_test_set/high_7B/Q${Q}
mkdir -p prob_tracking/image/track_test_set/low_7B/Q${Q}
mkdir -p prob_tracking/image/track_test_set/high_7B/Q${Q}/raw
mkdir -p prob_tracking/image/track_test_set/low_7B/Q${Q}/raw
mkdir -p prob_tracking/image/track_test_set/high_7B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_test_set/low_7B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_test_set/high_7B/Q${Q}/source
mkdir -p prob_tracking/image/track_test_set/low_7B/Q${Q}/source
mkdir -p prob_tracking/image/track_test_set/high_7B/Q${Q}/multiple_attributes
mkdir -p prob_tracking/image/track_test_set/low_7B/Q${Q}/multiple_attributes

# ### Raw visualization ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_7B/Q${Q}/raw --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_7B/Q${Q}/raw --no_show

# #### Color by correctness ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_7B/Q${Q}/correctness --color_by_correctness --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_7B/Q${Q}/correctness --color_by_correctness --no_show

# ### Color by source ###
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_high_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_7B/Q${Q}/source --color_by_source --no_show
# python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/test_amc_low_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_7B/Q${Q}/source --color_by_source --no_show

### Color by multiple attributes ###
python prob_tracking/visualize/visualize_multi_factor.py --summary_path prob_tracking/results/test_amc_high_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/high_7B/Q${Q}/multiple_attributes
python prob_tracking/visualize/visualize_multi_factor.py --summary_path prob_tracking/results/test_amc_low_7B_all_checkpoints_summary.json --question ${Q} --output_dir prob_tracking/image/track_test_set/low_7B/Q${Q}/multiple_attributes

done