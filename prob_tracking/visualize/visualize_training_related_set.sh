QUESTIONS="1 2 3 4 5"

mkdir -p prob_tracking/image/track_training_related_set

############### 1.5B ###############
for Q in $QUESTIONS; do
mkdir -p prob_tracking/image/track_training_related_set/high_1.5B/Q${Q}
mkdir -p prob_tracking/image/track_training_related_set/low_1.5B/Q${Q}
mkdir -p prob_tracking/image/track_training_related_set/high_1.5B/Q${Q}/raw
mkdir -p prob_tracking/image/track_training_related_set/low_1.5B/Q${Q}/raw
mkdir -p prob_tracking/image/track_training_related_set/high_1.5B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_training_related_set/low_1.5B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_training_related_set/high_1.5B/Q${Q}/source
mkdir -p prob_tracking/image/track_training_related_set/low_1.5B/Q${Q}/source

### Raw visualization ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_1.5B_all_checkpoints_summary.json prob_tracking/results/track_push_down_1.5B_high.json prob_tracking/results/train_high_1.5B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_1.5B/Q${Q}/raw --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_1.5B_all_checkpoints_summary.json prob_tracking/results/track_push_down_1.5B_low.json prob_tracking/results/train_low_1.5B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_1.5B/Q${Q}/raw --no_show

#### Color by correctness ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_1.5B_all_checkpoints_summary.json prob_tracking/results/track_push_down_1.5B_high.json prob_tracking/results/train_high_1.5B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_1.5B/Q${Q}/correctness --color_by_correctness --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_1.5B_all_checkpoints_summary.json prob_tracking/results/track_push_down_1.5B_low.json prob_tracking/results/train_low_1.5B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_1.5B/Q${Q}/correctness --color_by_correctness --no_show

### Color by source ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_1.5B_all_checkpoints_summary.json prob_tracking/results/track_push_down_1.5B_high.json prob_tracking/results/train_high_1.5B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_1.5B/Q${Q}/source --color_by_source --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_1.5B_all_checkpoints_summary.json prob_tracking/results/track_push_down_1.5B_low.json prob_tracking/results/train_low_1.5B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_1.5B/Q${Q}/source --color_by_source --no_show
done


############### 3B ###############
for Q in $QUESTIONS; do
mkdir -p prob_tracking/image/track_training_related_set/high_3B/Q${Q}
mkdir -p prob_tracking/image/track_training_related_set/low_3B/Q${Q}
mkdir -p prob_tracking/image/track_training_related_set/high_3B/Q${Q}/raw
mkdir -p prob_tracking/image/track_training_related_set/low_3B/Q${Q}/raw
mkdir -p prob_tracking/image/track_training_related_set/high_3B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_training_related_set/low_3B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_training_related_set/high_3B/Q${Q}/source
mkdir -p prob_tracking/image/track_training_related_set/low_3B/Q${Q}/source

### Raw visualization ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_3B_all_checkpoints_summary.json prob_tracking/results/track_push_down_3B_high.json prob_tracking/results/train_high_3B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_3B/Q${Q}/raw --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_3B_all_checkpoints_summary.json prob_tracking/results/track_push_down_3B_low.json prob_tracking/results/train_low_3B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_3B/Q${Q}/raw --no_show

#### Color by correctness ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_3B_all_checkpoints_summary.json prob_tracking/results/track_push_down_3B_high.json prob_tracking/results/train_high_3B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_3B/Q${Q}/correctness --color_by_correctness --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_3B_all_checkpoints_summary.json prob_tracking/results/track_push_down_3B_low.json prob_tracking/results/train_low_3B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_3B/Q${Q}/correctness --color_by_correctness --no_show

### Color by source ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_3B_all_checkpoints_summary.json prob_tracking/results/track_push_down_3B_high.json prob_tracking/results/train_high_3B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_3B/Q${Q}/source --color_by_source --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_3B_all_checkpoints_summary.json prob_tracking/results/track_push_down_3B_low.json prob_tracking/results/train_low_3B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_3B/Q${Q}/source --color_by_source --no_show
done


############### 7B ###############
for Q in $QUESTIONS; do
mkdir -p prob_tracking/image/track_training_related_set/high_7B/Q${Q}
mkdir -p prob_tracking/image/track_training_related_set/low_7B/Q${Q}
mkdir -p prob_tracking/image/track_training_related_set/high_7B/Q${Q}/raw
mkdir -p prob_tracking/image/track_training_related_set/low_7B/Q${Q}/raw
mkdir -p prob_tracking/image/track_training_related_set/high_7B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_training_related_set/low_7B/Q${Q}/correctness
mkdir -p prob_tracking/image/track_training_related_set/high_7B/Q${Q}/source
mkdir -p prob_tracking/image/track_training_related_set/low_7B/Q${Q}/source

### Raw visualization ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_7B_all_checkpoints_summary.json prob_tracking/results/track_push_down_7B_high.json prob_tracking/results/train_high_7B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_7B/Q${Q}/raw --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_7B_all_checkpoints_summary.json prob_tracking/results/track_push_down_7B_low.json prob_tracking/results/train_low_7B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_7B/Q${Q}/raw --no_show

#### Color by correctness ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_7B_all_checkpoints_summary.json prob_tracking/results/track_push_down_7B_high.json prob_tracking/results/train_high_7B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_7B/Q${Q}/correctness --color_by_correctness --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_7B_all_checkpoints_summary.json prob_tracking/results/track_push_down_7B_low.json prob_tracking/results/train_low_7B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_7B/Q${Q}/correctness --color_by_correctness --no_show

### Color by source ###
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_high_7B_all_checkpoints_summary.json prob_tracking/results/track_push_down_7B_high.json prob_tracking/results/train_high_7B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/high_7B/Q${Q}/source --color_by_source --no_show
python prob_tracking/visualize/visualize_llh_dynamics.py --summary_path prob_tracking/results/train_low_7B_all_checkpoints_summary.json prob_tracking/results/track_push_down_7B_low.json prob_tracking/results/train_low_7B_track_related_all_checkpoints_summary.json --id_patterns id_${Q}_highest* id_${Q}_lowest* id_${Q}_random* id_${Q}_base* id_${Q}_problem_paraphrased* --output_dir prob_tracking/image/track_training_related_set/low_7B/Q${Q}/source --color_by_source --no_show
done
