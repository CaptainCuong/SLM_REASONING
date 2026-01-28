# Probability Tracking

This module tracks how model likelihood metrics (log-likelihood, perplexity) evolve across training checkpoints for different sample types.

## Workflow Overview

```
eval/outputs/                    prob_tracking/data/              prob_tracking/results/           prob_tracking/image/
├── Qwen_Math_high/         ──►  pool_high.json              ──►  summary.json               ──►  avg_llh.png
│   ├── checkpoint-555/                                                                           avg_perplexity.png
│   ├── checkpoint-1110/
│   └── ...
└── Qwen/Qwen2.5-Math-7B/

   [convert_eval_with_base_to_pool.py]  [evaluate_llh_all_checkpoints.py]  [visualize_llh_dynamics.py]
```

---

## 1. Convert Evaluation Results to Pool

**Script:** `generate_utils/convert_eval_with_base_to_pool.py`

Converts evaluation results from the `eval/` folder into a pool.json file that can be used for likelihood tracking.

### What it does:
- Reads JSONL evaluation outputs from checkpoint directories
- Optionally includes base model (pre-training) evaluation results
- Creates a unified pool with consistent question IDs across all checkpoints
- Labels each entry with type: `id_{question_id}_{source}_{correct/incorrect}`

### Usage:

```bash
# Convert with base model included
python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py \
    --model_dir eval/outputs/cuongdc/Qwen_Math_high \
    --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B \
    --dataset amc_test \
    --output_file prob_tracking/data/amc_high_pool.json

# Convert checkpoints only (no base model)
python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py \
    --model_dir eval/outputs/cuongdc/Qwen_Math_high \
    --dataset math \
    --output_file prob_tracking/data/math_high_pool.json
```

### Arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_dir` | Directory containing checkpoint subdirectories | Required |
| `--base_model_dir` | Directory containing base model eval results | None |
| `--dataset` | Dataset name (subfolder in checkpoint dirs) | `math` |
| `--output_file` | Output pool JSON path | Required |
| `--instruction` | Instruction text for pool entries | `"Please reason step by step..."` |

### Output Format:
```json
[
  {
    "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
    "input": "What is 2 + 2?",
    "output": "The answer is \\boxed{4}.",
    "type": "id_1_base_correct"
  },
  {
    "instruction": "...",
    "input": "What is 2 + 2?",
    "output": "The answer is \\boxed{5}.",
    "type": "id_1_cp555_incorrect"
  }
]
```

### Type Format:
- Base model: `id_{question_id}_base_{correct/incorrect}`
- Checkpoints: `id_{question_id}_cp{step}_{correct/incorrect}`

---

## 2. Evaluate Log-Likelihood Across Checkpoints

**Script:** `dynamics_utils/evaluate_llh_all_checkpoints.py`

Calculates log-likelihood and perplexity metrics for each sample in the pool across all checkpoints.

### What it does:
- Loads each checkpoint sequentially
- Calculates metrics (perplexity, log-likelihood) for all pool samples
- Aggregates results by sample type
- Outputs a summary JSON with metrics across training steps

### Usage:

```bash
# Evaluate all checkpoints
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py \
    --model_folder /path/to/checkpoints/Qwen_Math_high \
    --pool_path prob_tracking/data/traceback_pool_high.json \
    --output_path prob_tracking/results/Qwen_Math_high_summary.json \
    --base_model_path Qwen/Qwen2.5-Math-7B

# Calculate on full sequence (prompt + output)
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py \
    --model_folder /path/to/checkpoints/Qwen_Math_high \
    --pool_path prob_tracking/data/pool.json \
    --include_instruction
```

### Arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_folder` | Directory containing checkpoint-* subdirs | Required |
| `--pool_path` | Path to pool.json file | `./prob_tracking/data/traceback_pool_high.json` |
| `--output_path` | Output summary JSON path | Auto-generated |
| `--device` | Device (cuda/cpu) | Auto-detect |
| `--base_model_path` | Path to base model | `Qwen/Qwen2.5-Math-7B` |
| `--include_instruction` | Calculate on full sequence | False |

### Output Format:
```json
{
  "model_folder": "/path/to/Qwen_Math_high",
  "num_checkpoints": 21,
  "checkpoints": ["base_model", "checkpoint-555", "checkpoint-1110", ...],
  "steps": [0, 555, 1110, ...],
  "results_by_type": {
    "id_1_base_correct": {
      "perplexity": [[...], [...], ...],
      "log_likelihood": [[...], [...], ...],
      "avg_perplexity": [1.23, 1.19, 1.15, ...],
      "avg_log_likelihood": [-0.21, -0.18, -0.14, ...]
    }
  }
}
```

---

## 3. Visualize Log-Likelihood Dynamics

**Script:** `visualize/visualize_llh_dynamics.py`

Creates plots showing how log-likelihood and perplexity evolve across training.

### What it does:
- Loads the checkpoint summary JSON
- Generates line plots for each sample type
- Shows metrics evolution from base model through all checkpoints
- Prints summary statistics (initial, final, change)

### Usage:

```bash
# Generate all plots
python prob_tracking/visualize/visualize_llh_dynamics.py \
    --summary_path prob_tracking/results/Qwen_Math_high_summary.json \
    --output_dir prob_tracking/image

# Generate only log-likelihood plot (no display)
python prob_tracking/visualize/visualize_llh_dynamics.py \
    --summary_path prob_tracking/results/summary.json \
    --plot_type llh \
    --no_show

# Generate only perplexity plot
python prob_tracking/visualize/visualize_llh_dynamics.py \
    --summary_path prob_tracking/results/summary.json \
    --plot_type perplexity
```

### Arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--summary_path` | Path to checkpoint summary JSON | `prob_tracking/results/Qwen_Math_high_all_checkpoints_summary.json` |
| `--output_dir` | Directory to save plots | `prob_tracking/image` |
| `--no_show` | Don't display plots (only save) | False |
| `--plot_type` | Plot type: `all`, `llh`, `perplexity` | `all` |

### Output:
- `avg_llh.png` - Average log-likelihood across training steps
- `avg_perplexity.png` - Average perplexity across training steps

---

## Complete Example

```bash
# Step 1: Convert evaluation results to pool
python prob_tracking/generate_utils/convert_eval_with_base_to_pool.py \
    --model_dir eval/outputs/cuongdc/Qwen_Math_high \
    --base_model_dir eval/outputs/Qwen/Qwen2.5-Math-7B \
    --dataset amc_test \
    --output_file prob_tracking/data/amc_high_pool.json

# Step 2: Evaluate log-likelihood across all checkpoints
python prob_tracking/dynamics_utils/evaluate_llh_all_checkpoints.py \
    --model_folder /helios-storage/helios3-data/cuong/model/Qwen_Math_high \
    --pool_path prob_tracking/data/amc_high_pool.json \
    --output_path prob_tracking/results/amc_high_summary.json \
    --base_model_path Qwen/Qwen2.5-Math-7B

# Step 3: Visualize the dynamics
python prob_tracking/visualize/visualize_llh_dynamics.py \
    --summary_path prob_tracking/results/amc_high_summary.json \
    --output_dir prob_tracking/image
```

---

## Directory Structure

```
prob_tracking/
├── README.md
├── data/                          # Pool files
│   ├── traceback_pool_high.json
│   └── amc_high_pool.json
├── results/                       # Summary JSONs
│   └── Qwen_Math_high_summary.json
├── image/                         # Plots
│   ├── avg_llh.png
│   └── avg_perplexity.png
├── dynamics_utils/
│   ├── calculate_llh.py           # Core metrics calculation
│   └── evaluate_llh_all_checkpoints.py
├── generate_utils/
│   ├── convert_results_to_pool.py
│   └── convert_eval_with_base_to_pool.py
├── visualize/
│   └── visualize_llh_dynamics.py
└── compare_utils/
    └── compare_eval_results.py
```
