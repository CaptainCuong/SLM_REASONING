# SLM Reasoning

Research project investigating small language model reasoning capabilities through log-likelihood-based sample selection and training dynamics analysis.

## Overview

This project explores how different training data selections (based on log-likelihood) affect small language model performance on mathematical reasoning tasks. We train models on MATH12K samples selected by highest, lowest, and random log-likelihood scores, then track training dynamics across checkpoints.

## Project Structure

```
SLM_REASONING/
├── data/                          # Training datasets
│   ├── dataset_info.json         # Dataset registry
│   └── pool.json                 # Sample pool for analysis
├── train/                         # Training configurations
│   ├── train_high.yaml           # High log-likelihood samples
│   ├── train_low.yaml            # Low log-likelihood samples
│   └── train_rand.yaml           # Random samples
├── eval/                          # Evaluation scripts
│   └── eval_GRAPE.sh             # Main evaluation script
├── utils/                         # Utility modules
│   ├── dataset/                  # Dataset processing
│   │   └── generate_solutions.py # Solution generation with vLLM
│   └── prob_processing/          # Probability metrics
│       └── metrics.py            # Log-likelihood, perplexity, entropy
└── prob_tracking/                # Training dynamics tracking
    ├── data/
    │   └── pool.json             # Sample pool with different types
    ├── dynamics_utils/           # Checkpoint evaluation tools
    │   ├── calculate_llh.py      # Single checkpoint metrics
    │   └── evaluate_all_checkpoints.py  # Multi-checkpoint analysis
    └── results/                  # Evaluation outputs
```

## Training

We have 3 training configurations using MATH12K samples selected by different criteria:

### Training Commands

```bash
# Train on high log-likelihood samples
llamafactory-cli train train/train_high.yaml

# Train on low log-likelihood samples
llamafactory-cli train train/train_low.yaml

# Train on random samples
llamafactory-cli train train/train_rand.yaml
```

### Training Configuration

Each YAML file configures:
- Base model: Qwen2.5-Math-7B
- Dataset selection strategy (high/low/random log-likelihood)
- Training hyperparameters
- Checkpoint saving frequency

## Evaluation

### Quick Evaluation

Evaluate all datasets across all checkpoints:

```bash
bash eval/eval_GRAPE.sh
```

### Generate Solutions

Generate solutions for custom datasets using vLLM:

```bash
python utils/dataset/generate_solutions.py \
  --input_file data/test.json \
  --model_name Qwen/Qwen2.5-Math-7B \
  --temperature 0 \
  --max_tokens 32768 \
  --n_sampling 1 \
  --p 0.9
```

**Parameters:**
- `--input_file`: JSON file with questions
- `--model_name`: HuggingFace model name or local path
- `--temperature`: Sampling temperature (0 for greedy decoding)
- `--max_tokens`: Maximum generation length
- `--n_sampling`: Number of solutions per question
- `--p`: Top-p (nucleus sampling) parameter
- `--batch_size`: Limit number of samples to process

## Log-Likelihood Dynamics Tracking

Track how model metrics evolve across training checkpoints.

### 1. Create Sample Pool

Create a pool of diverse samples in `prob_tracking/data/pool.json`:

```json
[
  {
    "instruction": "Please reason step by step...",
    "input": "Question text here",
    "output": "Solution text here",
    "type": "high_llh_train"
  },
  ...
]
```

**Sample Types:**
- `high_llh_train`: High log-likelihood training samples
- `low_llh_train`: Low log-likelihood training samples
- `rand_llh_train1-5`: Random samples
- `student_answer_*`: Model-generated answers
- `paraphrase_gemma`: Paraphrased solutions

### 2. Evaluate Single Checkpoint

Calculate metrics for one checkpoint:

```bash
python prob_tracking/dynamics_utils/calculate_llh.py \
  --checkpoint_path /path/to/model/checkpoint-555/ \
  --pool_path prob_tracking/data/pool.json \
  --output_path prob_tracking/results/checkpoint-555_metrics.json
```

**Output:** JSON file with `perplexity`, `log_likelihood`, and `token_count` for each sample.

### 3. Evaluate All Checkpoints

Process all checkpoints in a model folder:

```bash
python prob_tracking/dynamics_utils/evaluate_all_checkpoints.py \
  --model_folder /helios-storage/helios3-data/cuong/model/Qwen_Math_high \
  --pool_path prob_tracking/data/pool.json
```

**Output:** Aggregated JSON with lists of metrics across training:

```json
{
  "steps": [100, 200, 300, ...],
  "results_by_type": {
    "high_llh_train": {
      "avg_perplexity": [1.25, 1.15, 1.10, ...],
      "avg_log_likelihood": [-11.4, -10.7, -10.2, ...]
    }
  }
}
```

## Metrics

- **Perplexity**: exp(-avg_log_likelihood). Lower = more confident
- **Log-Likelihood**: Sum of log probabilities. Higher = better
- **Token Count**: Number of tokens in response
- **Entropy**: Uncertainty in predictions. Lower = more confident