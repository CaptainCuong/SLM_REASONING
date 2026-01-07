#!/bin/bash
# Wrapper script to run LlamaFactory training with correct environment

# Source conda
source /workspace/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate llama310

# Run training with provided config file
if [ -z "$1" ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 train/train_high.yaml"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

echo "========================================"
echo "Running LlamaFactory Training"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Environment: llama310"
echo "Deepspeed: 0.16.9"
echo "========================================"
echo ""

# Run training
llamafactory-cli train "$CONFIG_FILE"
