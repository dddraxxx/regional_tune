#!/bin/bash
set -e

source ~/.bashrc
conda activate metamath
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

export MODEL_PATH="meta-llama/Llama-2-7b-hf"
export SAVE_PATH=${s:-$MODEL_PATH}
export MASTER_ADDR="localhost"
export MASTER_PORT=$(expr 10000 + $(od -An -N2 -i /dev/urandom) % 10000)
export WANDB_DISABLED=true
wandb offline

# Add data download logic
DATA_FILE="./data/test/HumanEval.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo "Downloading HumanEval test data..."
    mkdir -p $(dirname "$DATA_FILE")
    wget -q https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O "${DATA_FILE}.gz"
    gunzip "${DATA_FILE}.gz"
fi

# HumanEval evaluation with default parameters
python eval_humaneval.py --model $SAVE_PATH \
    --data_file "$DATA_FILE" \
    --batch_size 20 \
    --tensor_parallel_size 8