set -e
# set -x

source ~/.bashrc
conda activate metamath
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

export MODEL_PATH="meta-llama/Llama-2-7b-hf"
export SAVE_PATH=${s:-'checkpoints/llama-2-7b-ft'}
export MASTER_ADDR="localhost"
# random port
export MASTER_PORT=$(expr 10000 + $(od -An -N2 -i /dev/urandom) % 10000)
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true
wandb offline

# tune layers
tune_layers=${l:-"all"}
if [ "$tune_layers" != "all" ]; then
    SAVE_PATH="${SAVE_PATH}_l${tune_layers}"
fi

total_batch_size=128
per_device_train_batch_size=2
gradient_accumulation_steps=$((total_batch_size / per_device_train_batch_size))

python eval_gsm8k.py --model $SAVE_PATH --data_file ./data/test/GSM8K_test.jsonl
python eval_math.py --model $SAVE_PATH --data_file ./data/test/MATH_test.jsonl
