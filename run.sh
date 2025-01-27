set -e
# set -x

source ~/.bashrc
conda activate metamath
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

export DATA_PATH="data/MetaMathQA/MetaMathQA-395K.json"
# download from "git clone https://huggingface.co/datasets/meta-math/MetaMathQA" to $DATA_PATH
if [ ! -f "$DATA_PATH" ]; then
    echo "Data path not found, downloading from Hugging Face..."
    PARENT_DIR=$(dirname "$DATA_PATH")  # Get the parent directory of DATA_PATH
    git clone https://huggingface.co/datasets/meta-math/MetaMathQA "$PARENT_DIR"
fi

export MODEL_PATH="meta-llama/Llama-2-7b-hf"
export SAVE_PATH='checkpoints/llama-2-7b-ft'
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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 train_math.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --tune_layers $tune_layers \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 3 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True

python eval_gsm8k.py --model $SAVE_PATH --data_path ./data/test/GSM8K_test.jsonl
python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl
