set -e
# set -x

source ~/.bashrc
conda activate metamath
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Dataset selection
export DATASET=${data:-"metamath"}  # Default to metamath
dp=${dp:-100}  # Data percentage
ep=${ep:-3}  # Epochs

# Batch size configuration
case $DATASET in
    "metamath"|"code")
        total_batch_size=128  # Same as original math setup
        ;;
    "gsm8k")
        total_batch_size=64
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

# Data path handling
case $DATASET in
    "metamath")
        export DATA_PATH="data/MetaMathQA/MetaMathQA-395K.json"
        [ ! -f "$DATA_PATH" ] && echo "Downloading MetaMathQA..." && \
            git clone https://huggingface.co/datasets/meta-math/MetaMathQA $(dirname $DATA_PATH)
        ;;
    "gsm8k")
        export DATA_PATH="data/gsm8k/train.jsonl"
        [ ! -f "$DATA_PATH" ] && echo "Downloading GSM8K..." && \
            mkdir -p $(dirname $DATA_PATH) && \
            wget -q https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -O $DATA_PATH
        ;;
    "code")
        export DATA_PATH="data/codealpaca_20k.json"
        [ ! -f "$DATA_PATH" ] && echo "Downloading CodeAlpaca..." && \
            mkdir -p $(dirname $DATA_PATH) && \
            wget -q https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json -O $DATA_PATH
        ;;
esac

# Model configuration
export MODEL_PATH="meta-llama/Llama-2-7b-hf"
base_save_dir=${bsd:-"checkpoints"}
export SAVE_PATH="$base_save_dir/$DATASET/llama-2-7b-ft"

# Port and network setup
export MASTER_ADDR="localhost"
export MASTER_PORT=$(expr 10000 + $(od -An -N2 -i /dev/urandom) % 10000)
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

# Layer tuning
tune_layers=${l:-"all"}
if [ "$tune_layers" != "all" ]; then
    SAVE_PATH="${SAVE_PATH}_l${tune_layers}"
fi
SAVE_PATH="${SAVE_PATH}_ep${ep}"
[ "$dp" != "100" ] && SAVE_PATH="${SAVE_PATH}dp${dp}"

# GPU setup
gpu_ids=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
num_gpus=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES=$gpu_ids

# Batch calculation (preserve original logic)
per_device_train_batch_size=2
gradient_accumulation_steps=$((total_batch_size / per_device_train_batch_size / num_gpus))
echo "GPUs: $gpu_ids (${num_gpus}), GAS: $gradient_accumulation_steps"

# Training command
eval_only=${eval_only:-false}
if [ "$eval_only" != "1" ]; then
    ddp_cmd="torchrun --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nproc_per_node=$num_gpus --nnodes=1"
    py_cmd="python"

    args=(
        "train.py"
        "--model_name_or_path $MODEL_PATH"
        "--data_path $DATA_PATH"
        "--data_percent $dp"
        "--bf16 True"
        "--output_dir $SAVE_PATH"
        "--num_train_epochs $ep"
        "--tune_layers $tune_layers"
        "--per_device_train_batch_size $per_device_train_batch_size"
        "--gradient_accumulation_steps $gradient_accumulation_steps"
        "--learning_rate 2e-5"
        "--weight_decay 0."
        "--warmup_ratio 0.03"
        "--lr_scheduler_type cosine"
        "--logging_steps 3"
        "--fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer"
        "--tf32 True"
    )

    # Dataset-specific args
    if [ "$DATASET" = "code" ]; then
        args+=("--code_context_length 4096")
    fi

    # Execution
    if [ $num_gpus -gt 1 ]; then
        $ddp_cmd "${args[@]}" --fsdp "full_shard auto_wrap"
    else
        $py_cmd "${args[@]}"
    fi
fi

# Evaluation (preserve original flow)
case $DATASET in
    "metamath"|"gsm8k")
        python eval_gsm8k.py --model $SAVE_PATH --data_file ./data/test/GSM8K_test.jsonl
        python eval_math.py --model $SAVE_PATH --data_file ./data/test/MATH_test.jsonl
        ;;
    "code")
        python eval_code.sh --model $SAVE_PATH
        ;;
esac