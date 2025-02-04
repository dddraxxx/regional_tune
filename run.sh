set -e
# set -x

source ~/.bashrc
conda activate metamath
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Add dataset choice
export DATASET=${data:-"metamath"}  # Default to metamath
dp=${dp:-100}  # Default to 100% of data
train_py=${train_py:-"train_math.py"}
if [ "$DATASET" = "med" ]; then
    train_py="train_med.py"
fi

# total_batch_size=128
case $DATASET in
    "metamath")
        export DATA_PATH="data/MetaMathQA/MetaMathQA-395K.json"
        if [ ! -f "$DATA_PATH" ]; then
            echo "MetaMathQA data not found, downloading from Hugging Face..."
            PARENT_DIR=$(dirname "$DATA_PATH")
            git clone https://huggingface.co/datasets/meta-math/MetaMathQA "$PARENT_DIR"
        fi
        total_batch_size=128
        ep=${ep:-3}
        ;;
    "med")
        export DATA_PATH="data/medical_meadow/medical_meadow_medqa.json"
        if [ ! -f "$DATA_PATH" ]; then
            echo "Medical MeadowQA data not found, downloading from Hugging Face..."
            PARENT_DIR=$(dirname "$DATA_PATH")
            mkdir -p "$PARENT_DIR"
            git clone https://huggingface.co/datasets/medalpaca/medical_meadow_medqa "$PARENT_DIR"
        fi
        total_batch_size=64  # Using a smaller batch size like GSM8K since medical data might be complex
        ep=${ep:-3}
        ;;
    "gsm8k")
        export DATA_PATH="data/gsm8k/train.jsonl"
        if [ ! -f "$DATA_PATH" ]; then
            echo "GSM8K data not found, downloading..."
            PARENT_DIR=$(dirname "$DATA_PATH")
            mkdir -p "$PARENT_DIR"
            wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -O "$DATA_PATH"
        fi
        total_batch_size=64
        ep=${ep:-3}
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

export MODEL_PATH="meta-llama/Llama-2-7b-hf"
base_save_dir=${bsd:-"checkpoints"}
export SAVE_PATH=$base_save_dir'/'$DATASET'/llama-2-7b-ft'
export MASTER_ADDR="localhost"
# random port
export MASTER_PORT=$(expr 10000 + $(od -An -N2 -i /dev/urandom) % 10000)
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
# export WANDB_DISABLED=true
# wandb offline

# tune layers
tune_layers=${l:-"all"}
if [ "$tune_layers" != "all" ]; then
    SAVE_PATH="${SAVE_PATH}_l${tune_layers}"
fi
SAVE_PATH="${SAVE_PATH}_ep${ep}"
if [ "$dp" != "100" ]; then
    SAVE_PATH="${SAVE_PATH}dp${dp}"
fi

per_device_train_batch_size=2

gpu_ids=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
num_gpus=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES=${gpu_ids}
gradient_accumulation_steps=$((total_batch_size / per_device_train_batch_size / num_gpus))
echo "using cuda device: $gpu_ids"
echo "number of GPUs: $num_gpus"
echo "gradient accumulation steps: $gradient_accumulation_steps"

eval_only=${eval_only:-false}
if [ "$eval_only" != "1" ]; then
    ddp_cmd="torchrun --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=${num_gpus} --nnodes=1"
    py_cmd="python"
    args=("${train_py} \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --data_length 10000000 \
        --data_percent $dp \
        --bf16 True \
        --output_dir $SAVE_PATH \
        --num_train_epochs $ep \
        --tune_layers $tune_layers \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --evaluation_strategy no \
        --save_strategy no \
        --save_steps 1000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 3 \
        --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
        --tf32 True")
    if [ $num_gpus -gt 1 ]; then
        echo "run: ${ddp_cmd} ${args}"
        ${ddp_cmd} ${args} --fsdp "full_shard auto_wrap"
    else
        ${py_cmd} ${args}
    fi
elif [ "$eval_only" = "1" ]; then
    SAVE_PATH=$MODEL_PATH
fi

if [ "$DATASET" = "gsm8k" ] || [ "$DATASET" = "metamath" ]; then
    python eval_gsm8k.py --model $SAVE_PATH --data_file ./data/test/GSM8K_test.jsonl
    python eval_math.py --model $SAVE_PATH --data_file ./data/test/MATH_test.jsonl
elif [ "$DATASET" = "med" ]; then
    # Check and download medical test data if needed
    TEST_DATA_PATH="./data/test/medqa_test.jsonl"
    if [ ! -f "$TEST_DATA_PATH" ]; then
        echo "Medical test data not found, downloading from Hugging Face..."
        mkdir -p $(dirname "$TEST_DATA_PATH")
        python build_medqa_jsonl.py
    fi
    echo "Evaluating model on $TEST_DATA_PATH"
    python eval_med.py --model $SAVE_PATH --data_file "$TEST_DATA_PATH"
fi
