#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
# 原 ts-tg200：/home/yaowenxuan/ff/pretrain/minicpm
readonly WORK_DIR="${WORK_DIR:-${_SCRIPT_DIR}}"
# 原 ts-tg200：/home/yaowenxuan/ff/tg200/ixmegatron-speedup-kernel
readonly MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/Megatron-LM-dgc-corex}"
# 原 ts-tg200：/home/yaowenxuan/ff/tg200/dgc-ops
readonly DGC_OPS_PATH="${DGC_OPS_PATH:-${REPO_ROOT}/dgc-ops}"
readonly PRETRAIN_FILE="${PRETRAIN_FILE:-${_SCRIPT_DIR}/pretrain_gpt_ixmegatron.py}"
readonly DATA_PATH="/data-aisoft/Dataset/minicpm_preprocessed_text_document"
readonly TOKENIZER_PATH="/data-aisoft/zenghua/models/minicpm5.16a3.v0314"
readonly IXSMI="/usr/local/corex/bin/ixsmi"

readonly SELECTED_GPUS="6,7,8,9,10,11,12,13"
readonly EXPECTED_TOTAL_GPUS=14
readonly EXPECTED_BDFS="00000000:b6:00.0,00000000:b9:00.0,00000000:c4:00.0,00000000:c7:00.0,00000000:d2:00.0,00000000:d5:00.0,00000000:e0:00.0,00000000:e3:00.0"

readonly GPUS_PER_NODE=8
readonly TP=1
readonly PP=4
readonly CP=1
readonly DP=2
readonly SEQ_LENGTH="${SEQ_LENGTH:-64}"
readonly TRAIN_ITERS="${TRAIN_ITERS:-10}"
readonly GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
readonly MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
readonly MASTER_PORT="${MASTER_PORT:-6000}"

readonly PROFILE_ENABLED="${PROFILE_ENABLED:-1}"
readonly PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
readonly PROFILE_STEP_END="${PROFILE_STEP_END:-10}"
readonly DGC_DENSITY="${DGC_DENSITY:-0.001}"
readonly DGC_MOMENTUM="${DGC_MOMENTUM:-0.9}"
readonly DGC_MIN_NUMEL="${DGC_MIN_NUMEL:-16384}"
readonly DGC_IMPL="${DGC_IMPL:-corex}"
readonly -a PROFILE_RANKS=(0 1 2 3 4 5 6 7)

readonly RUN_ID="$(date +%Y%m%d_%H%M%S)"
readonly RUN_DIR="${WORK_DIR}/runs/${RUN_ID}-dgc-${DGC_IMPL}-density${DGC_DENSITY}"
readonly PROFILE_DIR="${RUN_DIR}/profile"
readonly DATA_CACHE_DIR="${RUN_DIR}/data_cache"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    [[ -r "$1" ]] || fail "Required file is not readable: $1"
}

require_file "$PRETRAIN_FILE"
require_file "${MEGATRON_PATH}/megatron/__init__.py"
require_file "${DATA_PATH}.bin"
require_file "${DATA_PATH}.idx"
require_file "${TOKENIZER_PATH}/tokenizer.json"
[[ -x "$IXSMI" ]] || fail "ixsmi is unavailable: $IXSMI"
command -v torchrun >/dev/null 2>&1 \
    || fail "torchrun is unavailable; run inside the corex:1.2.0 container"

[[ "$TRAIN_ITERS" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_ITERS must be positive"
[[ "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "GLOBAL_BATCH_SIZE must be positive"
[[ "$SEQ_LENGTH" =~ ^[1-9][0-9]*$ ]] || fail "SEQ_LENGTH must be positive"
[[ "$PROFILE_ENABLED" == 0 || "$PROFILE_ENABLED" == 1 ]] \
    || fail "PROFILE_ENABLED must be 0 or 1"
if [[ "$PROFILE_ENABLED" == 1 ]]; then
    (( PROFILE_STEP_START < PROFILE_STEP_END )) \
        || fail "PROFILE_STEP_START must be smaller than PROFILE_STEP_END"
    (( PROFILE_STEP_END <= TRAIN_ITERS )) \
        || fail "PROFILE_STEP_END must not exceed TRAIN_ITERS"
fi

export PYTHONPATH="${DGC_OPS_PATH}:${MEGATRON_PATH}:${PYTHONPATH:-}"
loaded_megatron="$(python3 -c 'import importlib.util; print(importlib.util.find_spec("megatron").origin)')"
[[ "$loaded_megatron" == "${MEGATRON_PATH}/megatron/__init__.py" ]] \
    || fail "Expected ${MEGATRON_PATH}, resolved ${loaded_megatron}"
python3 -c 'import torch; import dgc_ops_corex' \
    || fail "dgc_ops_corex is not importable from ${DGC_OPS_PATH}"

actual_gpu_count="$($IXSMI -L | grep -c '^GPU')"
[[ "$actual_gpu_count" -eq "$EXPECTED_TOTAL_GPUS" ]] \
    || fail "Expected ${EXPECTED_TOTAL_GPUS} GPUs, found ${actual_gpu_count}"

actual_bdfs="$($IXSMI --query-gpu=index,pci.bus_id --format=csv,noheader,nounits \
    | awk -F',' '$1 + 0 >= 6 && $1 + 0 <= 13 {gsub(/[[:space:]]/, "", $2); print tolower($2)}' \
    | paste -sd, -)"
[[ "$actual_bdfs" == "$EXPECTED_BDFS" ]] \
    || fail "Selected GPU topology changed: ${actual_bdfs}"

mkdir -p "$RUN_DIR" "$PROFILE_DIR" "$DATA_CACHE_DIR"

export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_DEBUG=INFO
export NCCL_CROSS_NIC=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVB
export NCCL_IB_GDR_LEVEL=PXB
export NCCL_PXN_DISABLE=1

cat > "${RUN_DIR}/run_config.txt" <<EOF
run_id=${RUN_ID}
selected_host_gpus=${SELECTED_GPUS}
selected_pci_bdfs=${EXPECTED_BDFS}
world_size=${GPUS_PER_NODE}
tp=${TP}
pp=${PP}
cp=${CP}
dp=${DP}
sequence_length=${SEQ_LENGTH}
train_iters=${TRAIN_ITERS}
global_batch_size=${GLOBAL_BATCH_SIZE}
data_path=${DATA_PATH}
tokenizer_path=${TOKENIZER_PATH}
megatron_loaded_from=${loaded_megatron}
distributed_optimizer=false
dgc_enabled=true
dgc_density=${DGC_DENSITY}
dgc_momentum=${DGC_MOMENTUM}
dgc_min_numel=${DGC_MIN_NUMEL}
dgc_impl=${DGC_IMPL}
profile_enabled=${PROFILE_ENABLED}
profile_schedule_start=${PROFILE_STEP_START}
profile_schedule_end=${PROFILE_STEP_END}
profile_dir=${PROFILE_DIR}
EOF

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes 1
    --node_rank 0
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --num-layers 16
    --hidden-size 2560
    --ffn-hidden-size 9728
    --num-attention-heads 32
    --kv-channels 128
    --qk-layernorm
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings "$SEQ_LENGTH"
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --attention-softmax-in-fp32
    --make-vocab-size-divisible-by 128
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

TRAINING_ARGS=(
    --train-iters "$TRAIN_ITERS"
    --seq-length "$SEQ_LENGTH"
    --micro-batch-size 1
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP"
    --sequence-parallel
    --use-flash-attn
    --transformer-impl transformer_engine
    --ckpt-format torch
    --bf16
)

OPTIMIZER_ARGS=(
    --lr 1e-4
    --lr-decay-style cosine
    --min-lr 1e-5
    --weight-decay 1e-1
    --lr-warmup-fraction 0.2
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.01
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --initial-loss-scale 4096
)

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_DIR"
    --split 100,0,0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_PATH"
    --no-load-optim
    --no-load-rng
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval "$TRAIN_ITERS"
    --eval-interval "$TRAIN_ITERS"
    --eval-iters 0
)

DGC_ARGS=(
    --dgc-enabled
    --dgc-density "$DGC_DENSITY"
    --dgc-momentum "$DGC_MOMENTUM"
    --dgc-min-numel-to-compress "$DGC_MIN_NUMEL"
    --dgc-impl "$DGC_IMPL"
)

PROFILE_ARGS=()
if [[ "$PROFILE_ENABLED" == 1 ]]; then
    PROFILE_ARGS=(
        --profile
        --use-pytorch-profiler
        --profile-step-start "$PROFILE_STEP_START"
        --profile-step-end "$PROFILE_STEP_END"
        --profile-ranks "${PROFILE_RANKS[@]}"
        --tensorboard-dir "$PROFILE_DIR"
    )
fi

echo "Starting MiniCPM run ${RUN_ID}"
echo "GPUs=${SELECTED_GPUS}; TP=${TP}, PP=${PP}, CP=${CP}, DP=${DP}"
echo "Megatron=${loaded_megatron}"
echo "DGC: impl=${DGC_IMPL}, density=${DGC_DENSITY}, momentum=${DGC_MOMENTUM}, min_numel=${DGC_MIN_NUMEL}"
echo "seq=${SEQ_LENGTH}; global batch=${GLOBAL_BATCH_SIZE}; iterations=${TRAIN_ITERS}"
echo "log=${RUN_DIR}/train.log"

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$PRETRAIN_FILE" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DGC_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/train.log"
