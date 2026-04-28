#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT + MTP — 16 nodes × 8 GPU (= 128 × 80G), 32k context.
#
# Recipe: qwen35_vl_122b_a10b_sft_config
#
# Layout (128 GPU): TP=2  PP=4  EP=8  DP=16
#   world_size = 128; DP = 128/(TP×PP) = 128/(2×4) = 16; EP ≤ DP → EP=8 ✓
#
# Why this layout (vs 12-node):
#   - **PP=4 chosen specifically to keep MTP affordable.**
#     Layer pattern [G,G,G,F]×12. PP=4 → 12 layers/stage = 3 complete groups
#     = 9G + 3F. Layer-balanced ✓.
#   - MTP adds 1 transformer block + LM head ONLY on the last PP stage.
#     Relative cost on that stage = 1 / layers_per_stage:
#       PP=4  → 1/12 = 8%   ← acceptable
#       PP=6  → 1/8  = 12.5% ← borderline at 32k
#       PP=12 → 1/4  = 25%  ← unusable
#   - EP=8 leverages all 256 experts (32 experts/EP rank), better load balance.
#   - DP=16 gives strong throughput scaling.
#
# MTP cost (params/grads/optim only on last PP stage):
#   ~244 GB / 48 layers ≈ 5 GB raw per layer. After TP=2 sharding ≈ 2.5 GB
#   bf16 params + 2.5 GB bf16 grads. Adam fp32 stays on CPU.
#   On last-stage GPUs the extra burden is ~5-7 GB total — fits in 80G.
#
# Memory per GPU (TP=2 PP=4 EP=8 DP=16, SEQ=32k, MBS=1, hidden≈4096, full
# block recompute, Adam on CPU):
#   non-last stages: params ~5 / grads ~5 / activations (12 layers) ~14 /
#                    attn live ~2 / step transients 2-4 →  ~28-30 / 80 GB
#   last stage:     +5-7 GB (MTP block + LM head) →   ~33-37 / 80 GB
#
# CP: still 1 (GDN constraint).
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_16node_mtp.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEG_RUN_DIR="${MEG_RUN_DIR:-$(dirname "$REPO_ROOT")/meg-run}"

if [[ -z "${HF_MODEL:-}" ]]; then
    echo "[ERROR] HF_MODEL is not set." >&2; exit 1
fi
if [[ -z "${MCORE_PATH:-}" ]]; then
    echo "[ERROR] MCORE_PATH is not set." >&2; exit 1
fi

TRAIN_DATA="${TRAIN_DATA:-${MEG_RUN_DIR}/demo_data/train_data_demo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_16node_mtp}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_16node_mtp_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-16}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

TP="${TP:-2}"
PP="${PP:-4}"
EP="${EP:-8}"

RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-32768}"
GBS="${GBS:-64}"
MBS="${MBS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-12}"   # = layers/stage at PP=4
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"

[[ -d "$REPO_ROOT" ]]                          || { echo "[ERROR] missing REPO_ROOT" >&2; exit 1; }
[[ -d "$MCORE_PATH/iter_0000000" ]]            || { echo "[ERROR] missing mcore ckpt" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]                         || { echo "[ERROR] missing train data" >&2; exit 1; }
[[ -d "$HF_MODEL" ]]                           || { echo "[ERROR] missing HF model" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
cd "$REPO_ROOT"

VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
[[ -x "$VENV_PY" ]] || { echo "[ERROR] venv python missing: $VENV_PY" >&2; exit 1; }

echo "[run_sft] node ${NODE_RANK}: running install_runtime_deps.sh..."
bash "$REPO_ROOT/scripts/install_runtime_deps.sh"

export MBRIDGE_PATCH_NVIDIA_FILE="${MBRIDGE_PATCH_NVIDIA_FILE:-0}"
export MBRIDGE_DISABLE_CUDNN="${MBRIDGE_DISABLE_CUDNN:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MBRIDGE_PATCH_GRAD_FUSION="${MBRIDGE_PATCH_GRAD_FUSION:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export HF_HOME="${HF_HOME:-${MEG_RUN_DIR}/hf_cache}"
mkdir -p "$HF_HOME"

OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    --step_func vlm_step
    --hf_path "$HF_MODEL"

    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.expert_model_parallel_size="$EP"

    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers="$RECOMPUTE_NUM_LAYERS"

    # MTP enabled — recipe default is 1; expose env knob in case we want 2.
    model.mtp_num_layers="$MTP_NUM_LAYERS"

    dataset.pack_sequences_in_batch=False

    optimizer.optimizer_cpu_offload="${OPTIM_OFFLOAD:-True}"
    optimizer.optimizer_offload_fraction="${OPTIM_OFFLOAD_FRAC:-1.0}"
    optimizer.overlap_cpu_optimizer_d2h_h2d="${OPTIM_OFFLOAD_OVERLAP:-True}"
    optimizer.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPT:-True}"

    train.train_iters="$ITERS"
    train.global_batch_size="$GBS"
    train.micro_batch_size="$MBS"
    train.eval_iters=0
    train.eval_interval=1000000

    checkpoint.pretrained_checkpoint="$MCORE_PATH"
    checkpoint.save="$OUTPUT_DIR"
    checkpoint.save_interval="$SAVE_INTERVAL"
    checkpoint.load=null

    dataset.train_data_path="$TRAIN_DATA"
    dataset.hf_processor_path="$HF_MODEL"
    dataset.seq_length="$SEQ"

    logger.log_interval="$LOG_INTERVAL"
    logger.tensorboard_dir=null
    logger.wandb_project=null
)

cat <<EOF
============================================================
Qwen3.5-122B-A10B FULL SFT + MTP (16-node, 32k)
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  EP=${EP}   DP=$((NNODES * NPROC / TP / PP))
  Layer balance: 12 layers/stage = 3 × [G,G,G,F]  → 9G+3F per stage  ✓
  MTP          : ${MTP_NUM_LAYERS}   (last-stage relative cost ~$((100 / RECOMPUTE_NUM_LAYERS))%)
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}    SEQ=${SEQ}
  Recompute    : full / block / ${RECOMPUTE_NUM_LAYERS}
  Log file     : $LOG_FILE
============================================================
EOF

RDZV_TIMEOUT="${RDZV_TIMEOUT:-1800}"
"$VENV_PY" -u -m torch.distributed.run \
    --nproc_per_node="$NPROC" --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --rdzv_conf "timeout=${RDZV_TIMEOUT}" \
    scripts/training/run_recipe.py "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"
