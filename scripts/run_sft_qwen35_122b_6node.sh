#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 6 nodes × 8 GPU (= 48 × 80G), long context, NO MTP.
#
# Recipe: qwen35_vl_122b_a10b_sft_config
#
# Layout (48 GPU): TP=2  PP=6  EP=4  DP=4
#   world_size = 48; DP = 48/(TP×PP) = 48/(2×6) = 4; EP ≤ DP → EP=4 ✓
#
# Layer balance:
#   48 layers = [G,G,G,F] × 12  (full_attention_interval=4, see qwen35_vl_bridge.py:134)
#   PP=6 → 8 layers/stage = 2 complete [G,G,G,F] groups → 6G + 2F per stage. ✓ Perfect.
#
# MTP: disabled (model.mtp_num_layers=0). To enable MTP, switch to the 16-node
# script (PP=4 → only ~8% last-stage imbalance; PP=6's 12.5% imbalance is workable
# but tight under SEQ=32k).
#
# Memory per GPU (TP=2 PP=6 EP=4 DP=4, SEQ=32k, MBS=1, hidden≈4096, full block
# recompute, Adam on CPU):
#   params ~6 / grads ~6 / activations (8 layers) ~10-12 / attn live ~2 /
#   step transients 2-4  →  total ~26-30 / 80 GB.  Comfortable.
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_6node.sh
#
# Override knobs (env): TP PP EP ITERS SEQ GBS MBS RECOMPUTE_NUM_LAYERS ...

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
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_6node}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_6node_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-6}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

TP="${TP:-2}"
PP="${PP:-6}"
EP="${EP:-4}"

RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-32768}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-8}"   # = layers/stage at PP=6

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

    # No MTP for PP=6 (last-stage relative cost 12.5%; tighter than PP=4).
    model.mtp_num_layers=0

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
Qwen3.5-122B-A10B FULL SFT (6-node, 32k, no MTP)
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  EP=${EP}   DP=$((NNODES * NPROC / TP / PP))
  Layer balance: 8 layers/stage = 2 × [G,G,G,F]  → 6G+2F per stage  ✓
  MTP          : 0
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
