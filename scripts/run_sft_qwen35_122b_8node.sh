#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 8 nodes × 8 GPU (= 64 × H800/L20Y 80G), long context.
#
# Recipe: qwen35_vl_122b_a10b_sft_config
#   default TP=2, PP=6, EP=8, LR=2e-5, GBS=36, seq=2048
#
# IMPORTANT — parallelism on 64 GPU (long-context layout):
#   We deliberately use TP=2  PP=4  EP=4  DP=8 (NOT PP=8).
#
#   Why not PP=8?
#     Layer pattern is [G,G,G,F] x 12 = 48 layers (full_attention_interval=4,
#     see qwen35_vl_bridge.py:134). PP=8 → 6 layers/stage = 1.5 groups, so
#     stages alternate 5G+1F and 4G+2F. Compute per stage differs ~10-15%,
#     enlarging pipeline bubbles. PP=4 → 12 layers/stage = 3 complete groups
#     = 9G + 3F per stage. Perfectly balanced.
#
#   World math:
#     world_size = 64;  DP = 64 / (TP×PP) = 64 / (2×4) = 8;  EP ≤ DP → EP=4 ✓.
#     EP=8 also valid (≤ DP=8) but EP=4 is what we validated on the 4-node
#     run; stick with it.
#
#   For 48-GPU (6-node) the recipe-default PP=6 is even better — also
#   layer-balanced (8 layers/stage = 2 groups = 6G+2F). Use the 6-node
#   sibling script if you have exactly 6 nodes.
#
# Memory estimate per GPU (TP=2 PP=4 EP=4 DP=8, SEQ=32768, MBS=1, hidden≈4096,
#                          full block recompute, Adam on CPU):
#   - Params bf16, sharded               ~5-7 GB
#   - Grads  bf16                        ~5-7 GB
#   - Adam fp32 (CPU offload)             ~0 GB on GPU
#   - Stage activations (12 layers, full block recompute,
#     stores 1 input boundary per block)  ~12-16 GB
#   - GDN/full-attn live tensors          ~2-4 GB
#   - Optimizer step D2H/H2D transients   ~2-4 GB
#   Total ≈ 28-38 GB / 80 GB — comfortable headroom.
#
# Notes on CP and long context (see online_memory.md, 2026-04-28):
#   GDN (fla chunk gated delta rule) accumulates state along the sequence
#   axis; CP > 1 breaks the cross-chunk hand-off. Keep CP=1.
#   Full-attention layers use flash-attn 2.7.4 — memory O(s), compute O(s²).
#   32k @ CP=1 is trainable; do NOT raise SEQ above 32k without re-evaluating
#   full-attn cost.
#
# Container assumptions: same as run_sft_qwen35_122b_4node.sh (NGC 25.06).
#
# Required env (per-node):
#   MASTER_ADDR, MASTER_PORT  cluster-injected
#   RANK or NODE_RANK         this node's index in [0, NNODES)
#   WORLD_SIZE                number of nodes (default 8)
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_8node.sh
#
# Override knobs:
#   TP=2 PP=4 EP=4 ITERS=20 SEQ=32768 GBS=32 MBS=1 ...

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo + paths
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEG_RUN_DIR="${MEG_RUN_DIR:-$(dirname "$REPO_ROOT")/meg-run}"

if [[ -z "${HF_MODEL:-}" ]]; then
    echo "[ERROR] HF_MODEL is not set. Export the path to your HF model directory." >&2
    echo "  export HF_MODEL=/path/to/Qwen3.5-122B-A10B" >&2
    exit 1
fi
if [[ -z "${MCORE_PATH:-}" ]]; then
    echo "[ERROR] MCORE_PATH is not set. Export the path to your Megatron-Core checkpoint." >&2
    echo "  export MCORE_PATH=/path/to/Qwen3.5-122B-A10B-mcore" >&2
    exit 1
fi

TRAIN_DATA="${TRAIN_DATA:-${MEG_RUN_DIR}/demo_data/train_data_demo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_8node}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_8node_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-8}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Parallelism overrides (64 GPU layout — see header for math/why-not-PP=8)
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-4}"
EP="${EP:-4}"

# ---------------------------------------------------------------------------
# Training hyperparameters (long-context defaults)
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-32768}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

# Recompute layers should match layers/stage. PP=4 → 12 layers/stage.
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-12}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -d "$REPO_ROOT" ]]                          || { echo "[ERROR] missing REPO_ROOT: $REPO_ROOT" >&2; exit 1; }
[[ -d "$MCORE_PATH/iter_0000000" ]]            || { echo "[ERROR] missing mcore ckpt: $MCORE_PATH/iter_0000000" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]                         || { echo "[ERROR] missing train data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$HF_MODEL" ]]                           || { echo "[ERROR] missing HF model: $HF_MODEL" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Venv python
# ---------------------------------------------------------------------------
VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] venv python missing: $VENV_PY" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Install runtime deps + wire megatron.bridge → /mnt (ALL nodes; idempotent).
# ---------------------------------------------------------------------------
echo "[run_sft] node ${NODE_RANK}: running install_runtime_deps.sh..."
bash "$REPO_ROOT/scripts/install_runtime_deps.sh"

# ---------------------------------------------------------------------------
# Env (NGC container — patches off; long-context tweaks)
# ---------------------------------------------------------------------------
export MBRIDGE_PATCH_NVIDIA_FILE="${MBRIDGE_PATCH_NVIDIA_FILE:-0}"
export MBRIDGE_DISABLE_CUDNN="${MBRIDGE_DISABLE_CUDNN:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MBRIDGE_PATCH_GRAD_FUSION="${MBRIDGE_PATCH_GRAD_FUSION:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# FLA Triton autotuner OOMs at 32k under tight VRAM (Pitfall #22).
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export HF_HOME="${HF_HOME:-${MEG_RUN_DIR}/hf_cache}"
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# Recipe + dataset overrides
# ---------------------------------------------------------------------------
OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    --step_func vlm_step
    --hf_path "$HF_MODEL"

    # Parallelism (64-GPU layer-balanced layout)
    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.expert_model_parallel_size="$EP"

    # Activation recompute — full block, every layer of every PP stage.
    # PP=4 → 12 layers/stage → recompute_num_layers=12.
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers="$RECOMPUTE_NUM_LAYERS"

    # Disable MTP (Pitfall #19 — last-stage OOM).
    model.mtp_num_layers=0

    # GDN constraint (memory_legacy.md #18): no THD packing; CP=1 implicit.
    dataset.pack_sequences_in_batch=False

    # Optimizer CPU offload — same rationale as 4-node script (Pitfall #28).
    optimizer.optimizer_cpu_offload="${OPTIM_OFFLOAD:-True}"
    optimizer.optimizer_offload_fraction="${OPTIM_OFFLOAD_FRAC:-1.0}"
    optimizer.overlap_cpu_optimizer_d2h_h2d="${OPTIM_OFFLOAD_OVERLAP:-True}"
    optimizer.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPT:-True}"

    # Iters / batch
    train.train_iters="$ITERS"
    train.global_batch_size="$GBS"
    train.micro_batch_size="$MBS"
    train.eval_iters=0
    train.eval_interval=1000000

    # Checkpoint
    checkpoint.pretrained_checkpoint="$MCORE_PATH"
    checkpoint.save="$OUTPUT_DIR"
    checkpoint.save_interval="$SAVE_INTERVAL"
    checkpoint.load=null

    # Dataset
    dataset.train_data_path="$TRAIN_DATA"
    dataset.hf_processor_path="$HF_MODEL"
    dataset.seq_length="$SEQ"

    # Logging
    logger.log_interval="$LOG_INTERVAL"
    logger.tensorboard_dir=null
    logger.wandb_project=null
)

cat <<EOF
============================================================
Qwen3.5-122B-A10B FULL SFT (8-node, long context)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  EP=${EP}   (DP = $((NNODES * NPROC / TP / PP)))
  Layer balance: 12 layers/stage = 3 × [G,G,G,F]  → 9G+3F per stage  (balanced ✓)
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  Recompute    : full / block / ${RECOMPUTE_NUM_LAYERS} layers
  Log file     : $LOG_FILE
  Venv python  : $VENV_PY
============================================================
EOF

RDZV_TIMEOUT="${RDZV_TIMEOUT:-1800}"

"$VENV_PY" -u -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_conf "timeout=${RDZV_TIMEOUT}" \
    scripts/training/run_recipe.py \
    "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"
