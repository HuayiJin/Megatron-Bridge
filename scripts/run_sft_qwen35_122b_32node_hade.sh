#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 32 nodes × 8 GPU (= 256 × H800 80G).
#
# This script intentionally stays aligned with the verified 24-node bring-up
# script and only changes the target node count/context-parallel layout for
# 96K context.
#
# IMPORTANT — parallelism on 256 GPU with 96K context:
#   Use TP=2, PP=16, CP=4, EP=2, DP=2.
#   Math: world_size=256, DP=256/(TP×PP×CP)=256/(2×16×4)=2.
#   EP must ≤ DP → EP=2 ≤ DP=2 ✓.
#   CP=4 shards 96K to 24K local context per attention rank, which is below
#   the verified 24-node local context of 64K/CP=2=32K.
#
#   PP=16 intentionally gives 3 layers/stage and does not preserve the 4-layer
#   GDN cycle. This is a fit-first bring-up setting after PP=4 OOMed on stage 0.
#
# Keep the same memory-saving settings as the verified 24-node baseline:
#   1. freeze_vision_model=True, freeze_vision_projection=True
#   2. full block activation recompute over each pipeline stage
#   3. optimizer CPU offload
#   4. MTP disabled
#   5. fused cross entropy disabled under CP
#   6. FLA autotune disabled
#
# Required env (per-node):
#   MASTER_ADDR, MASTER_PORT  cluster-injected
#   RANK or NODE_RANK         this node's index in [0, NNODES)
#   WORLD_SIZE                number of nodes (default 32)
#
# Required user-set env:
#   HF_MODEL    path to HuggingFace model directory
#   MCORE_PATH  path to converted Megatron-Core checkpoint directory
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_32node_hade.sh
#
# Override knobs:
#   TP=2 PP=16 CP=4 EP=2 ITERS=20 SEQ=98304 GBS=32 MBS=1

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

TRAIN_DATA="${TRAIN_DATA:-${MEG_RUN_DIR}/demo_data/total.v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_32node_seq98304}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs_32node}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_32node_seq98304_$(date +%Y%m%d_%H%M)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-32}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Parallelism (256 GPU / 96K layout — see header for math)
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-16}"
CP="${CP:-4}"
EP="${EP:-2}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-98304}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"

VALID_SPLIT_RATIO="${VALID_SPLIT_RATIO:-0.05}"
VALID_SPLIT_SEED="${VALID_SPLIT_SEED:-1234}"
EVAL_ITERS="${EVAL_ITERS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

OPTIM_OFFLOAD="${OPTIM_OFFLOAD:-True}"
OPTIM_OFFLOAD_FRAC="${OPTIM_OFFLOAD_FRAC:-1.0}"
OPTIM_OFFLOAD_OVERLAP="${OPTIM_OFFLOAD_OVERLAP:-True}"
USE_PRECISION_AWARE_OPT="${USE_PRECISION_AWARE_OPT:-True}"

# PP=16 → 48/16 = 3 layers/stage. This intentionally breaks 4-layer GDN alignment.
PP_STAGE_LAYERS="${PP_STAGE_LAYERS:-$(( 48 / PP ))}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -d "$REPO_ROOT" ]]               || { echo "[ERROR] missing REPO_ROOT: $REPO_ROOT" >&2; exit 1; }
[[ -d "$MCORE_PATH/iter_0000000" ]] || { echo "[ERROR] missing mcore ckpt: $MCORE_PATH/iter_0000000" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]              || { echo "[ERROR] missing train data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$HF_MODEL" ]]                || { echo "[ERROR] missing HF model: $HF_MODEL" >&2; exit 1; }

_world=$(( NNODES * NPROC ))
_dp=$(( _world / TP / PP / CP ))
if (( _world % (TP * PP * CP) != 0 )); then
    echo "[ERROR] world_size=$_world not divisible by TP×PP×CP=$(( TP * PP * CP ))" >&2; exit 1
fi
if (( EP > _dp )); then
    echo "[ERROR] EP=$EP > DP=$_dp — EP must be ≤ DP" >&2; exit 1
fi
if (( SEQ % CP != 0 )); then
    echo "[ERROR] SEQ=$SEQ must be divisible by CP=$CP" >&2; exit 1
fi
if (( 48 % PP != 0 )); then
    echo "[ERROR] PP=$PP does not divide 48 model layers" >&2; exit 1
fi
if (( (48 / PP) % 4 != 0 )); then
    echo "[WARNING] PP=$PP gives $((48/PP)) layers/stage — not a multiple of 4 (GDN cycle broken)." >&2
fi

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
# Install runtime deps (ALL nodes, idempotent)
# ---------------------------------------------------------------------------
echo "[run_sft] node ${NODE_RANK}: running install_runtime_deps.sh..."
bash "$REPO_ROOT/scripts/install_runtime_deps.sh"

# ---------------------------------------------------------------------------
# Environment
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
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export HF_HOME="${HF_HOME:-${MEG_RUN_DIR}/hf_cache}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_RS_NUM_CPUS="${RAYON_RS_NUM_CPUS:-1}"

# ---------------------------------------------------------------------------
# Recipe + dataset overrides
# ---------------------------------------------------------------------------
OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    --step_func qwen3_vl_step
    --hf_path "$HF_MODEL"

    # Parallelism (256-GPU / 96K, PP=16, CP=4; local context = 24K)
    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.context_parallel_size="$CP"
    model.expert_model_parallel_size="$EP"
    model.seq_length="$SEQ"

    dist.distributed_timeout_minutes="${DIST_TIMEOUT_MINUTES:-30}"

    # Activation memory.
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers="$PP_STAGE_LAYERS"

    model.mtp_num_layers=0

    # Vision memory: keep frozen for long-context SFT.
    model.freeze_vision_model=True
    model.freeze_vision_projection=True

    # CP > 1 required flags copied from the verified 24-node path.
    model.calculate_per_token_loss=True
    ddp.average_in_collective=False

    # Disable fused CE under CP; qwen3_vl_step slices labels per CP rank.
    model.cross_entropy_loss_fusion=False

    # GDN constraint (Pitfall #7): no THD support
    dataset.pack_sequences_in_batch=False

    # Optimizer memory.
    optimizer.optimizer_cpu_offload="$OPTIM_OFFLOAD"
    optimizer.optimizer_offload_fraction="$OPTIM_OFFLOAD_FRAC"
    optimizer.overlap_cpu_optimizer_d2h_h2d="$OPTIM_OFFLOAD_OVERLAP"
    optimizer.use_precision_aware_optimizer="$USE_PRECISION_AWARE_OPT"

    # Iters / batch
    train.train_iters="$ITERS"
    train.global_batch_size="$GBS"
    train.micro_batch_size="$MBS"
    validation.eval_iters="$EVAL_ITERS"
    validation.eval_interval="$EVAL_INTERVAL"

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
Qwen3.5-122B-A10B FULL SFT (32-node / 256-GPU, 96K, PP=16, CP=4)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC} = $(( NNODES * NPROC )) GPU   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  CP=${CP}  EP=${EP}   (DP=$(( NNODES * NPROC / TP / PP / CP )))
  Layers/stage : ${PP_STAGE_LAYERS}   (48 total; GDN cycle alignment intentionally not enforced)
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  Local context: $(( SEQ / CP )) per CP rank
  Valid split  : ${VALID_SPLIT_RATIO}
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
