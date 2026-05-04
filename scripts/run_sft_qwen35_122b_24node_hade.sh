#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 24 nodes × 8 GPU, 96K context.
# Keep the memory-saving settings below unless there is new measured headroom:
#   - freeze vision encoder/projection
#   - TE chunked cross entropy
#   - full uniform activation recompute with num_layers=1
#   - optimizer CPU offload
#   - MTP disabled
#   - FLA autotune disabled

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEG_RUN_DIR="${MEG_RUN_DIR:-$(dirname "$REPO_ROOT")/meg-run}"

if [[ -z "${HF_MODEL:-}" ]]; then
    echo "[ERROR] HF_MODEL is not set. Export the path to your HF model directory." >&2
    exit 1
fi
if [[ -z "${MCORE_PATH:-}" ]]; then
    echo "[ERROR] MCORE_PATH is not set. Export the path to your Megatron-Core checkpoint." >&2
    exit 1
fi

TRAIN_DATA="${TRAIN_DATA:-${MEG_RUN_DIR}/demo_data/train_data_demo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_24node_seq98304}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_24node_seq98304_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-24}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

TP="${TP:-2}"
PP="${PP:-12}"
CP="${CP:-1}"
EP="${EP:-4}"

RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-98304}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
LR="${LR:-}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-}"
MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-}"
MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-1e-3}"
DROP_OVERLENGTH="${DROP_OVERLENGTH:-False}"
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
DIST_TIMEOUT_MIN="${DIST_TIMEOUT_MIN:-60}"
DIST_TIMEOUT_SEC="${DIST_TIMEOUT_SEC:-3600}"
RDZV_TIMEOUT="${RDZV_TIMEOUT:-1800}"

[[ -d "$REPO_ROOT" ]] || { echo "[ERROR] missing REPO_ROOT: $REPO_ROOT" >&2; exit 1; }
[[ -d "$MCORE_PATH/iter_0000000" ]] || { echo "[ERROR] missing mcore ckpt: $MCORE_PATH/iter_0000000" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]] || { echo "[ERROR] missing train data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$HF_MODEL" ]] || { echo "[ERROR] missing HF model: $HF_MODEL" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
cd "$REPO_ROOT"
exec > >(tee "$LOG_FILE") 2>&1

VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] venv python missing: $VENV_PY" >&2
    exit 1
fi

echo "[run_sft] node ${NODE_RANK}: running install_runtime_deps.sh..."
bash "$REPO_ROOT/scripts/install_runtime_deps.sh"

# Memory and stability settings.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_RS_NUM_CPUS="${RAYON_RS_NUM_CPUS:-1}"
export HF_HOME="${HF_HOME:-${MEG_RUN_DIR}/hf_cache}"
mkdir -p "$HF_HOME"

# NCCL/debug settings retained because this 192-GPU shape has slow first-iter init.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_NCCL_ENABLE_TIMING="${TORCH_NCCL_ENABLE_TIMING:-1}"
export TORCH_NCCL_TRACE_CPP_STACK="${TORCH_NCCL_TRACE_CPP_STACK:-1}"

OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    --step_func qwen3_vl_step
    --hf_path "$HF_MODEL"

    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.context_parallel_size="$CP"
    model.expert_model_parallel_size="$EP"
    model.seq_length="$SEQ"

    dist.distributed_timeout_minutes="$DIST_TIMEOUT_MIN"
    dist.distributed_timeout_seconds_after_init="$DIST_TIMEOUT_SEC"

    # Activation memory.
    model.recompute_granularity=full
    model.recompute_method=uniform
    model.recompute_num_layers=1
    model.mtp_num_layers=0

    # Vision memory: keep frozen for 96K SFT.
    model.freeze_vision_model=True
    model.freeze_vision_projection=True

    # Last-stage memory: TE chunked CE avoids materializing full 96K logits.
    model.cross_entropy_fusion_impl=te

    # GDN/linear-attention does not support packed THD layout.
    dataset.pack_sequences_in_batch=False

    model.moe_z_loss_coeff="$MOE_Z_LOSS_COEFF"

    # Optimizer memory.
    optimizer.optimizer_cpu_offload="$OPTIM_OFFLOAD"
    optimizer.optimizer_offload_fraction="$OPTIM_OFFLOAD_FRAC"
    optimizer.overlap_cpu_optimizer_d2h_h2d="$OPTIM_OFFLOAD_OVERLAP"
    optimizer.use_precision_aware_optimizer="$USE_PRECISION_AWARE_OPT"

    train.train_iters="$ITERS"
    train.global_batch_size="$GBS"
    train.micro_batch_size="$MBS"
    validation.eval_iters="$EVAL_ITERS"
    validation.eval_interval="$EVAL_INTERVAL"

    checkpoint.pretrained_checkpoint="$MCORE_PATH"
    checkpoint.save="$OUTPUT_DIR"
    checkpoint.save_interval="$SAVE_INTERVAL"
    checkpoint.load=null

    dataset.train_data_path="$TRAIN_DATA"
    dataset.hf_processor_path="$HF_MODEL"
    dataset.seq_length="$SEQ"
    dataset.drop_overlength_samples="$DROP_OVERLENGTH"
    dataset.validation_split_ratio="$VALID_SPLIT_RATIO"
    dataset.validation_split_seed="$VALID_SPLIT_SEED"

    logger.log_interval="$LOG_INTERVAL"
    logger.tensorboard_dir=null
    logger.wandb_project=null
)

if [[ -n "$LR" ]]; then
    OVERRIDES+=(optimizer.lr="$LR")
fi
if [[ -n "$LR_WARMUP_ITERS" ]]; then
    OVERRIDES+=(scheduler.lr_warmup_iters="$LR_WARMUP_ITERS")
fi
if [[ -n "$MOE_AUX_LOSS_COEFF" ]]; then
    OVERRIDES+=(model.moe_aux_loss_coeff="$MOE_AUX_LOSS_COEFF")
fi

cat <<EOF
============================================================
Qwen3.5-122B-A10B FULL SFT (24-node, 96K)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (node rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP} PP=${PP} CP=${CP} EP=${EP} DP=$(( NNODES * NPROC / TP / PP / CP ))
  Batch/Seq    : GBS=${GBS} MBS=${MBS} SEQ=${SEQ}
  Memory       : vision frozen, TE CE, recompute full/uniform/1, MTP off, optim offload=${OPTIM_OFFLOAD}
  Timeout      : init=${DIST_TIMEOUT_MIN}min steady=${DIST_TIMEOUT_SEC}s rdzv=${RDZV_TIMEOUT}s
  Log file     : $LOG_FILE
============================================================
EOF

"$VENV_PY" -u -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_conf "timeout=${RDZV_TIMEOUT}" \
    scripts/training/run_recipe.py \
    "${OVERRIDES[@]}"
