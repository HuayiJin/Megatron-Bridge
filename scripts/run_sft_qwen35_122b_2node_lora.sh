#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B LoRA SFT — 2 nodes × 8 GPU (= 16 × H800 80G).
#
# Why LoRA on 16 GPU:
#   Full bf16 SFT of 122B has ~244GB params + ~1TB Adam state. Even with
#   TP=2 PP=4 EP=8 + recompute, 16 × 80G = 1.28TB is too tight.
#   LoRA freezes the base, so peak GPU memory is dominated by activations
#   and a tiny LoRA adapter; this fits cleanly on 2 × 8 H800.
#
# Recipe used:
#   qwen35_vl_122b_a10b_peft_config (TP=2, PP=1, EP=8, LR=2e-4, GBS=36)
#   — official 2-node, 16-GPU recipe.
#
# Container assumptions (NGC 25.06 + cu12.9, see Dockerfile.qwen35):
#   - /opt/venv-mbridge/bin/python is the inherited NGC venv
#   - torch 2.8.0a / TE 2.4 / cuDNN 9.10 / APEX cuda ext / flash-attn 2.7.4 — all OK
#   - mamba_ssm / causal_conv1d / fla are NOT in the image; this script will
#     auto-install them on first launch via scripts/install_runtime_deps.sh
#   - MASTER_ADDR / MASTER_PORT / NCCL_* are already set by the cluster scheduler
#
# Required env (per-node, set by the scheduler — verify with `env | grep -E ...`):
#   MASTER_ADDR  master node hostname
#   MASTER_PORT  master port
#   RANK         this node's index in [0, NNODES) — used as torchrun --node_rank
#                (set NODE_RANK to override if your scheduler uses a different name)
#   WORLD_SIZE   number of nodes (default 2)
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_2node_lora.sh
#
# Override knobs (env):
#   ITERS=20   bash scripts/run_sft_qwen35_122b_2node_lora.sh   # train iters
#   SEQ=4096   bash scripts/run_sft_qwen35_122b_2node_lora.sh   # sequence length
#   GBS=16 MBS=1 ...

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo + paths
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge}"
HF_MODEL="${HF_MODEL:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B}"
MCORE_PATH="${MCORE_PATH:-/mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/demo_data/train_demo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/qwen35_122b_lora_demo}"
LOG_DIR="${LOG_DIR:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_lora_2node_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config — read from cluster-injected env vars, fall back to local
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"                                 # GPUs per node
NNODES="${NNODES:-${WORLD_SIZE:-2}}"                # number of nodes
NODE_RANK="${NODE_RANK:-${RANK:-0}}"                # this node's index
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Training hyperparameters (overridable via env)
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_122b_a10b_peft_config}"
PEFT_SCHEME="${PEFT_SCHEME:-lora}"
ITERS="${ITERS:-10}"            # short demo run
SEQ="${SEQ:-2048}"              # 2048 for fast demo; recipe default is 4096
GBS="${GBS:-16}"                # global batch size
MBS="${MBS:-1}"                 # micro batch size
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"   # don't save during demo

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -d "$REPO_ROOT" ]]; then
    echo "[ERROR] REPO_ROOT not found: $REPO_ROOT" >&2; exit 1
fi
if [[ ! -d "$MCORE_PATH/iter_0000000" ]]; then
    echo "[ERROR] mcore checkpoint not at $MCORE_PATH/iter_0000000" >&2; exit 1
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "[ERROR] train jsonl not found: $TRAIN_DATA" >&2; exit 1
fi
if [[ ! -d "$HF_MODEL" ]]; then
    echo "[ERROR] HF model dir not found: $HF_MODEL" >&2; exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Pick the venv python
# ---------------------------------------------------------------------------
VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] venv python missing: $VENV_PY" >&2
    echo "        Are you inside the qwen35-mbridge container?" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-install runtime-deferred deps (mamba-ssm / causal-conv1d / fla).
# Only on rank 0 to avoid races on shared filesystems; other nodes wait via
# the existence of fla in the venv (entrypoint usually handles this, this is
# a safety net for users who started the container with a custom CMD).
# ---------------------------------------------------------------------------
if [[ "$NODE_RANK" -eq 0 ]]; then
    if ! "$VENV_PY" -c 'import mamba_ssm, causal_conv1d, fla' >/dev/null 2>&1; then
        echo "[run_sft] runtime deps missing — installing (this may take 5-15 min on first run)..."
        bash "$REPO_ROOT/scripts/install_runtime_deps.sh"
    else
        echo "[run_sft] runtime deps already installed."
    fi
fi

# ---------------------------------------------------------------------------
# NGC container is healthy → no monkey-patches needed.
# Override only when running on bare-metal venv.
# ---------------------------------------------------------------------------
export MBRIDGE_PATCH_NVIDIA_FILE="${MBRIDGE_PATCH_NVIDIA_FILE:-0}"
export MBRIDGE_DISABLE_CUDNN="${MBRIDGE_DISABLE_CUDNN:-0}"
export MBRIDGE_PATCH_GRAD_FUSION="${MBRIDGE_PATCH_GRAD_FUSION:-0}"

# Quiet NCCL during normal runs; flip to INFO for debugging
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# Make HF processor load faster on multinode by avoiding repeated downloads
export HF_HOME="${HF_HOME:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/hf_cache}"
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# Recipe + dataset overrides
# ---------------------------------------------------------------------------
OVERRIDES=(
    --recipe "$RECIPE"
    --peft_scheme "$PEFT_SCHEME"
    --dataset vlm-preloaded
    --step_func vlm_step
    --hf_path "$HF_MODEL"

    # Parallelism: the LoRA recipe defaults are TP=2 PP=1 EP=8 (16 GPU). Keep them.

    # GDN constraint — never enable sequence packing on Qwen3.5-VL with linear-attn.
    dataset.pack_sequences_in_batch=False

    # Iters / batch
    train.train_iters="$ITERS"
    train.global_batch_size="$GBS"
    train.micro_batch_size="$MBS"
    train.eval_iters=0
    train.eval_interval=1000000

    # Checkpoint: load mcore base, do not save during demo
    checkpoint.pretrained_checkpoint="$MCORE_PATH"
    checkpoint.save="$OUTPUT_DIR"
    checkpoint.save_interval="$SAVE_INTERVAL"
    checkpoint.load=null

    # Dataset: PreloadedVLMConversationProvider
    dataset.train_data_path="$TRAIN_DATA"
    dataset.hf_processor_path="$HF_MODEL"
    dataset.seq_length="$SEQ"

    # Logging
    logger.log_interval="$LOG_INTERVAL"
    logger.tensorboard_dir=null
    logger.wandb_project=null
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
cat <<EOF
============================================================
Qwen3.5-122B-A10B LoRA SFT (2-node demo)
  Recipe       : $RECIPE  (peft=$PEFT_SCHEME)
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Iters/GBS    : ${ITERS} / ${GBS}
  Seq length   : ${SEQ}
  Log file     : $LOG_FILE
  Venv python  : $VENV_PY
============================================================
EOF

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
"$VENV_PY" -u -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/training/run_recipe.py \
    "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"
