#!/usr/bin/env bash
# Phase C smoke: Load mcore checkpoint + VLM forward + greedy generate.
# Verifies the conversion produced a working model end-to-end.
#
# Why this (vs full HF<->Mcore logits compare):
#   - 122B HF model needs device_map="auto" but compare.py hardcodes "cuda".
#   - Generation produces human-readable proof of correctness (not just numbers).
#   - 8-GPU EP=8 fits comfortably (~30-40GB/GPU after init).
#
# Usage:
#   bash scripts/run_smoke_qwen35_122b.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/temp/Megatron-Bridge}"
HF_MODEL="${HF_MODEL:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B}"
MCORE_PATH="${MCORE_PATH:-/data/temp/workspace/models/Qwen3.5-122B-A10B-mcore/iter_0000000}"
LOG_DIR="${LOG_DIR:-/data/temp/workspace/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/smoke_$(date +%Y%m%d_%H%M%S).log}"

NPROC="${NPROC:-8}"
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

# Default to the mbridge repo logo PNG; safe local file
IMAGE_PATH="${IMAGE_PATH:-${REPO_ROOT}/Repo-Mbridge.png}"
PROMPT="${PROMPT:-Describe this image in one sentence.}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$MCORE_PATH" ]]; then
    echo "[ERROR] mcore checkpoint dir missing: $MCORE_PATH"
    echo "        Run scripts/run_convert_qwen35_122b.sh first."
    exit 1
fi
if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "[WARN] image not found: $IMAGE_PATH; will run text-only smoke (no --image_path)"
    IMAGE_FLAG=""
else
    IMAGE_FLAG="--image_path $IMAGE_PATH"
fi

cd "$REPO_ROOT"

echo "============================================================"
echo "Qwen3.5-122B-A10B mcore -> VLM forward smoke"
echo "  HF (config/tokenizer): $HF_MODEL"
echo "  mcore ckpt: $MCORE_PATH"
echo "  Image:  ${IMAGE_PATH:-<none, text-only>}"
echo "  Prompt: $PROMPT"
echo "  Parallel: nproc=$NPROC TP=$TP PP=$PP EP=$EP"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  Log: $LOG_FILE"
echo "============================================================"

# cuDNN sublibrary loading workaround: the system has cuDNN 9.10.2 but our venv
# has 9.7.1, which collides. Prepend venv's cuDNN to LD_LIBRARY_PATH so all
# sublib symbols come from a consistent version.
VENV_CUDNN_LIB="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
VENV_CUBLAS_LIB="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
VENV_NCCL_LIB="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="${VENV_CUDNN_LIB}:${VENV_CUBLAS_LIB}:${VENV_NCCL_LIB}:${LD_LIBRARY_PATH:-}"

PYTHONUNBUFFERED=1 \
NCCL_DEBUG=WARN \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
.venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    scripts/smoke_qwen35_122b.py \
    --hf_model_path "$HF_MODEL" \
    --megatron_model_path "$MCORE_PATH" \
    $IMAGE_FLAG \
    --prompt "$PROMPT" \
    --tp $TP --pp $PP --ep $EP \
    --max_new_tokens $MAX_NEW_TOKENS \
    2>&1 | tee "$LOG_FILE"
