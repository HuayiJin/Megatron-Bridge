#!/usr/bin/env bash
# Phase C: Logits comparison between HF and Megatron mcore checkpoints.
# Run AFTER scripts/run_convert_qwen35_122b.sh succeeded.
#
# Default: TP=1 PP=1 EP=8 on a single 8-GPU node (matches conversion.sh recipe)
# Single GPU verifies: nproc=1 with TP=1 PP=1 EP=1 (uses much less GPU mem)
#
# Usage:
#   bash scripts/run_compare_qwen35_122b.sh              # 8-GPU EP=8
#   NPROC=1 EP=1 bash scripts/run_compare_qwen35_122b.sh # 1-GPU smoke (smaller)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/temp/Megatron-Bridge}"
HF_MODEL="${HF_MODEL:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B}"
MCORE_PATH="${MCORE_PATH:-/data/temp/workspace/models/Qwen3.5-122B-A10B-mcore}"
LOG_DIR="${LOG_DIR:-/data/temp/workspace/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/compare_$(date +%Y%m%d_%H%M%S).log}"

NPROC="${NPROC:-8}"
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-8}"

# Image + prompt for VL inference. Use Qwen demo image to keep network-friendly.
IMAGE_URL="${IMAGE_URL:-https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg}"
PROMPT="${PROMPT:-Describe this image.}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$MCORE_PATH" ]]; then
    echo "[ERROR] mcore checkpoint dir missing: $MCORE_PATH"
    echo "        Run scripts/run_convert_qwen35_122b.sh first."
    exit 1
fi

cd "$REPO_ROOT"

echo "============================================================"
echo "Qwen3.5-122B-A10B HF vs Megatron logits comparison"
echo "  HF:     $HF_MODEL"
echo "  Mcore:  $MCORE_PATH"
echo "  Image:  $IMAGE_URL"
echo "  Prompt: $PROMPT"
echo "  Parallel: nproc=$NPROC TP=$TP PP=$PP EP=$EP"
echo "  Log:    $LOG_FILE"
echo "============================================================"

PYTHONUNBUFFERED=1 \
NCCL_DEBUG=WARN \
.venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "$HF_MODEL" \
    --megatron_model_path "$MCORE_PATH" \
    --model_class "Qwen3_5MoeForConditionalGeneration" \
    --image_path "$IMAGE_URL" \
    --prompt "$PROMPT" \
    --tp $TP --pp $PP --ep $EP \
    2>&1 | tee "$LOG_FILE"
