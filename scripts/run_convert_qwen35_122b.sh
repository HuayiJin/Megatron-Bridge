#!/usr/bin/env bash
# Standard launcher for Qwen3.5-122B-A10B HF -> Megatron mcore conversion.
#
# Lessons baked in (see /data/temp/Megatron-LM/memory.md):
#   1. tmux session for long-running task; never `nohup &` (becomes zombie)
#   2. PYTHONUNBUFFERED=1 + python -u for real-time logs
#   3. NCCL_DEBUG=WARN to suppress 9000+ INFO lines
#   4. CUDA_VISIBLE_DEVICES=0 for single-GPU conversion (no multi-rank needed)
#   5. tee to both stdout and log file
#
# Usage:
#   bash scripts/run_convert_qwen35_122b.sh                 # interactive
#   tmux new -s qwen-conv 'bash scripts/run_convert_qwen35_122b.sh'   # detached
#
# Monitoring (in another shell):
#   watch -n 5 'free -g | head -3; echo ---GPU---; \
#       nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; \
#       echo ---DISK---; du -sh /data/temp/workspace/models/Qwen3.5-122B-A10B-mcore 2>/dev/null'

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/temp/Megatron-Bridge}"
HF_MODEL="${HF_MODEL:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B}"
OUT_DIR="${OUT_DIR:-/data/temp/workspace/models/Qwen3.5-122B-A10B-mcore}"
LOG_DIR="${LOG_DIR:-/data/temp/workspace/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/convert_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$OUT_DIR")" "$LOG_DIR"

# Sanity: env file
if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "[ERROR] venv python not found at $REPO_ROOT/.venv/bin/python"
    exit 1
fi
if [[ ! -d "$HF_MODEL" ]]; then
    echo "[ERROR] HF model dir not found: $HF_MODEL"
    exit 1
fi

# Pre-check: any leftover process?
if pgrep -f convert_qwen35_122b.py > /dev/null; then
    echo "[WARN] Found running convert_qwen35_122b.py; killing it"
    pkill -9 -f convert_qwen35_122b.py || true
    sleep 1
fi

# Pre-check: pre-existing output dir?
if [[ -d "$OUT_DIR" ]]; then
    echo "[INFO] Removing existing $OUT_DIR (will be re-created)"
    rm -rf "$OUT_DIR"
fi

echo "============================================================"
echo "Qwen3.5-122B-A10B HF -> Megatron conversion"
echo "============================================================"
echo "  HF source : $HF_MODEL"
echo "  Output    : $OUT_DIR"
echo "  Log file  : $LOG_FILE"
echo "  Repo      : $REPO_ROOT"
echo "============================================================"

cd "$REPO_ROOT"

PYTHONUNBUFFERED=1 \
NCCL_DEBUG=WARN \
TRANSFORMERS_VERBOSITY=info \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
.venv/bin/python -u scripts/convert_qwen35_122b.py \
    --hf-model "$HF_MODEL" \
    --megatron-path "$OUT_DIR" \
    --torch-dtype bfloat16 \
    2>&1 | tee "$LOG_FILE"

echo "============================================================"
echo "Conversion finished. Output: $OUT_DIR"
ls -la "$OUT_DIR" 2>/dev/null | head -20
echo "============================================================"
