#!/usr/bin/env bash
# Monitoring dashboard for the long-running conversion.
# Usage: bash scripts/watch_convert.sh
# Or:    watch -n 5 'bash scripts/watch_convert.sh'

OUT_DIR="${OUT_DIR:-/data/temp/workspace/models/Qwen3.5-122B-A10B-mcore}"
LOG_DIR="${LOG_DIR:-/data/temp/workspace/logs}"

echo "=== Time: $(date '+%F %T') ==="

echo "--- Process ---"
ps -ef | grep convert_qwen35_122b | grep -v grep | head -3
PID=$(pgrep -f convert_qwen35_122b.py | head -1 || echo "")
if [[ -n "$PID" ]]; then
    cat /proc/$PID/status 2>/dev/null | grep -E "^(State|VmRSS|VmSize|Threads)"
fi

echo "--- Memory ---"
free -h | head -3

echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>&1 | head -8

echo "--- Output dir size ---"
if [[ -d "$OUT_DIR" ]]; then
    du -sh "$OUT_DIR" 2>/dev/null
    ls "$OUT_DIR" 2>/dev/null | head -5
else
    echo "  (not created yet)"
fi

echo "--- Last 10 log lines ---"
LATEST_LOG=$(ls -t "$LOG_DIR"/convert_*.log 2>/dev/null | head -1)
if [[ -n "$LATEST_LOG" ]]; then
    grep -vE "NCCL INFO|Channel " "$LATEST_LOG" | tail -10
else
    echo "  (no log found)"
fi
