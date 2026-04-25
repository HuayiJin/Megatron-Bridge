#!/usr/bin/env bash
# Quick liveness + progress probe for the running 2-node LoRA SFT.
# Run on either node. Refreshes every 10 s.

set -u

LATEST_LOG="$(ls -t /mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/logs/sft_lora_2node_*_rank*.log 2>/dev/null | head -1)"

while true; do
    clear
    echo "============================================================"
    echo " SFT progress probe — $(date +%H:%M:%S) on $(hostname)"
    echo "============================================================"

    # 1. torchrun process liveness
    NTORCH=$(pgrep -fc "torch.distributed.run" || true)
    NWORKER=$(pgrep -fc "run_recipe.py" || true)
    echo " torchrun procs : $NTORCH    workers (run_recipe): $NWORKER"

    # 2. GPU usage (memory + utilization)
    echo ""
    echo " --- GPU memory / util ---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '{printf "  GPU%s  mem=%5d MiB  util=%3d%%\n", $1, $2, $3}'

    # 3. Latest log tail (look for iter / loss)
    echo ""
    echo " --- Latest log (last 12 lines) ---"
    if [[ -n "$LATEST_LOG" && -f "$LATEST_LOG" ]]; then
        echo " $LATEST_LOG"
        echo "  ..."
        tail -12 "$LATEST_LOG" | sed 's/^/  /'
    else
        echo " (no log file found yet)"
    fi

    # 4. iter / loss highlight (whole log)
    if [[ -n "$LATEST_LOG" && -f "$LATEST_LOG" ]]; then
        ITER=$(grep -cE "^[ ]*iter[ ]+[0-9]+/" "$LATEST_LOG" 2>/dev/null || echo 0)
        if [[ "$ITER" -gt 0 ]]; then
            echo ""
            echo " --- Iter lines so far ($ITER total) ---"
            grep -E "^[ ]*iter[ ]+[0-9]+/" "$LATEST_LOG" | tail -5 | sed 's/^/  /'
        fi
    fi

    sleep 10
done
