#!/usr/bin/env bash
# Start or join the Ray cluster only.
#
# Run this script once on every node/container.  Rank 0 starts the Ray head;
# all other ranks join it as Ray workers.  Training is intentionally NOT
# launched here.  After all nodes have joined, manually run on rank 0:
#
#   /opt/venv-mbridge/bin/python run/run_on_all_nodes.py scripts/<train_script>.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHARED_DIR="${QS_TMP_DATA_PATH:-$(dirname "$REPO_ROOT")/meg-run/ray}"
HEAD_IP_FILE="${SHARED_DIR}/ray_head_ip.txt"
HEAD_PORT="${RAY_HEAD_PORT:-6379}"
GPU_NUM="${GPU_NUM:-8}"
VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"

if [[ ! -x "$VENV_PY" ]]; then
    echo "[start_ray] FATAL: $VENV_PY not found. Are you inside the qwen35-mbridge container?" >&2
    exit 1
fi

mkdir -p "$SHARED_DIR"

wait_for_port() {
    local host="$1"
    local port="$2"
    "$VENV_PY" - "$host" "$port" <<'PYEOF'
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
while True:
    try:
        with socket.create_connection((host, port), timeout=5):
            break
    except OSError:
        print(f"HEAD not ready at {host}:{port}, retrying in 10s...", flush=True)
        time.sleep(10)
PYEOF
}

get_host_ip() {
    "$VENV_PY" - <<'PYEOF'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    try:
        sock.connect(("8.8.8.8", 80))
        print(sock.getsockname()[0])
    except OSError:
        print(socket.gethostbyname(socket.gethostname()))
PYEOF
}

get_cluster_gpus() {
    "$VENV_PY" - <<'PYEOF' 2>/dev/null || printf '0\n'
import ray

ray.init(address="auto", ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
PYEOF
}

RANK="${RANK:-${NODE_RANK:-0}}"
WORLD_SIZE="${WORLD_SIZE:-${NNODES:-1}}"

if [[ "$RANK" -eq 0 ]]; then
    echo "[start_ray][rank 0] starting Ray head..."
    rm -f "$HEAD_IP_FILE"
    HEAD_IP="${HOST_IP:-${MASTER_ADDR:-$(get_host_ip)}}"
    TMP_FILE="${HEAD_IP_FILE}.tmp"
    printf '%s\n' "$HEAD_IP" > "$TMP_FILE"
    mv "$TMP_FILE" "$HEAD_IP_FILE"

    ray stop --force >/dev/null 2>&1 || true
    ray start --head --node-ip-address="$HEAD_IP" --port="$HEAD_PORT" --num-gpus="$GPU_NUM"

    echo "[start_ray][rank 0] Ray head ready at ${HEAD_IP}:${HEAD_PORT}"
    EXPECTED_GPUS=$((WORLD_SIZE * GPU_NUM))
    echo "[start_ray][rank 0] waiting for ${EXPECTED_GPUS} GPUs to join the Ray cluster..."
    while true; do
        AVAILABLE_GPUS="$(get_cluster_gpus)"
        echo "[start_ray][rank 0] available GPUs: ${AVAILABLE_GPUS}/${EXPECTED_GPUS}"
        if [[ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]]; then
            break
        fi
        sleep 10
    done

    echo "[start_ray][rank 0] Ray cluster is ready."
    echo "[start_ray][rank 0] Next, manually run on rank 0:"
    echo "  ${VENV_PY} ${REPO_ROOT}/run/run_on_all_nodes.py <TRAIN_SCRIPT> --master-addr ${HEAD_IP}"
else
    echo "[start_ray][rank ${RANK}] waiting for Ray head file: ${HEAD_IP_FILE}"
    while [[ ! -f "$HEAD_IP_FILE" ]]; do
        sleep 10
    done

    HEAD_IP="$(<"$HEAD_IP_FILE")"
    HEAD_ADDR="${HEAD_IP}:${HEAD_PORT}"
    echo "[start_ray][rank ${RANK}] waiting for Ray head ${HEAD_ADDR}..."
    wait_for_port "$HEAD_IP" "$HEAD_PORT"

    echo "[start_ray][rank ${RANK}] joining Ray cluster at ${HEAD_ADDR}..."
    ray stop --force >/dev/null 2>&1 || true
    ray start --address="$HEAD_ADDR" --num-gpus="$GPU_NUM"
    echo "[start_ray][rank ${RANK}] worker connected. Training is launched separately from rank 0."
fi
