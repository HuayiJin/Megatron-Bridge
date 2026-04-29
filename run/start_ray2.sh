#!/usr/bin/env bash
set -euo pipefail

SHARED_DIR=${QS_TMP_DATA_PATH}
HEAD_IP_FILE="${SHARED_DIR}/ray_head_ip.txt"

mkdir -p "${SHARED_DIR}"

HEAD_PORT=6379

if [ "${RANK}" -eq 0 ]; then
    echo "[Rank 0] Starting Ray head..."
    
    rm -f "${HEAD_IP_FILE}"
    HEAD_IP="${HOST_IP}"
    TMP_FILE="${HEAD_IP_FILE}.tmp"
    echo "${HEAD_IP}" > "${TMP_FILE}"
    mv "${TMP_FILE}" "${HEAD_IP_FILE}"

    ray stop && ray start --head --port=${HEAD_PORT}

    echo "[Rank 0] Head ready at ${HEAD_IP}:${HEAD_PORT}"
    
    # Wait for all GPUs to be available
    WORLD_SIZE=${WORLD_SIZE:-1}
    GPU_NUM=${GPU_NUM:-8}
    EXPECTED_GPUS=$((WORLD_SIZE * GPU_NUM))
    echo "[Rank 0] Waiting for ${EXPECTED_GPUS} GPUs to join the cluster..."
    while true; do
        # Use Python API to directly query cluster resources
        AVAILABLE_GPUS=$(python3 -c "import ray; ray.init(ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo "0")
        echo "[Rank 0] Available GPUs: ${AVAILABLE_GPUS}/${EXPECTED_GPUS}"
        if [ "${AVAILABLE_GPUS}" -ge "${EXPECTED_GPUS}" ]; then
            echo "[Rank 0] All ${EXPECTED_GPUS} GPUs are ready!"
            break
        fi
        sleep 10
    done

else
    echo "[Rank ${RANK}] Waiting for Ray head..."
    while [ ! -f "${HEAD_IP_FILE}" ]; do
        echo "${HEAD_IP_FILE} not exist yet, retrying in 10s..."
        sleep 10
    done

    HEAD_IP=$(cat "${HEAD_IP_FILE}")
    echo "===> Waiting for HEAD node (${HEAD_IP}:${HEAD_PORT}) to be ready..."
    command -v nc > /dev/null 2>&1 || apt update && apt install netcat-openbsd -y
    while true; do
        nc -z ${HEAD_IP} ${HEAD_PORT} && break
        echo "HEAD not ready yet, retrying in 10s..."
        sleep 10
    done

    echo "===> HEAD is ready! Starting worker..."
    HEAD_ADDR="${HEAD_IP}:${HEAD_PORT}"
    ray stop && ray start --address=${HEAD_ADDR}
    echo "===> Worker connected to Ray HEAD at ${HEAD_ADDR}"
fi
