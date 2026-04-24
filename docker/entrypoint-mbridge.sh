#!/usr/bin/env bash
# Entrypoint for the qwen35-mbridge image.
#
# Behaviour:
#   1. If MBRIDGE_MOUNT_SRC_DIR is mounted (has pyproject.toml), prefer it over
#      the baked-in copy at MBRIDGE_IMAGE_SRC_DIR.
#      - Re-install megatron-bridge as editable against the mount, so any
#        edits the user makes on the host are immediately visible inside.
#      - chdir into the mount.
#   2. Otherwise fall back to the baked-in source.
#   3. Exec the user's CMD (default: bash).
#
# Env vars (set by the Dockerfile, can be overridden at `docker run -e`):
#   MBRIDGE_VENV_DIR        Path to the uv-managed venv  (default /opt/venv-mbridge)
#   MBRIDGE_IMAGE_SRC_DIR   Baked-in source path         (default /opt/Megatron-Bridge)
#   MBRIDGE_MOUNT_SRC_DIR   Expected mount path          (default /workspace/Megatron-Bridge)

set -e

VENV_DIR="${MBRIDGE_VENV_DIR:-/opt/venv-mbridge}"
IMAGE_SRC="${MBRIDGE_IMAGE_SRC_DIR:-/opt/Megatron-Bridge}"
MOUNT_SRC="${MBRIDGE_MOUNT_SRC_DIR:-/workspace/Megatron-Bridge}"

# Pick source directory: prefer mount if it looks like a megatron-bridge tree.
SRC_DIR=""
if [[ -f "${MOUNT_SRC}/pyproject.toml" ]]; then
    SRC_DIR="${MOUNT_SRC}"
    echo "[entrypoint] using mounted source at ${SRC_DIR}"
elif [[ -f "${IMAGE_SRC}/pyproject.toml" ]]; then
    SRC_DIR="${IMAGE_SRC}"
    echo "[entrypoint] using baked-in source at ${SRC_DIR}"
else
    echo "[entrypoint] WARNING: no megatron-bridge source found at ${MOUNT_SRC} or ${IMAGE_SRC}"
fi

# If we picked the mount, re-install editable so paths are correct.
# We use --no-deps because all deps are already satisfied from build-time.
if [[ "${SRC_DIR}" == "${MOUNT_SRC}" ]]; then
    # Best-effort init of submodule on the mount (no-op if already populated).
    if [[ -f "${SRC_DIR}/.gitmodules" && ! -f "${SRC_DIR}/3rdparty/Megatron-LM/megatron/__init__.py" ]]; then
        echo "[entrypoint] initialising submodule under mounted source..."
        (cd "${SRC_DIR}" && git submodule update --init --recursive) || \
            echo "[entrypoint] WARNING: submodule init failed; expect missing megatron-core"
    fi
    echo "[entrypoint] re-installing megatron-bridge editable against mount..."
    "${VENV_DIR}/bin/uv" pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${SRC_DIR}" \
        >/tmp/entrypoint-pip.log 2>&1 \
        || { echo "[entrypoint] WARNING: editable re-install failed; see /tmp/entrypoint-pip.log"; }
    cd "${SRC_DIR}"
elif [[ -n "${SRC_DIR}" ]]; then
    cd "${SRC_DIR}"
fi

# Make venv binaries first on PATH (Dockerfile already does this; safety net).
export PATH="${VENV_DIR}/bin:${PATH}"

# Hand off to user command (default `bash`).
exec "$@"
