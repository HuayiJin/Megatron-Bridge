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
#   3. Auto-install runtime-deferred deps (mamba-ssm/causal-conv1d/fla) if
#      not yet present. Skip if MBRIDGE_SKIP_RUNTIME_INSTALL=1 is set.
#   4. Exec the user's CMD (default: bash).
#
# Env vars (set by the Dockerfile, can be overridden at `docker run -e`):
#   MBRIDGE_VENV_DIR             venv path                (default /opt/venv-mbridge)
#   MBRIDGE_IMAGE_SRC_DIR        baked-in source          (default /opt/Megatron-Bridge)
#   MBRIDGE_MOUNT_SRC_DIR        expected mount path      (default /workspace/Megatron-Bridge)
#   MBRIDGE_SKIP_RUNTIME_INSTALL set to 1 to skip auto-install on launch
#   MAMBA_SSM_VERSION,
#   CAUSAL_CONV1D_VERSION,
#   FLA_VERSION                  version pins for the deferred install
#
# Tip for fast container restarts:
#   Mount a host directory onto /opt/venv-mbridge so site-packages persists
#   across `docker run` invocations:
#     -v /shared/qwen35-venv:/opt/venv-mbridge

set -e

VENV_DIR="${MBRIDGE_VENV_DIR:-/opt/venv-mbridge}"
IMAGE_SRC="${MBRIDGE_IMAGE_SRC_DIR:-/opt/Megatron-Bridge}"
MOUNT_SRC="${MBRIDGE_MOUNT_SRC_DIR:-/workspace/Megatron-Bridge}"

# ---- 1. Pick source directory: prefer mount if it looks like a mbridge tree.
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

# ---- 2. If we picked the mount, re-install editable so paths are correct.
# We use --no-deps because deps are already satisfied from build-time + runtime install.
if [[ "${SRC_DIR}" == "${MOUNT_SRC}" ]]; then
    # Best-effort init of submodule on the mount.
    # Megatron-LM has a namespace `megatron/` package, so check `megatron/core/__init__.py`.
    if [[ -f "${SRC_DIR}/.gitmodules" && ! -f "${SRC_DIR}/3rdparty/Megatron-LM/megatron/core/__init__.py" ]]; then
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

# ---- 3. Auto-install runtime-deferred deps (idempotent, fast no-op when present).
# These are installed at runtime to keep the image small and avoid the build-
# time CUDA-driver issue with `import mamba_ssm`. See scripts/install_runtime_deps.sh.
if [[ "${MBRIDGE_SKIP_RUNTIME_INSTALL:-0}" != "1" ]]; then
    if [[ -x "${SRC_DIR:-}/scripts/install_runtime_deps.sh" ]]; then
        echo "[entrypoint] checking runtime-deferred deps (mamba-ssm/causal-conv1d/fla)..."
        VENV_PY="${VENV_DIR}/bin/python" \
        VENV_DIR="${VENV_DIR}" \
        bash "${SRC_DIR}/scripts/install_runtime_deps.sh" \
            || { echo "[entrypoint] WARNING: runtime deps install failed"; }
    elif [[ -x "${IMAGE_SRC}/scripts/install_runtime_deps.sh" ]]; then
        echo "[entrypoint] (using baked-in install_runtime_deps.sh)"
        VENV_PY="${VENV_DIR}/bin/python" \
        VENV_DIR="${VENV_DIR}" \
        bash "${IMAGE_SRC}/scripts/install_runtime_deps.sh" \
            || { echo "[entrypoint] WARNING: runtime deps install failed"; }
    fi
else
    echo "[entrypoint] MBRIDGE_SKIP_RUNTIME_INSTALL=1; skipping deferred deps install"
fi

# Make venv binaries first on PATH (Dockerfile already does this; safety net).
export PATH="${VENV_DIR}/bin:${PATH}"

# Hand off to user command (default `bash`).
exec "$@"
