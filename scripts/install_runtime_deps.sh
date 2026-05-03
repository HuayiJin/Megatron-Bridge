#!/usr/bin/env bash
# Prepare the megatron-bridge runtime environment inside the qwen35-mbridge container.
#
# This script is the SINGLE setup step that must be run once per container
# launch, from the /mnt source tree, before starting training.
#
# What it does:
#   1. Verify the venv (/opt/venv-mbridge) is present.
#   2. Initialize 3rdparty/Megatron-LM git submodule if empty.
#   3. Run `uv sync` to install megatron-core (editable, from submodule) and
#      all other Python deps declared in pyproject.toml, skipping packages
#      already provided by the NGC base image (torch, TE, flash-attn, etc.)
#      and packages already baked into the image (mamba-ssm, causal-conv1d,
#      flash-linear-attention, fla-core).
#      NOTE: Most pure-Python packages (transformers, peft, datasets, etc.) and
#      nvidia-resiliency-ext are pre-installed in the image via
#      docker/requirements-prebaked.txt. `uv sync` detects them in site-packages
#      and skips re-downloading, making this step fast (~1 min vs ~10 min).
#      This script is fully backward-compatible: it works identically on images
#      built before the prebaked optimization was introduced.
#   4. Run `uv pip install -e .` to install megatron-bridge itself (editable,
#      pointing at the /mnt source tree so live edits are picked up).
#   5. Verify baked-in packages (mamba_ssm, causal_conv1d, fla) and a sample
#      of prebaked packages (transformers, nvidia_resiliency_ext) import OK.
#
# Idempotent:
#   Steps 3-4 are fast no-ops if already installed (uv detects no changes).
#   Run with --force to reinstall everything unconditionally.
#
# Usage (can be called from any directory):
#   bash /path/to/Megatron-Bridge/scripts/install_runtime_deps.sh
#   bash /path/to/Megatron-Bridge/scripts/install_runtime_deps.sh --force

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-/opt/venv-mbridge}"
VENV_PY="${VENV_PY:-${VENV_DIR}/bin/python}"
VENV_UV="${VENV_UV:-/usr/local/bin/uv}"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "[install_runtime_deps] unknown arg: $arg"; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 0. Locate repo root from this script's own path (no cd required)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    echo "[install_runtime_deps] FATAL: pyproject.toml not found under ${REPO_ROOT}"
    echo "  Is this script inside the Megatron-Bridge repo at scripts/install_runtime_deps.sh?"
    exit 1
fi

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Verify venv
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV_PY}" ]]; then
    echo "[install_runtime_deps] FATAL: venv python not found at ${VENV_PY}"
    echo "  Is the qwen35-mbridge container running?"
    exit 1
fi
echo "[install_runtime_deps] venv: ${VENV_DIR}"
"${VENV_PY}" -c "import torch; print('  torch:', torch.__version__)"
"${VENV_PY}" -c "import transformer_engine; print('  TE:   ', transformer_engine.__version__)"

# ---------------------------------------------------------------------------
# 2. Initialize Megatron-LM submodule if empty
# ---------------------------------------------------------------------------
MCORE_INIT="${REPO_ROOT}/3rdparty/Megatron-LM/megatron/core/__init__.py"
if [[ ! -f "${MCORE_INIT}" ]]; then
    echo "[install_runtime_deps] 3rdparty/Megatron-LM submodule is empty — initializing..."
    git submodule update --init --recursive
    if [[ ! -f "${MCORE_INIT}" ]]; then
        echo "[install_runtime_deps] FATAL: submodule init failed; ${MCORE_INIT} still missing"
        exit 1
    fi
    echo "[install_runtime_deps] submodule OK"
else
    echo "[install_runtime_deps] submodule already initialized"
fi

# ---------------------------------------------------------------------------
# 3. uv sync — install megatron-core (editable) + all pyproject.toml deps
#    Skip packages provided by NGC image or already baked into the image.
# ---------------------------------------------------------------------------
echo "[install_runtime_deps] running uv sync (megatron-core + Python deps)..."

FORCE_FLAGS=()
[[ "${FORCE}" -eq 1 ]] && FORCE_FLAGS=("--reinstall")

UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
"${VENV_UV}" sync \
    --link-mode copy \
    --inexact \
    --all-extras \
    --all-groups \
    "${FORCE_FLAGS[@]}" \
    --no-install-package torch \
    --no-install-package torchvision \
    --no-install-package triton \
    --no-install-package transformer-engine \
    --no-install-package transformer-engine-torch \
    --no-install-package transformer-engine-cu12 \
    --no-install-package flash-attn \
    --no-install-package mamba-ssm \
    --no-install-package causal-conv1d \
    --no-install-package flash-linear-attention \
    --no-install-package fla-core \
    --no-install-package nvidia-cublas \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cublas-cu13 \
    --no-install-package nvidia-cuda-cupti \
    --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-cupti-cu13 \
    --no-install-package nvidia-cuda-nvrtc \
    --no-install-package nvidia-cuda-nvrtc-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu13 \
    --no-install-package nvidia-cuda-runtime \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cuda-runtime-cu13 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cudnn-cu13 \
    --no-install-package nvidia-cufft \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-cufft-cu13 \
    --no-install-package nvidia-cufile \
    --no-install-package nvidia-curand \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-curand-cu13 \
    --no-install-package nvidia-cusolver \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusolver-cu13 \
    --no-install-package nvidia-cusparse \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-cusparse-cu13 \
    --no-install-package nvidia-cusparselt-cu12 \
    --no-install-package nvidia-cusparselt-cu13 \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-nccl-cu13 \
    --no-install-package nvidia-nvjitlink \
    --no-install-package nvidia-nvjitlink-cu12 \
    --no-install-package nvidia-nvjitlink-cu13 \
    --no-install-package nvidia-nvshmem-cu12 \
    --no-install-package nvidia-nvshmem-cu13 \
    --no-install-package nvidia-nvtx \
    --no-install-package nvidia-nvtx-cu12 \
    --no-install-package nvidia-nvtx-cu13 \
    --no-install-package cuda-toolkit \
    --no-install-package apex

echo "[install_runtime_deps] uv sync done"

# ---------------------------------------------------------------------------
# 4. Install megatron-bridge itself as editable (points at /mnt source tree)
# ---------------------------------------------------------------------------
echo "[install_runtime_deps] installing megatron-bridge editable from ${REPO_ROOT}..."

"${VENV_UV}" pip install \
    --link-mode copy \
    --python "${VENV_PY}" \
    --no-deps \
    "${FORCE_FLAGS[@]}" \
    -e "${REPO_ROOT}"

# Verify it resolves to the /mnt path (not /opt)
MBRIDGE_FILE=$("${VENV_PY}" -c "import megatron.bridge; print(megatron.bridge.__file__)")
echo "[install_runtime_deps] megatron.bridge → ${MBRIDGE_FILE}"
if [[ "${MBRIDGE_FILE}" != "${REPO_ROOT}"* ]]; then
    echo "[install_runtime_deps] WARNING: megatron.bridge resolved to ${MBRIDGE_FILE}"
    echo "  expected path under ${REPO_ROOT}"
    echo "  There may be a stale .pth in the venv pointing elsewhere."
fi

# ---------------------------------------------------------------------------
# 4b. TE 2.4 compatibility: clean up any stale te_distributed_compat.pth.
#
# Historical note: an earlier version of this script wrote a .pth file to
# site-packages to force `import transformer_engine.pytorch.distributed` at
# Python startup. This was unsound: Python's site module processes venv
# .pth files BEFORE adding /usr/local/lib (NGC dist-packages) to sys.path,
# so the import fails with ModuleNotFoundError on every fresh container start.
# The fix (2026-04-26) is to do the import explicitly in run_recipe.py after
# the process is fully initialized, at which point sys.path is complete.
# Remove any leftover .pth file so it does not cause spurious errors.
# ---------------------------------------------------------------------------
SITE_PACKAGES="$("${VENV_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TE_COMPAT_PTH="${SITE_PACKAGES}/te_distributed_compat.pth"
if [[ -f "${TE_COMPAT_PTH}" ]]; then
    rm -f "${TE_COMPAT_PTH}"
    echo "[install_runtime_deps] removed stale TE compat pth: ${TE_COMPAT_PTH}"
fi

# ---------------------------------------------------------------------------
# 5. Verify packages baked into the image.
#
#    Group A — wheel-installed at image build time (always present):
#      mamba_ssm, causal_conv1d, fla
#      fla is installed via `pip install flash-linear-attention` (pulls
#      fla-core); it is NOT installed by uv sync above.
#
#    Group B — prebaked pure-Python packages (present on images built after
#      the docker/requirements-prebaked.txt optimization; absent on older
#      images).  We verify these with a best-effort check and print a warning
#      rather than failing, so the script remains backward-compatible.
# ---------------------------------------------------------------------------
echo "[install_runtime_deps] verifying baked-in and prebaked packages..."
"${VENV_PY}" - <<'PYEOF'
import sys

ok = True

# Group A: must always be present (hard failure)
required = ("mamba_ssm", "causal_conv1d", "fla")
for name in required:
    try:
        m = __import__(name)
        v = getattr(m, "__version__", "<no version>")
        print(f"  ok    {name:26s} {v}")
    except Exception as exc:
        print(f"  FAIL  {name:26s} {type(exc).__name__}: {exc}")
        ok = False

# Group B: prebaked packages — soft check (warn but don't fail on old images)
prebaked_sample = (
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "omegaconf",
    "wandb",
    "nvidia_resiliency_ext",
)
prebaked_missing = []
for name in prebaked_sample:
    try:
        m = __import__(name)
        v = getattr(m, "__version__", "<no version>")
        print(f"  ok    {name:26s} {v}  [prebaked]")
    except Exception:
        prebaked_missing.append(name)

if prebaked_missing:
    print(
        f"\n  NOTE: {len(prebaked_missing)} prebaked package(s) not found in image "
        f"({', '.join(prebaked_missing)}).\n"
        "  This is expected on images built before docker/requirements-prebaked.txt\n"
        "  was introduced. `uv sync` (step 3) will have installed them already.\n"
        "  Rebuild the image to get the faster startup benefit."
    )

sys.exit(0 if ok else 1)
PYEOF

echo "[install_runtime_deps] all done — container is ready to run training."
echo "  Next: start Ray on every node with: bash run/start_ray.sh"
echo "        then run on rank 0: /opt/venv-mbridge/bin/python run/run_on_all_nodes.py <TRAIN_SCRIPT>"
