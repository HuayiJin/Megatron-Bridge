#!/usr/bin/env bash
# Per-node bootstrap + SFT launcher for Qwen3.5-122B-A10B (2-node × 8 GPU LoRA).
#
# Run this on EACH node (after entering the qwen35-mbridge container):
#
#   bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start.sh
#
# Cluster injects MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK already.
# If your scheduler did NOT inject RANK, prepend it manually:
#
#   RANK=0 MASTER_PORT=23456 bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start.sh   # master node
#   RANK=1 MASTER_PORT=23456 bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start.sh   # worker node
#
# IMPORTANT: this cluster's true default MASTER_PORT is 23456, but the
# master container sometimes inherits a stale value (e.g. 23964) while the
# worker sees the correct one. If they mismatch, torchrun rendezvous hangs
# silently forever with no error. ALWAYS pass MASTER_PORT=23456 explicitly.
# See memory.md Pitfall #11.
#
# What this script does (idempotent — safe to re-run):
#   1. Installs causal-conv1d / mamba-ssm / fla from prebuilt wheels on NAS
#      (skips if already installed). NO source build (which would take 30+ min).
#   2. Patches mamba_ssm/__init__.py so the failing selective_scan_cuda load
#      becomes a warning instead of an error (NGC torch 2.8a ABI mismatch).
#      Qwen3.5-VL GDN goes through the Triton path and does not need
#      selective_scan_cuda. See memory.md §Mamba-SSM ABI Workaround.
#   3. Launches LoRA SFT via scripts/run_sft_qwen35_122b_2node_lora.sh.

set -euo pipefail
export https_proxy=http://10.7.4.2:3128 && export HTTPS_PROXY=http://10.7.4.2:3128
# ---------------------------------------------------------------------------
# Path resolution
#   REPO_ROOT  — auto-derived from this script's location (run/ subdir).
#                Override by exporting REPO_ROOT=/your/path before calling.
#   MEG_RUN_DIR — working directory for wheels, logs, outputs, hf_cache.
#                Defaults to a sibling directory of REPO_ROOT named "meg-run".
#                Override by exporting MEG_RUN_DIR=/your/path.
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEG_RUN_DIR="${MEG_RUN_DIR:-$(dirname "$REPO_ROOT")/meg-run}"
WHEEL_DIR="${WHEEL_DIR:-${MEG_RUN_DIR}/wheels}"
VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"

CAUSAL_WHL="${CAUSAL_WHL:-${WHEEL_DIR}/causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
MAMBA_WHL="${MAMBA_WHL:-${WHEEL_DIR}/mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

# ---------------------------------------------------------------------------
# 0. Sanity
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV_PY" ]]; then
    echo "[start.sh] FATAL: $VENV_PY not found. Are you inside the qwen35-mbridge container?" >&2
    exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
    echo "[start.sh] FATAL: repo root not found: $REPO_ROOT" >&2
    exit 1
fi

cd "$REPO_ROOT"
echo "[start.sh] node-local hostname=$(hostname)  RANK=${RANK:-?}  WORLD_SIZE=${WORLD_SIZE:-?}  MASTER=${MASTER_ADDR:-?}:${MASTER_PORT:-?}"

# ---------------------------------------------------------------------------
# 1. Install runtime-deferred deps from prebuilt wheels (idempotent)
# ---------------------------------------------------------------------------
need_causal=0
need_mamba=0
need_fla=0
"$VENV_PY" -c 'import causal_conv1d' >/dev/null 2>&1 || need_causal=1
"$VENV_PY" -c 'import mamba_ssm' >/dev/null 2>&1 || need_mamba=1
"$VENV_PY" -c 'import fla' >/dev/null 2>&1 || need_fla=1

if (( need_causal || need_mamba || need_fla )); then
    echo "[start.sh] installing missing runtime deps:"
    (( need_causal )) && echo "  - causal_conv1d (wheel)"
    (( need_mamba ))  && echo "  - mamba_ssm     (wheel + __init__.py patch)"
    (( need_fla ))    && echo "  - fla           (pure python)"

    install_args=()
    if (( need_causal )); then
        [[ -f "$CAUSAL_WHL" ]] || { echo "[start.sh] FATAL: missing wheel $CAUSAL_WHL" >&2; exit 1; }
        install_args+=("$CAUSAL_WHL")
    fi
    if (( need_mamba )); then
        [[ -f "$MAMBA_WHL" ]] || { echo "[start.sh] FATAL: missing wheel $MAMBA_WHL" >&2; exit 1; }
        install_args+=("$MAMBA_WHL")
    fi
    if (( ${#install_args[@]} > 0 )); then
        "$VENV_PY" -m pip install --no-cache-dir --no-deps "${install_args[@]}"
    fi
    if (( need_fla )); then
        "$VENV_PY" -m pip install --no-cache-dir --no-deps flash-linear-attention==0.4.2
    fi
else
    echo "[start.sh] runtime deps already installed; skipping pip."
fi

# ---------------------------------------------------------------------------
# 2. Patch mamba_ssm/__init__.py (idempotent — only writes if not yet patched)
#
# IMPORTANT: locate the file via filesystem, NOT `import mamba_ssm`.
# Before the patch, the unpatched __init__.py imports selective_scan_cuda,
# which on NGC torch 2.8a fails with the c10::cuda::SetDevice ABI error.
# So `python -c 'import mamba_ssm'` raises and we'd never get here.
# ---------------------------------------------------------------------------
SITE_PACKAGES="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
MAMBA_INIT="${SITE_PACKAGES}/mamba_ssm/__init__.py"
if [[ ! -f "$MAMBA_INIT" ]]; then
    echo "[start.sh] FATAL: $MAMBA_INIT does not exist after pip install" >&2
    exit 1
fi
if grep -q "NGC nv25.06 patched torch" "$MAMBA_INIT" 2>/dev/null; then
    echo "[start.sh] mamba_ssm/__init__.py already patched."
else
    echo "[start.sh] patching $MAMBA_INIT"
    cat > "$MAMBA_INIT" <<'PYEOF'
__version__ = "2.3.1"

# NGC nv25.06 patched torch 2.8 has an ABI between stock torch 2.9 and 2.10,
# so neither stock cu12torch2.8/2.9/2.10 prebuilt wheel matches:
#   - torch2.8/2.9 wheel:  undefined c10::cuda::SetDevice(int8_t, bool)
#   - torch2.10  wheel:    undefined c10::cuda::c10_cuda_check_implementation(..., bool)
# The `selective_scan_cuda` extension is only used by the legacy
# selective_scan_interface / Mamba1 / mamba_inner_fn paths. Megatron-Core's
# Qwen3.5-VL GDN path uses mamba_ssm.ops.triton.ssd_combined which is pure
# Python + Triton + causal_conv1d (the latter has a working wheel). So we
# make the legacy CUDA imports best-effort: keep them if they load, but do
# not break `import mamba_ssm` if they don't.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn  # noqa: F401
except ImportError as _e:
    selective_scan_fn = None
    mamba_inner_fn = None
    import warnings
    warnings.warn(
        "mamba_ssm.selective_scan_cuda failed to load (NGC torch ABI mismatch); "
        "selective_scan_fn / mamba_inner_fn / Mamba1 will not work, but the "
        "Triton-based mamba_ssm.ops.triton.ssd_combined path used by "
        "megatron-core (Mamba2 / GDN) is unaffected. Original error: " + str(_e)
    )

try:
    from mamba_ssm.modules.mamba_simple import Mamba  # noqa: F401
except ImportError:
    Mamba = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2  # noqa: F401
except ImportError:
    Mamba2 = None

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: F401
except ImportError:
    MambaLMHeadModel = None
PYEOF
    echo "[start.sh] patch applied."
fi

# ---------------------------------------------------------------------------
# 3. Final verification (must succeed before launching SFT)
# ---------------------------------------------------------------------------
"$VENV_PY" -c "
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import mamba_ssm
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
import causal_conv1d, fla, torch
print('[verify] torch         ', torch.__version__, 'cuda', torch.version.cuda)
print('[verify] mamba_ssm     ', mamba_ssm.__version__, '(Mamba2 ok)')
print('[verify] causal_conv1d ', causal_conv1d.__version__)
print('[verify] fla           ', fla.__version__)
print('[verify] triton ssd_combined import OK')
" || { echo "[start.sh] FATAL: verification failed — runtime deps still broken" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 4. Launch SFT
# ---------------------------------------------------------------------------
echo "[start.sh] launching scripts/run_sft_qwen35_122b_12node_hade2.sh ..."
exec bash "$REPO_ROOT/scripts/run_sft_qwen35_122b_12node_hade2.sh"
