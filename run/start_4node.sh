#!/usr/bin/env bash
# Per-node bootstrap + SFT launcher for Qwen3.5-122B-A10B (4-node × 8 GPU FULL SFT).
#
# Image assumption: qwen35-mbridge:cu129 already has causal_conv1d + mamba_ssm
# (with patched __init__.py) + fla baked in. No runtime install needed —
# the deps check below will find them and skip pip.
#
# Run on EACH of the 4 nodes (after entering the container):
#
#   bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start_4node.sh
#
# Cluster injects MASTER_ADDR / WORLD_SIZE / RANK already.
# If your scheduler did NOT inject RANK, prepend it manually:
#
#   RANK=0 MASTER_PORT=23456 bash start_4node.sh   # master node
#   RANK=1 MASTER_PORT=23456 bash start_4node.sh   # worker 1
#   RANK=2 MASTER_PORT=23456 bash start_4node.sh   # worker 2
#   RANK=3 MASTER_PORT=23456 bash start_4node.sh   # worker 3
#
# IMPORTANT: this cluster's true default MASTER_PORT is 23456, but the
# master container sometimes inherits a stale value (e.g. 23964) while the
# worker sees the correct one. If they mismatch, torchrun rendezvous hangs
# silently forever with no error. ALWAYS pass MASTER_PORT=23456 explicitly.
# See memory.md Pitfall #11.
#
# Recipe: qwen35_vl_122b_a10b_sft_config (full SFT, no LoRA).
# Default parallelism in the launcher script is TP=2 PP=4 EP=8 (DP=2),
# which fits 32 × L20Y 80G with full activation recompute — tight but
# doable. Override TP/PP/EP/MBS/SEQ/ITERS/GBS via env if needed.

set -euo pipefail

REPO_ROOT="/mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge"
WHEEL_DIR="/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/wheels"
VENV_PY="/opt/venv-mbridge/bin/python"

CAUSAL_WHL="${WHEEL_DIR}/causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
MAMBA_WHL="${WHEEL_DIR}/mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# ---------------------------------------------------------------------------
# 0. Sanity
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV_PY" ]]; then
    echo "[start_4node] FATAL: $VENV_PY not found. Are you inside the qwen35-mbridge container?" >&2
    exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
    echo "[start_4node] FATAL: repo root not found: $REPO_ROOT" >&2
    exit 1
fi

cd "$REPO_ROOT"
echo "[start_4node] node-local hostname=$(hostname)  RANK=${RANK:-?}  WORLD_SIZE=${WORLD_SIZE:-?}  MASTER=${MASTER_ADDR:-?}:${MASTER_PORT:-?}"

# ---------------------------------------------------------------------------
# 1. Install runtime-deferred deps from prebuilt wheels (idempotent)
#    With the new image these are already baked in, so this is a no-op.
# ---------------------------------------------------------------------------
need_causal=0
need_mamba=0
need_fla=0
"$VENV_PY" -c 'import causal_conv1d' >/dev/null 2>&1 || need_causal=1
"$VENV_PY" -c 'import mamba_ssm' >/dev/null 2>&1 || need_mamba=1
"$VENV_PY" -c 'import fla' >/dev/null 2>&1 || need_fla=1

if (( need_causal || need_mamba || need_fla )); then
    echo "[start_4node] installing missing runtime deps:"
    (( need_causal )) && echo "  - causal_conv1d (wheel)"
    (( need_mamba ))  && echo "  - mamba_ssm     (wheel + __init__.py patch)"
    (( need_fla ))    && echo "  - fla           (pure python)"

    install_args=()
    if (( need_causal )); then
        [[ -f "$CAUSAL_WHL" ]] || { echo "[start_4node] FATAL: missing wheel $CAUSAL_WHL" >&2; exit 1; }
        install_args+=("$CAUSAL_WHL")
    fi
    if (( need_mamba )); then
        [[ -f "$MAMBA_WHL" ]] || { echo "[start_4node] FATAL: missing wheel $MAMBA_WHL" >&2; exit 1; }
        install_args+=("$MAMBA_WHL")
    fi
    if (( ${#install_args[@]} > 0 )); then
        "$VENV_PY" -m pip install --no-cache-dir --no-deps "${install_args[@]}"
    fi
    if (( need_fla )); then
        "$VENV_PY" -m pip install --no-cache-dir --no-deps flash-linear-attention==0.4.2
    fi
else
    echo "[start_4node] runtime deps already installed; skipping pip."
fi

# ---------------------------------------------------------------------------
# 2. Patch mamba_ssm/__init__.py (idempotent — no-op on baked image)
# ---------------------------------------------------------------------------
SITE_PACKAGES="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
MAMBA_INIT="${SITE_PACKAGES}/mamba_ssm/__init__.py"
if [[ ! -f "$MAMBA_INIT" ]]; then
    echo "[start_4node] FATAL: $MAMBA_INIT does not exist after pip install" >&2
    exit 1
fi
if grep -q "NGC nv25.06 patched torch" "$MAMBA_INIT" 2>/dev/null; then
    echo "[start_4node] mamba_ssm/__init__.py already patched."
else
    echo "[start_4node] patching $MAMBA_INIT"
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
# not break `import mamba_ssm` if they don't. Catch BaseException so
# RuntimeError from Triton autotune (no-GPU host) is also tolerated.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn  # noqa: F401
except BaseException as _e:
    selective_scan_fn = None
    mamba_inner_fn = None
    import warnings
    warnings.warn(
        "mamba_ssm.selective_scan_cuda failed to load "
        "(NGC torch ABI mismatch, or no-GPU host); "
        "selective_scan_fn / mamba_inner_fn / Mamba1 will not work, but the "
        "Triton-based mamba_ssm.ops.triton.ssd_combined path used by "
        "megatron-core (Mamba2 / GDN) is unaffected. Original error: " + str(_e)
    )

try:
    from mamba_ssm.modules.mamba_simple import Mamba  # noqa: F401
except BaseException:
    Mamba = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2  # noqa: F401
except BaseException:
    Mamba2 = None

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: F401
except BaseException:
    MambaLMHeadModel = None
PYEOF
    echo "[start_4node] patch applied."
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
" || { echo "[start_4node] FATAL: verification failed — runtime deps still broken" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 4. Launch FULL SFT (4 nodes × 8 GPU = 32 GPU, TP=2 PP=4 EP=8 default)
# ---------------------------------------------------------------------------
echo "[start_4node] launching scripts/run_sft_qwen35_122b_4node.sh ..."
exec bash "$REPO_ROOT/scripts/run_sft_qwen35_122b_4node.sh"
