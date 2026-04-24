#!/usr/bin/env bash
# Install runtime-deferred dependencies (mamba-ssm / causal-conv1d / fla)
# inside the qwen35-mbridge container.
#
# WHY at runtime instead of in the image:
#   - mamba-ssm + causal-conv1d need nvcc compile (5-15 min, +400MB)
#   - mamba-ssm import touches CUDA driver -> fails inside `docker build` (no GPU)
#   - flash-linear-attention is pure-python but only useful with CUDA
#   - Allows version overrides without rebuilding the image
#
# Idempotent:
#   - Skips installation if all three packages import OK (fast no-op ~1s)
#   - Use --force to reinstall anyway
#
# Usage:
#   bash scripts/install_runtime_deps.sh           # install if missing
#   bash scripts/install_runtime_deps.sh --force   # reinstall
#   MAMBA_SSM_VERSION=2.4.0 bash scripts/install_runtime_deps.sh   # version pin

set -euo pipefail

VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
VENV_DIR="${VENV_DIR:-/opt/venv-mbridge}"

MAMBA_SSM_VERSION="${MAMBA_SSM_VERSION:-2.3.1}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.6.1}"
FLA_VERSION="${FLA_VERSION:-0.4.2}"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        --help|-h)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *) echo "[install_runtime_deps] unknown arg: $arg" ; exit 2 ;;
    esac
done

if [[ ! -x "$VENV_PY" ]]; then
    echo "[install_runtime_deps] FATAL: venv python not found at $VENV_PY"
    exit 1
fi

# ---- Quick check: are they already installed? ----
need_install=0
if [[ "$FORCE" -eq 1 ]]; then
    need_install=1
    echo "[install_runtime_deps] --force: will reinstall all three"
else
    for pkg in mamba_ssm causal_conv1d fla; do
        if ! "$VENV_PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$pkg') else 1)" 2>/dev/null; then
            echo "[install_runtime_deps] missing: $pkg"
            need_install=1
        fi
    done
fi

if [[ "$need_install" -eq 0 ]]; then
    echo "[install_runtime_deps] all three (mamba_ssm / causal_conv1d / fla) already installed; nothing to do."
    echo "[install_runtime_deps] (use --force to reinstall)"
    exit 0
fi

# ---- Verify GPU / nvcc available (these are nvcc compiles) ----
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[install_runtime_deps] FATAL: nvcc not on PATH; cannot compile mamba-ssm / causal-conv1d"
    exit 1
fi

# ---- Use uv if present, else fall back to pip ----
INSTALL_CMD=(
    "$VENV_PY" -m pip install --no-cache-dir
)
if command -v uv >/dev/null 2>&1; then
    INSTALL_CMD=(
        uv pip install --link-mode copy --python "$VENV_PY"
    )
fi

echo "[install_runtime_deps] installing flash-linear-attention==${FLA_VERSION} (pure python, ~30s)"
"${INSTALL_CMD[@]}" --no-deps "flash-linear-attention==${FLA_VERSION}"

echo "[install_runtime_deps] installing causal-conv1d==${CAUSAL_CONV1D_VERSION} (nvcc compile, ~3-8 min)"
"${INSTALL_CMD[@]}" --no-build-isolation --no-deps \
    "causal-conv1d==${CAUSAL_CONV1D_VERSION}"

echo "[install_runtime_deps] installing mamba-ssm==${MAMBA_SSM_VERSION} (nvcc compile, ~5-15 min)"
"${INSTALL_CMD[@]}" --no-build-isolation --no-deps \
    "mamba-ssm==${MAMBA_SSM_VERSION}"

# ---- Verify ----
echo "[install_runtime_deps] verifying imports (with GPU)..."
"$VENV_PY" - <<'PYEOF'
import sys
ok = True
for name in ("mamba_ssm", "causal_conv1d", "fla"):
    try:
        m = __import__(name)
        v = getattr(m, "__version__", "<no version>")
        print(f"  ok    {name:18s} {v}")
    except Exception as exc:
        print(f"  FAIL  {name:18s} {type(exc).__name__}: {exc}")
        ok = False
sys.exit(0 if ok else 1)
PYEOF

echo "[install_runtime_deps] done."
