#!/usr/bin/env bash
# Runtime sanity check inside the qwen35-mbridge container.
# Run this AFTER `docker run --gpus all qwen35-mbridge:cu129`.
# Verifies GPU-aware imports that build-time sanity could not check.
#
# Usage (inside container):
#   bash scripts/runtime_sanity.sh
#
# Usage (one-shot from host, no interactive shell):
#   docker run --rm --gpus all qwen35-mbridge:cu129 \
#     bash scripts/runtime_sanity.sh

set -uo pipefail

VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"

if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] venv python not found at $VENV_PY"
    echo "        (override with VENV_PY=/path/to/python)"
    exit 1
fi

"$VENV_PY" - <<'PYEOF'
import sys

print("=" * 60)
print("Runtime sanity (requires GPU access)")
print("=" * 60)

ok = True


def check(name: str, attr: str = "__version__", critical: bool = True) -> bool:
    global ok
    try:
        m = __import__(name)
        for part in name.split(".")[1:]:
            m = getattr(m, part)
        v = getattr(m, attr, "<no version>") if attr else "ok"
        print(f"  ok    {name:42s} {v}")
        return True
    except Exception as exc:
        label = "FAIL" if critical else "warn"
        print(f"  {label:5s} {name:42s} {type(exc).__name__}: {exc}")
        if critical:
            ok = False
        return False


# 1. CUDA driver / device basics
import torch
print(f"  ok    {'torch':42s} {torch.__version__}  (cuda built: {torch.version.cuda})")

if not torch.cuda.is_available():
    print("  FAIL  torch.cuda.is_available()=False — GPU not exposed to container")
    print("        run with `docker run --gpus all` or check nvidia-container-toolkit")
    sys.exit(1)

ndev = torch.cuda.device_count()
print(f"  ok    {'torch.cuda.device_count':42s} {ndev}")
print(f"  ok    {'torch.cuda.get_device_name(0)':42s} {torch.cuda.get_device_name(0)}")
print(f"  ok    {'torch.backends.cudnn.version':42s} {torch.backends.cudnn.version()}")

# 2. Runtime-deferred deps (installed by entrypoint or install_runtime_deps.sh).
# If missing, point the user at the install script.
deferred_missing = []
for name in ("mamba_ssm", "causal_conv1d", "fla"):
    if not check(name, critical=False):
        deferred_missing.append(name)

if deferred_missing:
    print()
    print("  WARN  runtime-deferred deps missing: " + ", ".join(deferred_missing))
    print("        run: bash scripts/install_runtime_deps.sh")
    print()

check("megatron")
check("megatron.core")
check("megatron.bridge")

# 3. Project-critical entry points
try:
    from megatron.bridge import AutoBridge  # noqa: F401
    print(f"  ok    {'megatron.bridge.AutoBridge':42s} importable")
except Exception as exc:
    print(f"  FAIL  AutoBridge: {type(exc).__name__}: {exc}")
    ok = False

try:
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge  # noqa: F401
    print(f"  ok    {'Qwen35VLMoEBridge':42s} importable")
except Exception as exc:
    print(f"  FAIL  Qwen35VLMoEBridge: {type(exc).__name__}: {exc}")
    ok = False

try:
    from megatron.bridge.recipes.qwen_vl import qwen35_vl_122b_a10b_sft_config  # noqa: F401
    print(f"  ok    {'qwen35_vl_122b_a10b_sft_config recipe':42s} importable")
except Exception as exc:
    print(f"  FAIL  recipe import: {type(exc).__name__}: {exc}")
    ok = False

# 4. APEX cuda ext (already verified at build time, but double-check at runtime)
try:
    import fused_weight_gradient_mlp_cuda  # noqa: F401
    print(f"  ok    {'APEX fused_weight_gradient_mlp_cuda':42s} available")
except ImportError as exc:
    print(f"  warn  APEX cuda ext NOT available: {exc}")
    print("        recipe will need MBRIDGE_PATCH_GRAD_FUSION=1 to bypass")

# 5. Tiny actual GPU op (smoke test)
try:
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    print(f"  ok    {'GPU bf16 matmul (64x64)':42s} {y.shape} dtype={y.dtype}")
except Exception as exc:
    print(f"  FAIL  GPU matmul: {type(exc).__name__}: {exc}")
    ok = False

print("=" * 60)
if not ok:
    print("FAIL: runtime sanity failed")
    sys.exit(1)
print("OK: image is fully ready for conversion / smoke / SFT")
PYEOF
