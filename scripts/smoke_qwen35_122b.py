#!/usr/bin/env python3
# Smoke test: load mcore Qwen3.5-122B-A10B and run a VLM forward+generate.
#
# Wraps examples/conversion/hf_to_megatron_generate_vlm.py with three patches
# that are toggleable via environment variables. In the NGC-based container
# (recommended) the defaults make all patches no-ops because NGC ships the
# correctly-configured TE / cuDNN / APEX. On bare-metal hosts (without APEX,
# with broken cuDNN, or with TE 2.7) export the corresponding env var to 1.
#
# Env toggles (default value for the NGC container in parens):
#   MBRIDGE_PATCH_NVIDIA_FILE = 0    Patch nvidia.__file__ (TE 2.7 namespace bug)
#   MBRIDGE_DISABLE_CUDNN     = 0    torch.backends.cudnn.enabled = False
#   MBRIDGE_PATCH_GRAD_FUSION = 0    Force gradient_accumulation_fusion = False
#
# Usage:
#   torchrun --nproc_per_node=8 scripts/smoke_qwen35_122b.py \
#     --hf_model_path /mnt/.../Qwen3.5-122B-A10B \
#     --megatron_model_path /data/temp/workspace/models/Qwen3.5-122B-A10B-mcore/iter_0000000 \
#     --image_path /data/temp/Megatron-Bridge/Repo-Mbridge.png \
#     --prompt "Describe this image." \
#     --tp 1 --pp 1 --ep 8 --max_new_tokens 32

import sys
import os


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


# === Patch 0: nvidia.__file__ (TE 2.7 namespace package crash) ===
# NGC ships TE 2.4 which does not have this bug; default off in container.
if _truthy("MBRIDGE_PATCH_NVIDIA_FILE"):
    import nvidia as _nvidia_ns

    if getattr(_nvidia_ns, "__file__", None) is None and getattr(_nvidia_ns, "__path__", None):
        _nvidia_ns.__file__ = os.path.join(list(_nvidia_ns.__path__)[0], "__init__.py")
        print(f"[smoke] patched nvidia.__file__ -> {_nvidia_ns.__file__}", flush=True)
    else:
        print("[smoke] nvidia.__file__ already set, no patch needed", flush=True)
else:
    print("[smoke] skipping nvidia.__file__ patch (set MBRIDGE_PATCH_NVIDIA_FILE=1 if TE crashes)", flush=True)

# === Patch 1: cuDNN disable (bare-metal sublib loading bug) ===
# NGC ships a healthy cuDNN 9.10.2; default off in container.
if _truthy("MBRIDGE_DISABLE_CUDNN"):
    import torch as _torch

    _torch.backends.cudnn.enabled = False
    print("[smoke] cuDNN disabled via MBRIDGE_DISABLE_CUDNN=1", flush=True)
else:
    print("[smoke] cuDNN left enabled (set MBRIDGE_DISABLE_CUDNN=1 to disable on broken hosts)", flush=True)

# === Patch 2: gradient_accumulation_fusion = False (no APEX) ===
# NGC ships APEX with fused_weight_gradient_mlp_cuda; default off in container.
if _truthy("MBRIDGE_PATCH_GRAD_FUSION"):
    from megatron.bridge.utils import fusions as _fusions

    _fusions.can_enable_gradient_accumulation_fusion = lambda: False
    print("[smoke] forced gradient_accumulation_fusion=False (no APEX)", flush=True)
else:
    print(
        "[smoke] grad-accum-fusion left to APEX detection (set MBRIDGE_PATCH_GRAD_FUSION=1 if APEX missing)",
        flush=True,
    )

# Compute paths
EXAMPLE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "examples", "conversion"))
EXAMPLE = os.path.join(EXAMPLE_DIR, "hf_to_megatron_generate_vlm.py")

# The example does `from vlm_generate_utils import (...)`, a sibling module.
# runpy.run_path does NOT add the script's directory to sys.path; do it manually.
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

# Now invoke the example via runpy so its argparse picks up our argv.
import runpy

print(f"[smoke] invoking {EXAMPLE} with argv={sys.argv}", flush=True)
runpy.run_path(EXAMPLE, run_name="__main__")
