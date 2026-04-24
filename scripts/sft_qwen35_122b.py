#!/usr/bin/env python3
# SFT entry point for Qwen3.5-122B-A10B with megatron-bridge.
#
# Wraps scripts/training/run_recipe.py with three optional patches that are
# toggleable via environment variables. In the NGC-based container all
# defaults are no-ops because NGC ships TE 2.4 + cuDNN 9.10 + APEX cuda ext.
#
# Env toggles (default value for the NGC container in parens):
#   MBRIDGE_PATCH_NVIDIA_FILE = 0    Patch nvidia.__file__ (TE 2.7 namespace bug)
#   MBRIDGE_DISABLE_CUDNN     = 0    torch.backends.cudnn.enabled = False
#   MBRIDGE_PATCH_GRAD_FUSION = 0    Force gradient_accumulation_fusion = False
#
# When NGC ships APEX (the default for 25.06+), the recipe's default
# `gradient_accumulation_fusion=True` works correctly and you should NOT
# pass `model.gradient_accumulation_fusion=false` on the CLI.
#
# Usage (single-node 8 GPU smoke):
#   bash scripts/run_sft_qwen35_122b.sh smoke
#
# Usage (multi-node, 4 nodes × 8 GPU per the recipe default TP=2 PP=6 EP=8):
#   bash scripts/run_sft_qwen35_122b.sh full
#
# Recipe defaults that we use:
#   - qwen35_vl_122b_a10b_sft_config (TP=2 PP=6 EP=8, GBS=36, seq=4096, LR=2e-5)
#   - dataset = vlm-hf with cord_v2 (small public OCR dataset, fits in HF cache)
#   - MTP enabled (mtp_num_layers=1, mtp_loss_scaling_factor=0.1)
#   - pack_sequences_in_batch=False (REQUIRED — GDN does not support THD)

import os
import sys


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


# === Patch 0: nvidia.__file__ (TE 2.7 namespace package crash) ===
if _truthy("MBRIDGE_PATCH_NVIDIA_FILE"):
    import nvidia as _nvidia_ns

    if getattr(_nvidia_ns, "__file__", None) is None and getattr(_nvidia_ns, "__path__", None):
        _nvidia_ns.__file__ = os.path.join(list(_nvidia_ns.__path__)[0], "__init__.py")
        print(f"[sft] patched nvidia.__file__ -> {_nvidia_ns.__file__}", flush=True)

# === Patch 1: cuDNN disable (bare-metal sublib loading bug) ===
if _truthy("MBRIDGE_DISABLE_CUDNN"):
    import torch as _torch

    _torch.backends.cudnn.enabled = False
    print("[sft] cuDNN disabled via MBRIDGE_DISABLE_CUDNN=1", flush=True)

# === Patch 2: gradient_accumulation_fusion = False (no APEX) ===
if _truthy("MBRIDGE_PATCH_GRAD_FUSION"):
    from megatron.bridge.utils import fusions as _fusions

    _fusions.can_enable_gradient_accumulation_fusion = lambda: False
    print("[sft] forced gradient_accumulation_fusion=False (no APEX)", flush=True)

# Now hand off to the official entry point. It uses argparse so the rest of our
# argv is consumed there.
EXAMPLE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "scripts", "training"))
EXAMPLE = os.path.join(EXAMPLE_DIR, "run_recipe.py")
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

import runpy

print(f"[sft] invoking {EXAMPLE} with argv={sys.argv}", flush=True)
runpy.run_path(EXAMPLE, run_name="__main__")
