#!/usr/bin/env python3
# SFT entry point for Qwen3.5-122B-A10B with megatron-bridge.
#
# Wraps scripts/training/run_recipe.py with the exact patches required by this
# environment (see /data/temp/Megatron-LM/memory.md):
#   - nvidia.__file__ patch (TE 2.7 namespace package bug)
#   - gradient_accumulation_fusion = False (no APEX)
#   - torch.backends.cudnn.enabled = False (cuDNN sublib loading broken on host)
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
#   - gradient_accumulation_fusion=False (override; recipe default True)

import os
import sys

# === Patch 0: TE 2.7 expects nvidia.__file__ to be a real path ===
import nvidia as _nvidia_ns

if getattr(_nvidia_ns, "__file__", None) is None and getattr(_nvidia_ns, "__path__", None):
    _nvidia_ns.__file__ = os.path.join(list(_nvidia_ns.__path__)[0], "__init__.py")
    print(f"[sft] patched nvidia.__file__ -> {_nvidia_ns.__file__}", flush=True)

# === Patch 0.5: cuDNN sublib loading broken; fall back to native conv ===
import torch as _torch

_torch.backends.cudnn.enabled = False
print("[sft] patched torch.backends.cudnn.enabled = False", flush=True)

# === Patch 1: gradient_accumulation_fusion requires APEX (not installed) ===
from megatron.bridge.utils import fusions as _fusions

_fusions.can_enable_gradient_accumulation_fusion = lambda: False
print("[sft] patched can_enable_gradient_accumulation_fusion -> False", flush=True)

# Ensure the recipe override below also disables fusion. The recipe sets
# `cfg.model.gradient_accumulation_fusion = True` after building the provider;
# we will pass `model.gradient_accumulation_fusion=false` via CLI overrides.

# Now hand off to the official entry point. It uses argparse so the rest of our
# argv is consumed there.
EXAMPLE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "scripts", "training"))
EXAMPLE = os.path.join(EXAMPLE_DIR, "run_recipe.py")
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

import runpy

print(f"[sft] invoking {EXAMPLE} with argv={sys.argv}", flush=True)
runpy.run_path(EXAMPLE, run_name="__main__")
