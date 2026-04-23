#!/usr/bin/env python3
# Smoke test: load mcore Qwen3.5-122B-A10B and run a VLM forward+generate.
#
# Wraps examples/conversion/hf_to_megatron_generate_vlm.py with two patches:
#   1. Force gradient_accumulation_fusion=False (no APEX in this venv)
#   2. (optional) extra logging hooks
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

# === Patch 0 ===
# TransformerEngine 2.7 assumes `nvidia.__file__` exists, but `nvidia` is a
# PEP 420 namespace package (no __init__.py) so __file__ is None.
# Synthesize one so TE's _nvidia_cudart_include_dir() does not crash.
import nvidia as _nvidia_ns

if getattr(_nvidia_ns, "__file__", None) is None and getattr(_nvidia_ns, "__path__", None):
    _nvidia_ns.__file__ = os.path.join(list(_nvidia_ns.__path__)[0], "__init__.py")
    print(f"[smoke] patched nvidia.__file__ -> {_nvidia_ns.__file__}", flush=True)

# === Patch 0.5 ===
# Disable cuDNN entirely on this host: F.conv3d in the ViT patch_embed triggers
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED (broken cuDNN runtime install on the
# system; even after pinning nvidia-cudnn-cu12 to match the system version, the
# sublibraries fail to dlopen). Falling back to native PyTorch conv is correct
# and ~2x slower for the ViT, but acceptable for smoke + SFT (ViT is a tiny
# fraction of the 122B model).
import torch as _torch

_torch.backends.cudnn.enabled = False
print("[smoke] patched torch.backends.cudnn.enabled = False (sublib load failed)", flush=True)

# === Patch 1 ===
# Patch megatron-bridge before it is imported by the example.
# The provider class reads can_enable_gradient_accumulation_fusion() at construct
# time; force it to False so ColumnParallelLinear does not require APEX.
from megatron.bridge.utils import fusions as _fusions

_orig = _fusions.can_enable_gradient_accumulation_fusion


def _force_false():
    return False


_fusions.can_enable_gradient_accumulation_fusion = _force_false
print(
    f"[smoke] patched can_enable_gradient_accumulation_fusion -> False (was {_orig.__module__}.{_orig.__name__})",
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
