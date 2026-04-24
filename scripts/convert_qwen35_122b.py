#!/usr/bin/env python3
# Convert Qwen3.5-122B-A10B (HF) → Megatron mcore checkpoint.
#
# Workaround: AutoBridge.import_ckpt() defaults gradient_accumulation_fusion=True
# (because TE is installed), but ColumnParallelLinear hard-checks the APEX
# fused_weight_gradient_mlp_cuda extension. Apex is not installed in this venv,
# so we manually disable the fusion before finalize().
#
# Usage:
#   .venv/bin/python scripts/convert_qwen35_122b.py \
#     --hf-model /mnt/.../Qwen3.5-122B-A10B \
#     --megatron-path /data/temp/workspace/models/Qwen3.5-122B-A10B-mcore

import argparse
import os
import sys
from pathlib import Path

import torch

from megatron.bridge import AutoBridge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--megatron-path", required=True)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    print(f"🔄 Loading HF model: {args.hf_model}")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model,
        torch_dtype=dtype_map[args.torch_dtype],
        trust_remote_code=args.trust_remote_code,
    )

    print(f"🔧 Building Megatron provider")
    provider = bridge.to_megatron_provider(load_weights=True)

    # Conversion is a CPU-init, single-process workflow that never trains;
    # gradient_accumulation_fusion is irrelevant here. We always disable it
    # to avoid a runtime check in ColumnParallelLinear that requires APEX
    # (the check fires even though we never compute gradients).
    if hasattr(provider, "gradient_accumulation_fusion"):
        provider.gradient_accumulation_fusion = False
        print("   ⚙️  gradient_accumulation_fusion=False (irrelevant for CPU conversion)")

    # === Conversion-only: single-process, EP/PP/TP all = 1 (default) ===
    # The mcore ckpt layout is parallelism-agnostic for torch_dist.
    # We do NOT need to set EP=8 here; it's only for runtime.
    # However, for very large MoE models we may want EP > 1 to fit memory.
    # For 122B, on a single host with 2TB RAM and CPU init, this should fit.

    if hasattr(provider, "finalize"):
        provider.finalize()

    print("🏗️  Materialising Megatron model on CPU")
    model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
    )

    print(f"💾 Saving Megatron checkpoint → {args.megatron_path}")
    hf_tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    if args.trust_remote_code:
        hf_tokenizer_kwargs.setdefault("trust_remote_code", True)

    bridge.save_megatron_model(
        model,
        args.megatron_path,
        hf_tokenizer_path=args.hf_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        low_memory_save=True,
    )

    print(f"✅ Done. Checkpoint at: {args.megatron_path}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
