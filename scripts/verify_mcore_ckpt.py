#!/usr/bin/env python3
# Offline verification of an mcore torch_dist checkpoint.
#
# Checks (single-process, no GPU, no NCCL):
#   1. Required files exist: latest_checkpointed_iteration.txt, iter_XXXXXXX/
#      __0_0.distcp, .metadata, metadata.json, run_config.yaml, common.pt, tokenizer/
#   2. File sizes plausible (distcp >= 100GB for 122B bf16)
#   3. run_config.yaml parses; key model fields present and sensible
#   4. .metadata is a torch.distributed.checkpoint metadata file (loadable)
#   5. Sampled sharded tensors in metadata cover expected shapes: lm head / embed /
#      expert MLPs / GDN / MTP / vision
#
# Usage:
#   .venv/bin/python scripts/verify_mcore_ckpt.py /data/temp/workspace/models/Qwen3.5-122B-A10B-mcore

import sys
import os
import json
from pathlib import Path


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def check_files(root: Path) -> dict:
    results = {}
    iter_file = root / "latest_checkpointed_iteration.txt"
    assert iter_file.is_file(), f"missing {iter_file}"
    iteration = iter_file.read_text().strip()
    results["iteration"] = iteration
    iter_dir = root / f"iter_{int(iteration):07d}"
    assert iter_dir.is_dir(), f"missing {iter_dir}"
    results["iter_dir"] = str(iter_dir)

    required = ["__0_0.distcp", ".metadata", "metadata.json", "run_config.yaml", "common.pt", "tokenizer"]
    for rel in required:
        p = iter_dir / rel
        assert p.exists(), f"missing {p}"
        if p.is_file():
            results[rel] = {"size": p.stat().st_size, "human": _human(p.stat().st_size)}
        else:
            results[rel] = {"dir": True, "entries": len(list(p.iterdir()))}
    return results, iter_dir


def check_run_config(iter_dir: Path) -> dict:
    import yaml

    cfg = yaml.safe_load((iter_dir / "run_config.yaml").read_text())
    model = cfg["model"]
    required = {
        "_target_": "megatron.bridge.models.qwen_vl.qwen35_vl_provider.Qwen35VLMoEModelProvider",
        "num_layers": 48,
        "hidden_size": 3072,
        "num_attention_heads": 32,
        "kv_channels": 256,
        "num_moe_experts": 256,
        "moe_router_topk": 8,
        "moe_ffn_hidden_size": 1024,
        "mtp_num_layers": 1,
        "mtp_loss_scaling_factor": 0.1,
        "gradient_accumulation_fusion": False,
    }
    mismatches = {}
    for k, expected in required.items():
        actual = model.get(k)
        if actual != expected:
            mismatches[k] = {"expected": expected, "actual": actual}
    has_vision = "vision_config" in str(cfg) or any("vision" in str(v).lower() for v in model.values())
    return {
        "mismatches": mismatches,
        "has_vision": has_vision,
        "num_params_hint": model.get("num_layers", 0) * model.get("hidden_size", 0),
    }


def check_metadata(iter_dir: Path) -> dict:
    """Load .metadata and sample key tensors by name keyword."""
    import torch
    from torch.distributed.checkpoint.metadata import Metadata
    from torch.distributed.checkpoint import FileSystemReader

    reader = FileSystemReader(str(iter_dir))
    md: Metadata = reader.read_metadata()
    sd_md = md.state_dict_metadata
    all_keys = list(sd_md.keys())

    # Buckets we expect to be present
    buckets = {
        "vision/ViT": [k for k in all_keys if "vision" in k.lower()],
        "embedding": [k for k in all_keys if "embed" in k.lower()],
        "output_layer": [k for k in all_keys if "output_layer" in k.lower()],
        "expert_mlp": [
            k
            for k in all_keys
            if "experts" in k.lower()
            and ("weight1" in k.lower() or "weight2" in k.lower() or "linear_fc" in k.lower())
        ],
        # GDN params live under language_model.decoder.layers.<L>.self_attention.* but
        # carry signature attributes from Gated DeltaNet (A_log, dt_bias, conv1d,
        # in_proj.weight.{alpha,beta,z}).
        "gdn/linear_attn": [
            k
            for k in all_keys
            if "language_model" in k.lower()
            and (
                k.endswith(".A_log")
                or k.endswith(".dt_bias")
                or "conv1d.weight" in k.lower()
                or k.endswith("in_proj.weight.alpha")
                or k.endswith("in_proj.weight.beta")
                or k.endswith("in_proj.weight.z")
            )
        ],
        # Standard scaled-dot-product attention layers (the 12 "full_attention"
        # layers in language_model + ViT self-attention)
        "full_attention": [
            k
            for k in all_keys
            if "self_attention" in k.lower() and ("linear_qkv" in k.lower() or "linear_proj" in k.lower())
        ],
        "mtp": [k for k in all_keys if "mtp" in k.lower()],
        "shared_expert": [k for k in all_keys if "shared_expert" in k.lower()],
    }

    sizes = {}
    for name, keys in buckets.items():
        n = len(keys)
        sample = keys[:3]
        sizes[name] = {"count": n, "samples": sample}

    return {"total_keys": len(all_keys), "buckets": sizes}


def main():
    if len(sys.argv) != 2:
        print("Usage: verify_mcore_ckpt.py <mcore_root>")
        return 1
    root = Path(sys.argv[1]).resolve()
    print(f"\n=== Verifying: {root} ===\n")

    print("[1/3] File presence + sizes")
    files_info, iter_dir = check_files(root)
    for k, v in files_info.items():
        print(f"   {k}: {v}")
    distcp_size = files_info["__0_0.distcp"]["size"]
    assert distcp_size > 100 * 1024**3, f"distcp file suspiciously small: {_human(distcp_size)}"
    print(f"   ✅ distcp size OK: {files_info['__0_0.distcp']['human']}")

    print("\n[2/3] run_config.yaml sanity")
    rc = check_run_config(iter_dir)
    if rc["mismatches"]:
        print(f"   ❌ Mismatches: {json.dumps(rc['mismatches'], indent=2)}")
        return 2
    print("   ✅ All required model fields match expected values")
    print(f"   has_vision mentions: {rc['has_vision']}")

    print("\n[3/3] dist_checkpointing metadata (tensor catalogue)")
    md = check_metadata(iter_dir)
    print(f"   total_keys in state_dict metadata: {md['total_keys']}")
    for name, info in md["buckets"].items():
        marker = "✅" if info["count"] > 0 else "❌"
        print(f"   {marker} {name:22s} count={info['count']:6d}  e.g. {info['samples'][:1]}")

    # Hard requirements for this specific model
    must_have = {
        "vision/ViT": 1,  # has ViT params
        "embedding": 1,  # has word embeddings
        "output_layer": 1,  # has LM head
        "expert_mlp": 10,  # MoE experts (256 × many params, expect many)
        "gdn/linear_attn": 1,  # GDN layers
        "mtp": 1,  # MTP layer
    }
    failures = []
    for k, min_n in must_have.items():
        if md["buckets"][k]["count"] < min_n:
            failures.append(f"{k} has only {md['buckets'][k]['count']} keys (expected >= {min_n})")
    if failures:
        print("\n❌ HARD CHECK FAILURES:")
        for f in failures:
            print(f"   - {f}")
        return 3
    print("\n✅ All hard structural checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
