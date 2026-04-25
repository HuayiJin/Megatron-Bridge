# memory.md — Qwen3.5-122B-A10B Multimodal SFT

> **Project:** Run multimodal (image+text) SFT for **Qwen3.5-122B-A10B** on
> NVIDIA L20Y/H800 80G clusters via **NVIDIA NeMo Megatron-Bridge**
> (`megatron.bridge`, *not* `mbridge`).
>
> This file is the **single source of operational truth** for this task.
> Historical decisions and superseded configs live in `memory_legacy.md`;
> this file only carries facts active for the current environment.
>
> All paths and recipes here reflect the **2026-04-25 redesign**:
> pure-environment Docker image + all source code on NAS under `/mnt`.

## Boundaries

**NEVER:**
- Modify `3rdparty/Megatron-LM/` — submodule pinned to upstream
- Enable sequence packing (`pack_sequences_in_batch=true`) on Qwen3.5-VL —
  GDN/linear-attention only supports BSHD, *not* THD
- Use a `.venv` outside `/opt/venv-mbridge` — the NGC container's torch /
  TE / cuDNN / APEX / flash-attn are all under that prefix
- Pass `model.gradient_accumulation_fusion=false` inside the NGC container
  — APEX cuda ext is present and the recipe default `True` is correct
- Save `/data/temp/...` paths into scripts — that prefix only existed on the
  old bare-metal host; on the cluster use NAS paths under
  `/mnt/tidal-alsh01/dataset/redone/hade/...`
- Set `moe_router_fusion=True` in NGC 25.06 (TE 2.4) —
  `transformer_engine.pytorch.router` does not exist until TE ≥ 2.7; enabling
  it raises `ValueError: fused_topk_with_score_function is not available`.
  Use `False` (< 0.01% FLOP overhead). See Pitfall #14.
- `pip install --upgrade transformer-engine` inside the NGC container — TE is
  tightly ABI-coupled to NGC's patched torch 2.8a; a PyPI TE wheel will break
  fused_permute / FP8 / flash-attn interfaces. Upgrade TE only by switching to
  a new NGC base image that ships TE ≥ 2.7.
- Bake Megatron-Bridge source into the Docker image — the image is a **pure
  environment image** (OS + venv + wheels only). All source code lives on NAS
  under `/mnt` and is executed directly from there. See §Container & Toolchain.
- Run code from inside the container's own filesystem (e.g. any `/opt/...`
  path) — always `cd` into the `/mnt` source tree and run from there.
- Mount or pre-install Megatron-Bridge inside the container at build time —
  `install_runtime_deps.sh` handles `uv sync` + editable install at runtime
  from the `/mnt` tree. See §Per-node startup sequence.

**ASK FIRST:**
- Before changing recipe selection (LoRA vs full SFT vs PEFT scheme)
- Before changing parallelism beyond `TP × PP × EP × DP = world_size`
- Before introducing a new dataset format (current pipeline expects
  `PreloadedVLMConversationProvider` JSONL)

**ALWAYS:**
- Use `/opt/venv-mbridge/bin/python` as the interpreter
- Use the **current year (2026)** in any file you write
- Run `bash scripts/install_runtime_deps.sh` from the `/mnt` source tree
  **once per container launch** before starting training — this does
  `uv sync` (megatron-core) + `uv pip install -e .` (megatron-bridge)
- Update this file whenever a decision is made with reasoning behind it

## Hardware (active layout, 2026-04-24)

| Slot | Value |
|------|-------|
| Cluster | 2 nodes available now, 4 nodes available later |
| GPU per node | 8 × NVIDIA L20Y 80G (special-edition H800) |
| Driver | 570.148.08 (CUDA cap 12.9) |
| Interconnect | mlx5_bond IB (NCCL_IB_HCA=mlx5_bond, NCCL_SOCKET_IFNAME=bond1) |
| Cluster env vars | `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `NCCL_IB_*` injected by scheduler |

## Resources

| Resource | Path | Size |
|----------|------|------|
| HF weights (122B) | `/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B` | 39 safetensors (~244 GB) |
| Mcore checkpoint | `/mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore` | 234 GB (`iter_0000000/__0_0.distcp` + `tokenizer/` + `run_config.yaml`) |
| Demo train JSONL | `/mnt/tidal-alsh01/dataset/redone/hade/dd/train_data_demo.jsonl` | 14 records, multi-turn text-only chat |
| Cooked demo JSONL (text + 2 image samples) | `/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/demo_data/train_demo.jsonl` | 16 records |
| Output / runs / logs root | `/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/` | per-run subdirs |
| Megatron-Bridge source | `/mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge` | on NAS; run directly from here |

## Container & Toolchain

Image: built from `Dockerfile.qwen35`, base `nvcr.io/nvidia/pytorch:25.06-py3`
(CUDA 12.9.1 / torch 2.8.0a / TE 2.4 / cuDNN 9.10 / NCCL 2.27 / APEX cuda ext / flash-attn 2.7.4).

### Image design (decided 2026-04-25)

The image is a **pure environment image**. It contains only:
- OS tools (git, tmux, curl, …)
- uv package manager
- `/opt/venv-mbridge`: venv inheriting all NGC site-packages, plus
  causal-conv1d 1.6.1 + mamba-ssm 2.3.1 (prebuilt wheels, ABI-patched)
  and flash-linear-attention 0.4.2

**What the image does NOT contain:** Megatron-Bridge source, megatron-core,
any editable install. All business code lives on NAS under `/mnt`.

**Why:** eliminates stale-code problems — code changes on `/mnt` are
immediately effective without rebuilding the image. Image rebuild is only
needed when system packages or wheels change.

### Image build (run on build host)

```bash
# Only the two prebuilt wheels are needed — NOT the full source tree
cd /path/to/Megatron-Bridge
mkdir -p docker/runtime_wheels && cd docker/runtime_wheels
curl -fL -o causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
curl -fL -o mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
cd ../..
docker build -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 .
```

Wheels are NOT git-tracked (~780 MB). Build fails fast if they are missing.

### Per-node startup sequence (every container launch)

```bash
# Step 1 — start container, mount the NAS
docker run --rm -it --gpus all --shm-size=64g --ulimit memlock=-1 \
  --network host \
  -v /mnt/tidal-alsh01:/mnt/tidal-alsh01 \
  qwen35-mbridge:cu129 bash

# Step 2 — inside container: install megatron-core + megatron-bridge (~1 min, idempotent)
cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge
bash scripts/install_runtime_deps.sh
# This script does:
#   1. git submodule update --init (if 3rdparty/Megatron-LM is empty)
#   2. uv sync  — installs megatron-core (editable) + all pyproject.toml deps
#   3. uv pip install -e .  — installs megatron-bridge pointing at /mnt
#   4. verifies mamba_ssm / causal_conv1d / fla (already in image, fast check)
#   5. prints megatron.bridge.__file__ → must show /mnt path, not /opt

# Step 3 — launch training on each node
RANK=<0|1> MASTER_PORT=23456 bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start.sh
```

### Quick reference

| Action | Command |
|--------|---------|
| Interpreter | `/opt/venv-mbridge/bin/python` |
| Runtime setup (once per container) | `cd /mnt/.../Megatron-Bridge && bash scripts/install_runtime_deps.sh` |
| 2-node × 8-GPU LoRA SFT | `RANK=<0\|1> MASTER_PORT=23456 bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start.sh` |
| 4-node × 8-GPU full SFT | `RANK=<0\|1\|2\|3> MASTER_PORT=23456 bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start_4node.sh` |
| Underlying SFT scripts | `scripts/run_sft_qwen35_122b_2node_lora.sh` / `scripts/run_sft_qwen35_122b_4node.sh` |
| Verify mcore ckpt (no GPU needed) | `/opt/venv-mbridge/bin/python scripts/verify_mcore_ckpt.py /mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore` |
| Operator runbook | `docs/qwen35_122b_sft_runbook.md` |

### Manual multi-node launch — canonical recipe

```bash
# On EACH node, inside the container, after install_runtime_deps.sh:
cd /mnt/tidal-alsh01/dataset/redone/hade/dd

# Node 0 (master):
tmux new -s sft -d
tmux send-keys -t sft "RANK=0 MASTER_PORT=23456 bash start.sh" C-m

# Node 1 (worker):
tmux new -s sft -d
tmux send-keys -t sft "RANK=1 MASTER_PORT=23456 bash start.sh" C-m
```

**Always pass `MASTER_PORT=23456` explicitly** — the master's container
sometimes inherits a stale value (`23964`) while the worker sees `23456`.
Mismatch causes torchrun rendezvous to hang silently forever. See Pitfall #11.

`MASTER_ADDR`, `WORLD_SIZE` come from the cluster scheduler.
`RANK` you set per node (master = `RANK=0`).

Inside-container env vars:

| Variable | Default | What it does |
|----------|---------|--------------|
| `MBRIDGE_DISABLE_CUDNN` | `0` | NGC cuDNN 9.10 is healthy; set `1` only on bare-metal with sublib bug |

## Standard Recipes (table)

| Recipe | Mode | Min GPUs | TP / PP / EP | LR | GBS | Notes |
|--------|------|----------|--------------|-----|-----|-------|
| `qwen35_vl_122b_a10b_peft_config` | LoRA | **16** (2 node × 8) | 2 / 1 / 8 | 2e-4 | 36 | **Current demo target.** Fits cleanly on L20Y 80G. |
| `qwen35_vl_122b_a10b_sft_config` | Full SFT | 32 nominal (4 node × 8) | 2 / 6 / 8 default | 2e-5 | 36 | Default math is 2×6×8=96 GPU; on 32 GPU override to TP=2 PP=4 EP=8 + DP=2. |
| `qwen35_vl_35b_a3b_peft_config` | LoRA | 8 (1 node) | 2 / 1 / 4 | 2e-4 | — | Fast iteration on smaller MoE sibling. |
| `qwen35_vl_122b_a10b_pretrain_mock_config` | Pretrain (mock) | 8+ | — | — | — | Sanity test only; uses random tokens. |

## Step Function & Dataset Wiring

For VLM SFT use:

- `--step_func vlm_step` (defined in `src/megatron/bridge/training/vlm_step.py`)
- `--dataset vlm-preloaded` → resolves to `PreloadedVLMConversationProvider`
  (`src/megatron/bridge/data/vlm_datasets/preloaded_provider.py`)
- Provider expects records of one of:
  - `{"conversation": [{"role":..., "content":[{"type":"text"|"image", ...}, ...]}, ...]}` (passthrough)
  - `{"messages": [{"role": "user", "content": "<image>\nq?"}, ...], "images": ["/abs/path.png"]}` (legacy)
  - LLaVA `{"conversations": [{"from": "human", "value": "..."}, ...]}` (legacy)

`dataset.image_folder` is optional; if set, relative paths in `images` are
resolved against it. Absolute paths and `http(s)://` / `file://` URLs are
left alone.

## Known Pitfalls (active for this environment)

| # | Pitfall | Fix |
|---|---------|-----|
| 1 | `mamba_ssm` / `causal_conv1d` / `fla` missing after container start | These are **baked into the image** (`/opt/venv-mbridge`). If missing, the image was built from an old Dockerfile. Rebuild image with current `Dockerfile.qwen35`. Do NOT source-build inside the container (30+ min, see §Mamba-SSM ABI Workaround). |
| 2 | HF Hub unauthenticated request warning | Always pass `--hf_path /mnt/.../Qwen3.5-122B-A10B`. Set `HF_TOKEN` to silence. |
| 3 | Recipe defaults need more GPUs than available (e.g. SFT default TP=2 PP=6 EP=8 needs 96 GPU) | Override `model.tensor_model_parallel_size` / `pipeline_model_parallel_size` / `expert_model_parallel_size` on CLI. The 4-node script does this for you. |
| 4 | Loss explodes / OOM on full 122B SFT in 16 GPU | Use the LoRA recipe — full SFT requires ≥4 nodes. |
| 5 | NCCL hang after first step on multi-node | Confirm `MASTER_ADDR` resolves on every node and `NCCL_SOCKET_IFNAME=bond1`. Cluster injects `NCCL_IB_*` already. |
| 6 | "Sending unauthenticated requests to the HF Hub" | Cosmetic warning; safe to ignore for local-path workflow. |
| 7 | `pack_sequences_in_batch=true` on Qwen3.5-VL | Crashes inside GDN kernel. Recipe default is `False`; demo scripts pass it explicitly. |
| 8 | `gradient_accumulation_fusion=False` left over from old bare-metal scripts | Inside NGC container, this *hurts* performance (APEX is fine). Don't override it. |
| 9 | `megatron.bridge` resolves to `/opt/...` instead of `/mnt/...` after container start | `install_runtime_deps.sh` was not run, or ran from wrong directory. Run `cd /mnt/.../Megatron-Bridge && bash scripts/install_runtime_deps.sh`. The script prints `megatron.bridge → <path>` as verification; it must show the `/mnt` path. |
| 10 | NGC nv25.06 torch ABI is between stock 2.9 and 2.10 | NVIDIA backported some `c10::cuda::*` symbols. No stock `mamba_ssm` prebuilt wheel loads cleanly. Workaround: install the `cu12torch2.8` wheel + patch `mamba_ssm/__init__.py` (baked into image). See §Mamba-SSM ABI Workaround. |
| 11 | `MASTER_PORT` differs across nodes (master saw `23964`, worker saw `23456`) | Rendezvous hangs silently. Always pass `MASTER_PORT=23456` explicitly. Symptom: torchrun alive, 65 threads, sleeping; no worker fork. |
| 12 | torchrun gives no output for 6–15 min after the OMP banner | NORMAL. Sequence: (a) rendezvous ~10s, (b) `import megatron.bridge` 60–180s, (c) 234 GB mcore ckpt load 3–10 min, (d) iter 1 cold compile 1–3 min. Watch `nvidia-smi` GPU memory rising as liveness signal. |
| 13 | `import mamba_ssm` crashes inside `docker build` with `RuntimeError: 0 active drivers` | Build host has no GPU. `@triton.autotune(...)` fires at module load and calls `driver.active.get_benchmarker()`. **Fix:** `Dockerfile.qwen35` build-time sanity uses pure shell (`test -f` / `grep`) — no Python imports. See Dockerfile "Build-time sanity" comment. |
| 14 | `ValueError: fused_topk_with_score_function is not available. Please install TE >= 2.6.0.` | **Root cause:** `moe_router_fusion=True` in `_qwen35_vl_apply_moe()`. `transformer_engine.pytorch.router` does not exist in TE 2.4; `fused_topk_with_score_function` is `None`. **Analysis:** router topk is < 0.01% of MoE FLOPS (256 experts × topk=8 → ~4K ops vs ~50M ops expert GEMM); unfused path is negligible. Upgrading TE inside NGC breaks ABI (same class as mamba_ssm, Pitfall #10). **Fix (2026-04-25):** `cfg.model.moe_router_fusion = False` in `src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py`. `moe_permute_fusion` and `moe_grouped_gemm` remain `True`. Re-enable only after switching to NGC image with TE ≥ 2.7. |
| 15 | Training uses stale `/opt/Megatron-Bridge` code instead of `/mnt` source | Old Dockerfile baked source into image; entrypoint fell back to `/opt` when no mount was detected. **Fix (2026-04-25):** New `Dockerfile.qwen35` is a pure env image — no source baked in, `/opt/Megatron-Bridge` does not exist. `install_runtime_deps.sh` does `uv pip install -e <path>` from `/mnt` explicitly. |

## Mamba-SSM ABI Workaround (NGC nv25.06)

### The problem

NGC PyTorch 25.06 ships `torch==2.8.0a0+5228986c39.nv25.06`, a NVIDIA-patched
2.8 with selective backports from torch 2.9/2.10. Its `c10::cuda` ABI sits
**between stock 2.9 and 2.10**:

| Probe symbol | NGC torch 2.8a | stock torch 2.8/2.9 | stock torch 2.10 |
|---|---|---|---|
| `c10::cuda::SetDevice(int8_t, bool)` | ✓ | ✗ | ✓ |
| `c10::cuda::c10_cuda_check_implementation(..., bool)` | ✗ | ✗ | ✓ |

No stock `mamba_ssm` prebuilt wheel matches:

| Wheel tag | Result |
|---|---|
| `cu12torch2.8cxx11abiTRUE` | `undefined symbol: _ZN3c104cuda9SetDeviceEab` |
| `cu12torch2.9cxx11abiTRUE` | same as 2.8 |
| `cu12torch2.10cxx11abiTRUE` | `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` |

Only `mamba_ssm`'s `selective_scan_cuda.so` trips; `causal_conv1d` cu12torch2.8
wheel works fine.

### Decision (2026-04-24): prebuilt wheel + `__init__.py` patch, baked into image

| Option | Time | Decision | Reason |
|---|---|---|---|
| Source build (`pip install mamba-ssm`) | 30+ min | rejected | nvcc compiles 9 SM archs serially; `TORCH_CUDA_ARCH_LIST` ignored |
| Stock prebuilt wheel matching our ABI | n/a | unavailable | No upstream wheel matches NGC nv25.06 ABI (see table above) |
| **`cu12torch2.8` wheel + patch `__init__.py`** | **~1 min** | **chosen** | Wheel installs fast; patch makes failing `selective_scan_cuda` import non-fatal |

### Why the patch is safe for Qwen3.5-VL

`mamba_ssm/__init__.py` eagerly imports `selective_scan_interface` (→ loads
`selective_scan_cuda`). That path is only used by Mamba1
(`selective_scan_fn`, `mamba_inner_fn`, `mamba_simple.Mamba`).

Megatron-Core's GDN/Mamba2 path used by Qwen3.5-VL imports
`mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` — pure Python +
Triton + causal_conv1d. No dependency on `selective_scan_cuda`.

After the patch (baked into `Dockerfile.qwen35`):
- `import mamba_ssm` → succeeds (one warning)
- `mamba_ssm.selective_scan_fn` → `None` (Mamba1 disabled, not used by Qwen3.5-VL)
- `mamba_ssm.Mamba2` → working class
- `from mamba_ssm.ops.triton.ssd_combined import ...` → working

### Status

✅ **Baked into `Dockerfile.qwen35`** (2026-04-24, revised 2026-04-25):
- `COPY docker/runtime_wheels/*.whl` + `pip install --no-deps`
- `cat > .../mamba_ssm/__init__.py` — patched `__init__.py` written at build time
- Build-time sanity: pure shell `test -f` / `grep` checks (no Python, no GPU needed)

Fresh containers have all three deps + patch ready immediately.

### Open follow-ups

1. If `state-spaces/mamba` ships an `nv25.06`-tagged wheel, switch to it and
   drop the `__init__.py` patch.
2. If NVIDIA releases NGC 25.07+ with TE ≥ 2.7, re-enable `moe_router_fusion`.

## Project Status

| Stage | Status | Owner artifact |
|-------|--------|----------------|
| A. Env on cluster (NGC container) | ✅ Done | `/opt/venv-mbridge/bin/python` works, all NGC libs OK |
| B. HF → mcore conversion | ✅ Done (pre-existing, 234 GB) | `/mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore` |
| C. Pure-env image + /mnt runtime | ✅ Done (2026-04-25) | `Dockerfile.qwen35`, `scripts/install_runtime_deps.sh` |
| D. 2-node × 8-GPU LoRA SFT demo | 🟡 Blocked by Pitfall #14 fix; ready to retry | `scripts/run_sft_qwen35_122b_2node_lora.sh` via `start.sh` |
| E. 4-node × 8-GPU full SFT | 🟡 Script ready, not yet executed | `scripts/run_sft_qwen35_122b_4node.sh` |
| F. Real internal multimodal data | ⏳ Pending — no real images yet | needs spec |

## Open Questions (pending user)

1. Will the cluster scheduler always inject `RANK` as the **node** rank?
   If it injects per-process rank, scripts need `GROUP_RANK` or `SLURM_NODEID`.
2. When does the 4-node allocation become available?
3. Real multimodal data: format, fields, image storage path, total size?

## Legacy

Older bare-metal venv setup (cu128 + TE 2.7 + flash-attn 2.8.1, single-node
H20-141G, `/data/temp/...` paths), old Dockerfile iteration history (with
`COPY` + editable install baked in), conversion debugging — see `memory_legacy.md`.

_Last updated: 2026-04-25 (pure-env image design; install_runtime_deps.sh redesign; Pitfall #14 moe_router_fusion=False; Pitfall #15 stale /opt code; ALWAYS/NEVER boundaries updated)_
