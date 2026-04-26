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
| Megatron-Bridge source | any path — set `REPO_ROOT` env var, or `cd` into the repo and scripts auto-detect via `BASH_SOURCE` | clone to any directory |

## Container & Toolchain

Image: built from `Dockerfile.qwen35`, base `nvcr.io/nvidia/pytorch:25.06-py3`
(CUDA 12.9.1 / torch 2.8.0a / TE 2.4 / cuDNN 9.10 / NCCL 2.27 / APEX cuda ext / flash-attn 2.7.4).

### Image design (decided 2026-04-25)

The image is a **pure environment image**. It contains only:
- OS tools (git, tmux, curl, …)
- uv package manager
- `/opt/venv-mbridge`: venv inheriting all NGC site-packages, plus
  causal-conv1d 1.6.1 + mamba-ssm 2.3.1 (prebuilt wheels, ABI-patched)
  and flash-linear-attention 0.4.2 + fla-core 0.4.2 (pure Python, network install at build time)

**What the image does NOT contain:** Megatron-Bridge source, megatron-core,
any editable install. All business code lives on NAS under `/mnt`.

**fla归属：** `flash-linear-attention` + `fla-core` 在 **build time** 通过网络 `pip install` 装入镜像（纯 Python，无 CUDA 编译）。`uv sync` 在运行时**跳过**它们（`--no-install-package flash-linear-attention --no-install-package fla-core`）。

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

`REPO_ROOT` is the directory where you cloned Megatron-Bridge. All scripts
auto-detect it from their own location (`BASH_SOURCE`), so you never need to
hard-code a path — just `cd` into the repo before running anything.

```bash
# Step 1 — start container, mount the NAS (adjust -v mount to your site)
docker run --rm -it --gpus all --shm-size=64g --ulimit memlock=-1 \
  --network host \
  -v /your/nas/mount:/your/nas/mount \
  qwen35-mbridge:cu129 bash

# Step 2 — inside container: cd into wherever you cloned the repo, then setup
cd /path/to/Megatron-Bridge          # ← the only path you need to know
bash scripts/install_runtime_deps.sh
# This script does:
#   1. git submodule update --init (if 3rdparty/Megatron-LM is empty)
#   2. uv sync  — installs megatron-core (editable) + all pyproject.toml deps
#   3. uv pip install -e .  — installs megatron-bridge pointing at the repo
#   4. verifies mamba_ssm / causal_conv1d / fla (already in image, fast check)
#   5. prints megatron.bridge.__file__ → must show the repo path, not /opt

# Step 3 — set data/model paths (site-specific), then launch on each node
export HF_MODEL=/path/to/Qwen3.5-122B-A10B
export MCORE_PATH=/path/to/Qwen3.5-122B-A10B-mcore
export TRAIN_DATA=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/output
bash run/start_4node.sh              # for 4-node full SFT
# bash run/start.sh                 # for 2-node LoRA SFT
```

### Quick reference

All commands below assume you have already `cd /path/to/Megatron-Bridge`.

| Action | Command |
|--------|---------|
| Interpreter | `/opt/venv-mbridge/bin/python` |
| Runtime setup (once per container) | `cd <repo> && bash scripts/install_runtime_deps.sh` |
| 2-node × 8-GPU LoRA SFT | `bash run/start.sh` |
| 4-node × 8-GPU full SFT | `bash run/start_4node.sh` |
| Underlying SFT scripts | `scripts/run_sft_qwen35_122b_2node_lora.sh` / `scripts/run_sft_qwen35_122b_4node.sh` |
| Verify mcore ckpt (no GPU needed) | `/opt/venv-mbridge/bin/python scripts/verify_mcore_ckpt.py $MCORE_PATH` |
| Operator runbook | `docs/qwen35_122b_sft_runbook.md` |

### Manual multi-node launch — canonical recipe

```bash
# On EACH node, inside the container, after install_runtime_deps.sh:
cd /path/to/Megatron-Bridge

# Export site-specific paths (only needed if not injected by scheduler)
export HF_MODEL=/path/to/Qwen3.5-122B-A10B
export MCORE_PATH=/path/to/Qwen3.5-122B-A10B-mcore
export TRAIN_DATA=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/output

# Node 0 (master):
tmux new -s sft -d
tmux send-keys -t sft "RANK=0 MASTER_PORT=23456 bash run/start_4node.sh" C-m

# Node 1-3 (workers):
tmux new -s sft -d
tmux send-keys -t sft "RANK=<1|2|3> MASTER_PORT=23456 bash run/start_4node.sh" C-m
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
| `qwen35_vl_122b_a10b_sft_config` | Full SFT | 32 nominal (4 node × 8) | 2 / 6 / 8 default | 2e-5 | 36 | Default math is 2×6×8=96 GPU; on 32 GPU override to TP=2 PP=4 **EP=4** DP=4. EP must ≤ DP. |
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
| 16 | `uv sync` silently removes baked-in packages (`mamba_ssm`, `causal_conv1d`, `fla`) | `uv sync` default mode is **exact**: it removes anything not in the lockfile, even packages installed by `pip` at image build time. **Fix (2026-04-26):** Add `--inexact` to the `uv sync` call in `install_runtime_deps.sh`. This makes uv only install missing packages, never remove extras. |
| 17 | EP > DP assertion / silent memory error on 32 GPU with EP=8 | DP = world_size/(TP×PP) = 32/(2×4) = 4. Megatron-Core requires EP ≤ DP. EP=8 > DP=4 is invalid: triggers assert or wrong memory layout. **Fix (2026-04-26):** Set `EP=4` in `run_sft_qwen35_122b_4node.sh`. |
| 18 | Non-rank-0 nodes keep megatron.bridge pointing at `/opt` | `run_sft_qwen35_122b_4node.sh` ran `install_runtime_deps.sh` only on rank 0. Non-rank-0 nodes skipped the `uv pip install -e` step → megatron.bridge resolved to stale `/opt` path. **Fix (2026-04-26):** Remove the `if [[ NODE_RANK -eq 0 ]]` guard — run `install_runtime_deps.sh` on **all nodes** unconditionally (it is idempotent). |
| 19 | Last PP stage OOM while ranks 0-2 are fine (MTP layer imbalance) | `mtp_num_hidden_layers=1` adds one extra transformer block + LM head to the **last pipeline stage only**. On tight 80G GPUs this creates ~6-8 GB extra pressure on rank(PP-1) vs other stages. Symptom: OOM during optimizer-state initialization on last-stage ranks while all other ranks pass. **Fix (2026-04-26):** Set `model.mtp_num_layers=0` in overrides to disable MTP for SFT. See §MTP Memory Note for cost of re-enabling. |
| 20 | `FileNotFoundError: triton_poi_fused_mul_silu_1.json` on rank N during Triton JIT compilation | Root cause: `TRITON_CACHE_DIR` was pointed at a NAS path (`/mnt/...`). 32 ranks concurrently write to the same NFS directory; NFS cache-coherency delay means a rank can see the directory entry before the `.json` metadata file is visible → `FileNotFoundError`. Setting a shared NFS path for Triton cache is wrong: each rank compiles independently and caches locally by default; cross-rank sharing has no benefit and introduces NFS race conditions. **Fix (2026-04-26):** Remove `TRITON_CACHE_DIR` from `run_sft_qwen35_122b_4node.sh`; let each rank use its default local cache (`~/.triton/cache`). Do NOT set `TRITON_CACHE_DIR` to any NFS/NAS path in multi-process training. |
| 21 | `ModuleNotFoundError: No module named 'transformer_engine'` spam from `te_distributed_compat.pth` on every container restart | Root cause: `install_runtime_deps.sh` (old step 4b) wrote `import transformer_engine.pytorch.distributed` into a venv `.pth` file. Python's `site` module processes venv `.pth` files immediately after adding that directory to `sys.path` — before `/usr/local/lib/python3.12/dist-packages` (NGC's TE location) is added. So the import always fails at `.pth` execution time. Non-fatal but causes log noise and signals broken setup; next container restart repeats the error. **Fix (2026-04-26):** (1) `install_runtime_deps.sh` step 4b now **removes** any stale `.pth` instead of creating one. (2) `scripts/training/run_recipe.py` does the import explicitly at process startup, after `sys.path` is complete. **Rule:** never use `.pth` files in venv site-packages to import packages that live in NGC system site-packages. |
| 22 | `RuntimeError: Triton Error [CUDA]: out of memory` in `chunk_gated_delta_rule_bwd` during first backward pass — **initial misdiagnosis as autotune OOM** | **Initial (wrong) hypothesis:** Triton autotuner benchmarks ALL candidate configs on the first backward pass, allocating extra temp tensors → OOM. **Mitigation tried (still in code):** monkey-patch `triton.autotune` in `scripts/training/run_recipe.py` to keep only `configs[:1]`, disabled by `TRITON_DISABLE_AUTOTUNE_PATCH=1`. **Verification (2026-04-26, diag2):** the patch DOES work — `chunk_bwd_kernel_dqkwg.fn.configs len=1` after `import megatron.bridge.recipes`. Patch was called 200 times across the FLA + mamba_ssm + TE import chain; control test (no patch) shows 6 configs (Hopper non-fp8 path: NUM_WARPS=[2,4] × num_stages=[2,3,4]). **Real root cause is Pitfall #23.** The patch is harmless and may give 1-2% speedup by skipping bench step, so it stays in. Re-read autotuner.py: `autotuner.py:206` is the FINAL `self.fn.run(...)` call (the kernel launch), executed regardless of `len(configs)`. A traceback ending at L206 does NOT prove autotune was responsible. |
| 23 | OOM in `chunk_gated_delta_rule_bwd` even with autotune patch active (configs len=1) | **Real root cause:** GDN backward pass for Qwen3.5-122B-A10B is intrinsically memory-heavy. The `chunk_o.py:chunk_bwd_dqkwg` kernel allocates `dq, dk, dv, dw, dg` — five tensors, each shape `[B, SEQ, V_heads/TP, V_dim]` bf16. With SEQ=1024, V_heads=64, TP=2, V_dim=128, MBS=1: 5 × 1×1024×32×128×2 B = ~40 MB per kernel call, but multiple GDN layers' bwd workspaces are live simultaneously during pipeline bubble fill. The cumulative transient overshoot is the OOM trigger, not the steady-state baseline (which is comfortable at ~45 GB on 80G — see `run/diag_mem_budget.py`). **Fix (2026-04-26):** (a) reduce `SEQ` from 1024 → 256 (4× less GDN bwd workspace), (b) `recompute_method=block recompute_num_layers=12` so each PP stage's 12 layers all do recompute, freeing fwd activations before bwd starts, (c) keep parallelism at **TP=2 PP=4 EP=4 DP=4** (do NOT raise TP — see Pitfall #24). **If still OOM:** SEQ=128 first, then consider CP=2 (verify Qwen3.5 GDN supports CP — gated_delta_rule kernel currently does not parallelise across the chunk axis without ring-attn). **Diagnosis tool:** `python3 run/diag_mem_budget.py` (CPU-only, reads HF config.json). |
| 24 | Tempting but WRONG: raise TP=2→4 to "halve GDN per-GPU heads" and thus reduce bwd activation | **Why it's wrong for MoE Qwen3.5-122B-A10B:** the model has 256 experts. EP is the dominant sharding dimension for MoE params, not TP. With world=32: TP=4 EP=2 DP=2 holds 128 experts per GPU (~28 GB params + ~56 GB Adam fp32) = ~111 GB per-GPU baseline → guaranteed OOM. TP=2 EP=4 DP=4 holds 64 experts per GPU (~15 GB params + ~15 GB Adam) = ~45 GB baseline → comfortable. The GDN bwd activation reduction from raising TP (V_heads/TP smaller) is dwarfed by the MoE param + Adam memory increase from lowering EP. **Rule:** for MoE models, max EP first (subject to EP ≤ DP), then TP. Activation reduction comes from SEQ↓ or recompute, NOT from TP↑. Verified by `run/diag_mem_budget.py` (2026-04-26). |
| 25 | Confusion: "SEQ=1024 with DP=4 means each GPU runs SEQ=4096?" | **No.** Each DP rank processes a DIFFERENT micro-batch in parallel; per-GPU sequence length is unchanged. Per-dimension semantics: TP/PP/EP shard model weights, DP replicates the model across data shards, **CP** (context parallel) is the ONLY dim that splits a single sample's seq across GPUs. Global token count per step = `SEQ × MBS × DP × grad_accum_steps`, but each GPU only sees `SEQ × MBS` tokens at a time. **What can blow up activation memory beyond a single MBS:** PP=N + 1F1B schedule keeps up to PP-1 micro-batches' fwd activations alive on pipeline-stage-0 during warmup (bubble fill). With PP=4, that's 3× the per-MBS activation footprint on early stages. This is why `recompute_method=block` (every layer recomputed during bwd) is essential — it stashes only inputs at block boundaries, not full activations. The real OOM driver remains GDN bwd's transient workspace, which scales with `SEQ × V_heads/TP`, NOT with DP. |
| 26 | ~~OOM persists at SEQ=256 + block-recompute~~ — **MISDIAGNOSIS, see #27** | _Original entry retained for history. The "OOM" was actually a NameError in our diagnostic patch; the "99% GPU memory at startup" was a zombie CUDA context from a previous failed run. Real baseline is 48 GB / 80 GB — see Pitfall #27 for the correction._ |
| 29 | ✅ **Stage E completed (2026-04-26 21:36 run)** — full SFT 20/20 iters, loss 1.87 → 0.003 | **Final working config** (matches `scripts/run_sft_qwen35_122b_4node.sh` defaults today): `TP=1 PP=4 EP=8 DP=8`, `SEQ=256 MBS=1 GBS=32`, `recompute_method=block recompute_num_layers=12`, `mtp_num_layers=0`, `optimizer.optimizer_cpu_offload=True optimizer_offload_fraction=1.0 overlap_cpu_optimizer_d2h_h2d=True use_precision_aware_optimizer=True`, `expandable_segments=True`, triton.autotune monkey-patch on. **Measured per-GPU memory** (`peak_reserv` from `torch.cuda.memory_reserved`): after-setup 30.97 G alloc / 36.68 G reserved, peak 37.72 G alloc / **50.18 G reserved** out of 79.19 G — leaves ~30 G headroom for SEQ scaling or removing offload later. **Throughput**: 14 s/iter steady-state on 32 × L20Y 80G; cold-start iter 1 = 101 s (autotune + ckpt warmup). **Calibrator validation:** `run/diag_mem_budget.py` predicts 50.36 G iter-peak for this config, matches measured 50.18 G to within 0.4 % — calibration constants `MEM_CALIB_FACTOR=2.0` and `ACT_FACTOR=180` are accurate for the Qwen3.5-122B-A10B + L20Y combination. **OOM-diag hooks REMOVED 2026-04-26**: now that the working config is established, the diagnostic prints in `pretrain.py` / `train.py` / `run_recipe.py` (added during the misdiagnosis storm of #22-28) are reverted. The standalone tool `run/diag_mem_budget.py` is the surviving artifact for memory planning. |
| 28 | Real OOM at iter 0 `optimizer.step()` — `fused_adam.py:377 torch.empty_like(param, fp32)` for `exp_avg` | **Diagnostic data (run `2026-04-26_212214`):** alloc went 48 G (after-setup) → 69.6 G (at-exception) → tried to alloc another 24 MB and failed; `cudaMalloc retries: 0` followed by 12 MB free. **What is happening:** Megatron-Core's distributed optimizer (`HybridDeviceOptimizer` not used, plain Adam path) allocates Adam's `exp_avg` and `exp_avg_sq` fp32 buffers LAZILY inside `transformer_engine.fused_adam.initialize_state` on the FIRST `optimizer.step()`, not at setup. Per-GPU after DP=8 sharding: ~7.5 GB each (m + v = 15 GB fp32). Combined with iter-0 fwd+bwd activation peak (~21 GB above the 48 GB baseline), this crosses 80 G. **Why setup mem-print did not catch it:** setup completes before any train_step runs, so the lazy alloc has not happened yet. **Fix (2026-04-26):** add three CLI overrides in `scripts/run_sft_qwen35_122b_4node.sh`: `optimizer.optimizer_cpu_offload=True optimizer.optimizer_offload_fraction=1.0 optimizer.overlap_cpu_optimizer_d2h_h2d=True optimizer.use_precision_aware_optimizer=True`. The first three switch optimizer-state alloc to CPU (`HybridDeviceOptimizer`), saving the full ~15 GB of m+v on GPU. The fourth converts fp32 main_param storage into bf16 + fp16-residual (Megatron-Core precision-aware Adam), saving another ~7.5 GB. Combined savings: ~22 GB per GPU. Throughput penalty: ~2x optimizer.step (Adam math runs on CPU); since optimizer.step is < 5% of iter time on a 122B model, end-to-end iter time penalty is small. Override knobs: `OPTIM_OFFLOAD=False` to disable; `OPTIM_OFFLOAD_FRAC=0.5` for partial offload (better throughput, less memory savings). See `skills/perf-techniques/cpu-offloading/SKILL.md`. **Compatibility constraint:** activation offload is BLOCKED by PP > 1 (we have PP=4); only optimizer offload is available for this configuration. |
| 27 | False-positive "OOM" cascade — TP=1 PP=4 EP=8 SEQ=256 with `MBRIDGE_OOM_DIAG=1` finally exposed the truth | **Diagnostic data (run `2026-04-26_210829`):** `pretrain:before-setup` alloc=0 / free=78.7 G/79.2 G; `pretrain:after-setup (model+optim+ckpt loaded)` alloc=**48.09 G** reserv=53.83 G peak=53.80 G / free=23.86 G; `pretrain:before-train-call` identical; OOM excepthook fired with `Exception: NameError: name 'iteration' is not defined`, `CUDA OOMs: 0`, `cudaMalloc retries: 0`. **What was actually wrong:** our diag patch in `train.py` referenced a local var `iteration` that does not exist in this scope — the loop variable is `global_state.train_state.step`. rank-0 raised NameError before the first forward pass; other ranks NCCL-hung for 600 s; torchrun reported `exitcode 1` which we kept misreading as OOM. **What was wrong about Pitfalls #22–26:** Pitfall #22 (autotune patch) was correct in intent but did not need defending — `autotuner.py:206` is the regular kernel-launch line, not a benchmarking line, so a traceback there does NOT prove autotune was the culprit. Pitfalls #23/#26 (GDN bwd transient OOM, baseline saturating 80 G) were wrong: real baseline is 48 GB, leaving 24 GB for activations — comfortable for SEQ=256 and likely fine for SEQ=1024. The "99% GPU memory at startup" the user reported in nvidia-smi was a **zombie CUDA context from a previous failed run** that the driver had not reclaimed. **Lessons learned:** (a) ALWAYS install an `excepthook` + `memory_summary` hook before guessing OOM is OOM; (b) `cudaMalloc retries: 0` proves there was no actual OOM, regardless of what the traceback says; (c) `nvidia-smi` after a crashed run may show stale memory until a new process initialises a fresh CUDA context; (d) keep `expandable_segments:True` — it is more memory-efficient AND `memory_summary()` still prints true allocated/reserved on excepthook, so the original "expandable_segments hides numbers" rationale was wrong. **Fix (2026-04-26):** `train.py` now uses `global_state.train_state.step` for the iteration index. Diagnostic hook in `run_recipe.py` (lazy CUDA init via builtins._mbridge_mem) is kept ON by default — its overhead is microseconds per iter and the next mystery OOM will give us the truth in one run. |

## MTP Memory Note (for future re-enablement)

`mtp_num_hidden_layers=1` (the default for Qwen3.5-122B-A10B) places one extra
transformer block on the **last PP stage only**. Memory cost per GPU on that stage:

| Component | Estimate (TP=2, EP=4, DP=4, PP=4) |
|-----------|------------------------------------|
| MTP block params bf16 | ~244 GB / 48 layers / TP=2 / PP-stage ≈ **2.5 GB** |
| MTP block grads bf16 | ~2.5 GB |
| Adam optimizer fp32 (ZeRO-1 / DP=4) | ~2 × 2.5 / 4 ≈ **1.25 GB** |
| **Total extra on last stage** | **~6–7 GB** |

To re-enable MTP (`model.mtp_num_layers=1`) the last PP stage needs ~6-7 GB
of additional headroom. Options (in order of preference):
1. Increase PP (e.g. PP=6 on 48 nodes) — fewer layers per stage, less activation memory
2. Use `SEQ=1024` (further activation reduction)
3. Use TP=4 (halves per-GPU param/grad pressure, requires 2× TP bandwidth)

Do **not** re-enable MTP until the baseline full-SFT demo runs successfully.

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
| D. 2-node × 8-GPU LoRA SFT demo | ⏳ Deferred (user prefers full SFT) | `scripts/run_sft_qwen35_122b_2node_lora.sh` via `start.sh` |
| E. 4-node × 8-GPU full SFT | ✅ **DONE (2026-04-26 21:36 run)** — 20/20 iters completed, loss 1.87 → 0.003, ~14 s/iter (cold start 101 s). Final config: TP=1 PP=4 EP=8 DP=8, SEQ=256, MBS=1, GBS=32, block-recompute/12, MTP=0, optimizer CPU offload (frac=1.0), use_precision_aware_optimizer=True, autotune patch on. Per-GPU peak 50 GB / 80 GB ✓ comfortable. | `scripts/run_sft_qwen35_122b_4node.sh` |
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

_Last updated: 2026-04-26 (Pitfall #29 — **Stage E SUCCESS**: 20/20 iters, loss 1.87 → 0.003, peak 50 G / 80 G. Final config locked in `scripts/run_sft_qwen35_122b_4node.sh`. Calibrator `run/diag_mem_budget.py` validated to ±0.4 % accuracy. OOM diag hooks reverted from training code; standalone budget tool retained. Next: Stage F — real internal multimodal data.)_
