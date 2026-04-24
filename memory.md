# memory.md — Qwen3.5-122B-A10B Multimodal SFT

> **Project:** Run multimodal (image+text) SFT for **Qwen3.5-122B-A10B** on
> NVIDIA L20Y/H800 80G clusters via **NVIDIA NeMo Megatron-Bridge**
> (`megatron.bridge`, *not* `mbridge`).
>
> This file is the **single source of operational truth** for this task. All
> historical decisions and pitfalls live in `memory_legacy.md`; this file
> only carries facts that are still active for the current environment.
>
> Container, paths, and recipes here are **the configuration that works on
> 2026-04-24 in the L20Y/H800 cluster + NGC 25.06 container**. Do not
> blindly copy older paths from `memory_legacy.md` (they assumed a bare-metal
> uv venv at `/data/temp/Megatron-Bridge` which is no longer the case).

## Boundaries

**NEVER:**
- Modify `3rdparty/Megatron-LM/` — submodule pinned to upstream
- Modify `Dockerfile.qwen35` to install `mamba_ssm` / `causal_conv1d` / `fla`
  — they need a GPU at install time and break `docker build`
- Enable sequence packing (`pack_sequences_in_batch=true`) on Qwen3.5-VL —
  GDN/linear-attention only supports BSHD, *not* THD
- Use a `.venv` outside `/opt/venv-mbridge` — the NGC container's torch /
  TE / cuDNN / APEX / flash-attn are all under that prefix
- Pass `model.gradient_accumulation_fusion=false` inside the NGC container
  — APEX cuda ext is present and the recipe default `True` is correct
- Save `/data/temp/...` paths into scripts — that prefix only existed on the
  old bare-metal host; on the cluster use NAS paths under
  `/mnt/tidal-alsh01/dataset/redone/hade/...`

**ASK FIRST:**
- Before changing recipe selection (LoRA vs full SFT vs PEFT scheme)
- Before changing parallelism beyond `TP × PP × EP × DP = world_size`
- Before introducing a new dataset format (current pipeline expects
  `PreloadedVLMConversationProvider` JSONL)

**ALWAYS:**
- Use `/opt/venv-mbridge/bin/python` as the interpreter
- Use the **current year (2026)** in any file you write
- Mount source under `/workspace/Megatron-Bridge` so the container entrypoint
  re-installs editable; otherwise `/opt/Megatron-Bridge` baked-in copy is used
- On rank-0 of any new container, run `bash scripts/install_runtime_deps.sh`
  before launching training (entrypoint does this automatically only if you
  invoke the entrypoint, not if you `bash` into the container)

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
| Cooked demo JSONL (text + 2 image samples) | `/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/demo_data/train_demo.jsonl` | 16 records (built by hand from above + 2 PNG samples) |
| Output / runs / logs root | `/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/` | per-run subdirs |
| Megatron-Bridge source (this repo) | `/mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge` | mounted into container at `/workspace/Megatron-Bridge` (preferred) or `/opt/Megatron-Bridge` (baked-in fallback) |

## Container & Toolchain

Image: built from `Dockerfile.qwen35`, base `nvcr.io/nvidia/pytorch:25.06-py3`
(CUDA 12.9.1 / torch 2.8.0a / TE 2.4 / cuDNN 9.10 / NCCL 2.27 / APEX cuda ext / flash-attn 2.7.4).

| Action | Command |
|--------|---------|
| Interpreter | `/opt/venv-mbridge/bin/python` |
| Install runtime-deferred deps (mamba-ssm/causal-conv1d/fla) | `bash scripts/install_runtime_deps.sh` |
| 2-node × 8-GPU LoRA SFT (current task) | `RANK=<0|1> bash scripts/run_sft_qwen35_122b_2node_lora.sh` |
| 4-node × 8-GPU full SFT (later) | `RANK=<0|1|2|3> bash scripts/run_sft_qwen35_122b_4node.sh` |
| Verify mcore ckpt structure (no GPU) | `/opt/venv-mbridge/bin/python scripts/verify_mcore_ckpt.py /mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore` |
| 1-node 8-GPU VL inference smoke | `bash scripts/run_smoke_qwen35_122b.sh` |
| Operator runbook (per-node steps, troubleshooting) | `docs/qwen35_122b_sft_runbook.md` |

### Manual multi-node launch — the canonical recipe

```bash
# On EACH node, inside the container:
cd /mnt/tidal-alsh01/dataset/redone/hade/dd

# Node 0 (master):
tmux new -s sft -d
tmux send-keys -t sft "RANK=0 MASTER_PORT=23456 bash start.sh" C-m

# Node 1 (worker):
tmux new -s sft -d
tmux send-keys -t sft "RANK=1 MASTER_PORT=23456 bash start.sh" C-m
```

**Always pass `MASTER_PORT=23456` explicitly** — the master's container
sometimes inherits a stale value (`23964`) while the worker correctly sees
`23456`. If they don't match, torchrun rendezvous hangs silently forever
with no error message. See Pitfall #11 for symptoms.

`MASTER_ADDR`, `WORLD_SIZE` come from the cluster scheduler.
`RANK` you set per node (or trust whatever the scheduler injected — the
master sees `RANK=0`).

Inside-container env vars (defaults, override with `docker run -e ...`):

| Variable | Default | What it does |
|----------|---------|--------------|
| `MBRIDGE_PATCH_NVIDIA_FILE` | `0` | NGC TE 2.4 has no namespace bug; bare-metal TE 2.7 needs `1` |
| `MBRIDGE_DISABLE_CUDNN` | `0` | NGC cuDNN 9.10 is healthy; bare-metal sublib bug needs `1` |
| `MBRIDGE_PATCH_GRAD_FUSION` | `0` | NGC APEX cuda ext is present; bare-metal without APEX needs `1` |
| `MBRIDGE_SKIP_RUNTIME_INSTALL` | unset | Set `1` to skip entrypoint auto-install of mamba/causal-conv1d/fla |

## Standard Recipes (table)

| Recipe | Mode | Min GPUs | TP / PP / EP | LR | GBS | Notes |
|--------|------|----------|--------------|-----|-----|-------|
| `qwen35_vl_122b_a10b_peft_config` | LoRA | **16** (2 node × 8) | 2 / 1 / 8 | 2e-4 | 36 | **Current demo target.** Fits cleanly on L20Y 80G. |
| `qwen35_vl_122b_a10b_sft_config` | Full SFT | 32 nominal (4 node × 8) | 2 / 6 / 8 default | 2e-5 | 36 | Default math is 2×6×8=96 GPU; on 32 GPU we override to TP=2 PP=4 EP=8 + DP=2. |
| `qwen35_vl_35b_a3b_peft_config` | LoRA | 8 (1 node) | 2 / 1 / 4 | 2e-4 | — | Useful for fast iteration on smaller MoE sibling. |
| `qwen35_vl_122b_a10b_pretrain_mock_config` | Pretrain (mock) | 8+ | — | — | — | Sanity test only; uses random tokens. |

## Step Function & Dataset Wiring

For VLM SFT use:

- `--step_func vlm_step` (defined in `src/megatron/bridge/training/vlm_step.py`)
- `--dataset vlm-preloaded` → resolves to `PreloadedVLMConversationProvider`
  (`src/megatron/bridge/data/vlm_datasets/preloaded_provider.py`)
- Provider expects records of one of:
  - `{"conversation": [{"role":..., "content":[{"type":"text"|"image", ...}, ...]}, ...]}` (passthrough)
  - `{"messages": [{"role": "user", "content": "<image>\nq?"}, ...], "images": ["/abs/path.png"]}` (legacy, `<image>`/`<video>` placeholders are resolved against `images`/`videos` lists)
  - LLaVA `{"conversations": [{"from": "human", "value": "..."}, ...]}` (legacy)

`dataset.image_folder` is optional; if set, relative paths in `images` are
resolved against it. Absolute paths and `http(s)://` / `file://` URLs are
left alone.

## Known Pitfalls (active for this environment)

| # | Pitfall | Fix |
|---|---------|-----|
| 1 | `mamba_ssm` / `causal_conv1d` / `fla` missing in fresh container | **Do NOT use `bash scripts/install_runtime_deps.sh` source build (30+ min, see Pitfall 9 + §Mamba ABI Workaround).** Use the prebuilt-wheel + `__init__.py` patch flow documented below. ~1 min total. |
| 2 | HF Hub unauthenticated request warning when calling recipe without `--hf_path` | Always pass `--hf_path /mnt/.../Qwen3.5-122B-A10B` to use the local model directory and avoid network roundtrips. |
| 3 | Recipe defaults expect cluster sizes that don't divide our world size (e.g., SFT default TP=2 PP=6 EP=8 needs 96 GPU) | Override `model.tensor_model_parallel_size` / `pipeline_model_parallel_size` / `expert_model_parallel_size` on the CLI. The 4-node script does this for you. |
| 4 | Loss explodes / OOM on full 122B SFT in 16 GPU | Use the LoRA recipe instead — full SFT requires ≥4 nodes. |
| 5 | NCCL hang after first step on multi-node | Confirm `MASTER_ADDR` resolves on every node and `NCCL_SOCKET_IFNAME` matches the routing interface (`bond1` on this cluster). The cluster injects `NCCL_IB_*` already. |
| 6 | "Sending unauthenticated requests to the HF Hub" | Cosmetic warning; safe to ignore for our local-path workflow. Set `HF_TOKEN` if you want to silence it. |
| 7 | `pack_sequences_in_batch=true` on Qwen3.5-VL | Crashes inside GDN kernel. Recipe default is `False`; both demo scripts pass it explicitly. |
| 8 | `gradient_accumulation_fusion=False` left over from old bare-metal scripts | Inside the NGC container, this *hurts* performance (APEX is fine). Don't override it. |
| 9 | `mamba_ssm` source build takes 30+ min | mamba-ssm `setup.py` ignores `TORCH_CUDA_ARCH_LIST` and forces compile of 9 SM archs (sm_62..sm_120) serially. Use prebuilt wheel + `__init__.py` patch — see §Mamba ABI Workaround below. |
| 10 | NGC nv25.06 torch ABI is between stock 2.9 and 2.10 | NVIDIA backported some `c10::cuda::*` symbols. *No* stock `mamba_ssm` prebuilt wheel from upstream resolves cleanly. Workaround: install the `cu12torch2.8` wheel (its `selective_scan_cuda.so` won't load) + patch `mamba_ssm/__init__.py` to make that failure non-fatal. The Triton path used by Qwen3.5-VL GDN doesn't depend on `selective_scan_cuda`. Details in §Mamba ABI Workaround. |
| 11 | `MASTER_PORT` differs across nodes in this cluster (master saw `23964`, worker saw `23456`) | Two-node rendezvous silently hangs forever with no error. The cluster's true default is **23456** — always pass `MASTER_PORT=23456` explicitly when launching `start.sh` if you can't trust the per-node injected value. Symptom: torchrun process alive, 65 threads, sleeping; no worker fork; both nodes look "stuck after OMP banner". |
| 12 | torchrun gives no output for 6–15 min after the OMP banner | NORMAL. Sequence: (a) rendezvous (~10 s), (b) `import megatron.bridge` per rank (60–180 s, fully silent), (c) 234 GB mcore ckpt mmap+load across 16 ranks (3–10 min), (d) iter 1 cold compile+forward+backward (1–3 min). Watch `nvidia-smi` GPU memory rising as a liveness signal; first `iter 1 loss=...` line marks success. |
| 13 | `import mamba_ssm` (or submodules) crashes inside `docker build` with `RuntimeError: 0 active drivers ([])` | Build host has NO GPU. The chain `import mamba_ssm` → `selective_scan_interface.py` → `mamba_ssm.ops.triton.layer_norm` triggers `@triton.autotune(...)` at module load, which calls `driver.active.get_benchmarker()` and dies. **Important:** even `importlib.util.find_spec("mamba_ssm.ops.triton.ssd_combined")` triggers it, because resolving a submodule requires importing the parent package's `__init__.py`. Catching `BaseException` inside our patched `mamba_ssm/__init__.py` would help for the parent import, but not for `find_spec` of submodules whose own module-level autotune still fires. **Fix in Dockerfile.qwen35 (final form 2026-04-24)**: build-time sanity is **pure shell** (`test -f` / `grep`). NO Python imports of any flavour, NO `find_spec`, NO `importlib.metadata`. Real GPU-side verification still happens at container start via `scripts/runtime_sanity.sh` inside a `--gpus all` container. **Cross-cutting rule (also documented in Dockerfile.qwen35 STEP "Build-time sanity" header):** no `import mamba_ssm*` and no `import megatron.bridge` (same hazard, chains into `emerging_optimizers`) in any `RUN` step. |

## Mamba-SSM ABI Workaround (NGC nv25.06)

### The problem (2026-04-24)

NGC PyTorch 25.06 ships `torch==2.8.0a0+5228986c39.nv25.06`, which is a
NVIDIA-patched 2.8 with selective backports from torch 2.9 / 2.10. Its
runtime `c10::cuda` ABI is **between stock 2.9 and 2.10**:

| Probe symbol | NGC torch 2.8a | stock torch 2.8/2.9 | stock torch 2.10 |
|---|---|---|---|
| `c10::cuda::SetDevice(int8_t, bool)` | ✓ has it | ✗ missing | ✓ has it |
| `c10::cuda::c10_cuda_check_implementation(..., bool)` | ✗ missing | ✗ missing | ✓ has it |

So **no stock prebuilt `mamba_ssm` wheel from `state-spaces/mamba` v2.3.1
loads cleanly** in this container:

| Wheel tag | Result |
|---|---|
| `cu12torch2.8cxx11abiTRUE` | `undefined symbol: _ZN3c104cuda9SetDeviceEab` |
| `cu12torch2.9cxx11abiTRUE` | same as 2.8 |
| `cu12torch2.10cxx11abiTRUE` | `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` |

The **`causal_conv1d` v1.6.1.post4 `cu12torch2.8` wheel works fine** — only
`mamba_ssm`'s `selective_scan_cuda.so` extension trips.

### Options considered

| Option | Time | Status | Notes |
|---|---|---|---|
| Source build via `pip install mamba-ssm` (default) | **30+ min** | rejected | nvcc compiles 9 SM archs (sm_62..sm_120) serially. ~95% of compile is wasted on archs we don't have. |
| Source build with `TORCH_CUDA_ARCH_LIST="9.0"` | ~3-5 min | not used | mamba-ssm `setup.py` ignores the env var and forces all archs. Would need a `setup.py` patch. |
| Stock prebuilt wheel that matches our ABI | n/a | unavailable | See table above; no upstream wheel matches NGC nv25.06's patched 2.8a ABI. |
| **Prebuilt `cu12torch2.8` wheel + patch `mamba_ssm/__init__.py`** | **~1 min** | **chosen 2026-04-24** | Wheel installs in seconds; we make the failing CUDA-ext import non-fatal in the package's `__init__.py`. |

### Why patching `__init__.py` is safe for Qwen3.5-VL

`mamba_ssm/__init__.py` eagerly imports
`mamba_ssm.ops.selective_scan_interface` (which loads `selective_scan_cuda`).
That code path is only used by:

- `selective_scan_fn` / `mamba_inner_fn` (Mamba1)
- `mamba_ssm.modules.mamba_simple.Mamba` (Mamba1)

**Megatron-Core's GDN / Mamba2 path used by Qwen3.5-VL imports
`mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`** (see
`3rdparty/Megatron-LM/megatron/core/ssm/mamba_mixer.py:57-60`), which is
**pure Python + Triton + causal_conv1d** — none of it depends on
`selective_scan_cuda`. So as long as `import mamba_ssm` doesn't raise, the
GDN path runs.

The patch wraps every top-level eager import in `__init__.py` in a
`try/except ImportError`. After the patch:

- `import mamba_ssm` → succeeds, prints one warning
- `mamba_ssm.selective_scan_fn` → `None` (Mamba1 disabled — NOT used by Qwen3.5-VL)
- `mamba_ssm.Mamba2` → working class (Mamba2/GDN, used by megatron-core)
- `from mamba_ssm.ops.triton.ssd_combined import ...` → working

Verified on 2026-04-24 inside the live container with this exact probe:

```python
import mamba_ssm
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
import causal_conv1d, fla
# all four imports succeed; Mamba2 is non-None; selective_scan_fn is None
```

### Concrete steps (reproducible)

```bash
# 1) Download wheels into a shared NAS location (only once for the cluster)
WHEELS=/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/wheels
mkdir -p "$WHEELS" && cd "$WHEELS"

curl -fL -o causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

curl -fL -o mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

# 2) Install wheels (no source build)
/opt/venv-mbridge/bin/python -m pip install --no-cache-dir --no-deps \
  "$WHEELS/causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  "$WHEELS/mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# 3) Install fla (pure Python, no compile)
/opt/venv-mbridge/bin/python -m pip install --no-cache-dir --no-deps flash-linear-attention==0.4.2

# 4) Patch mamba_ssm/__init__.py to make the failing CUDA-ext import non-fatal.
#    The patched file lives at:
#      /opt/venv-mbridge/lib/python3.12/site-packages/mamba_ssm/__init__.py
#    Source of truth for the patched contents: see this file's git history,
#    rewritten on 2026-04-24. The file wraps each top-level import in
#    try/except ImportError so `import mamba_ssm` never raises.

# 5) Verify
/opt/venv-mbridge/bin/python -c "
import mamba_ssm
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
import causal_conv1d, fla
print('mamba_ssm', mamba_ssm.__version__,
      '(Mamba2 ok, selective_scan disabled)')
print('causal_conv1d', causal_conv1d.__version__)
print('fla', fla.__version__)
"
```

### Status (2026-04-24, after baking into image)

✅ **`Dockerfile.qwen35` has been updated** to bake everything in:
- `COPY docker/runtime_wheels/*.whl` into the image
- `pip install --no-deps` both wheels + `flash-linear-attention`
- `RUN cat > .../mamba_ssm/__init__.py <<'PYEOF' ... PYEOF` to write the
  patched `__init__.py` at build time
- Build-time sanity now does real `import mamba_ssm` /
  `import mamba_ssm.ops.triton.ssd_combined` and asserts
  `selective_scan_fn is None` (proves the patch is active) and
  `Mamba2 is not None` (proves GDN path is live).

After the next `docker build`, fresh containers come up with all three
deps + patch already present; `start.sh` and `install_runtime_deps.sh`
will report "already installed; skipping".

### Build host workflow (one-time per image rebuild)

```bash
# 1) Get the source + submodule
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git submodule update --init --recursive

# 2) Download the two prebuilt wheels (~780 MB total)
mkdir -p docker/runtime_wheels && cd docker/runtime_wheels
curl -fL -o causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
curl -fL -o mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  'https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
cd ../..

# 3) Build
docker build -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 .

# 4) Push to registry / save to tar / etc.
```

Wheels are NOT git-tracked (~780 MB binaries). The build will fail
fast if `docker/runtime_wheels/*.whl` is missing (with curl commands
in the error message). `.dockerignore` has explicit comments protecting
this path.

### Open follow-ups

1. **Per-node propagation**: now that the image is self-contained, each
   fresh node's container is ready immediately — no NAS wheel cache
   needed at runtime. Just `docker pull` the new image on each node.
2. **`start.sh` is still useful** as a safety net for older containers
   that pre-date the image rebuild (it will detect the patch is already
   present and skip everything in <1 s).
3. **Upstream watch**: if `state-spaces/mamba` ships an `nv25.06`-tagged
   wheel, switch to it and drop the `__init__.py` patch.

## Project Status

| Stage | Status | Owner artifact |
|-------|--------|----------------|
| A. Env on cluster (NGC container) | ✅ Done | `/opt/venv-mbridge/bin/python` works, all NGC libs OK |
| B. HF → mcore conversion | ✅ Done (pre-existing, 234 GB) | `/mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore` |
| C. mcore ckpt offline structure verify | ⏳ TBD this session | `scripts/verify_mcore_ckpt.py` |
| D. 2-node × 8-GPU LoRA SFT demo | 🟢 In progress (2026-04-24): both nodes torchrun launched with `MASTER_PORT=23456`, awaiting first `iter 1 loss=...` line (~6-15 min after launch). | `scripts/run_sft_qwen35_122b_2node_lora.sh` via `start.sh` |
| E. 4-node × 8-GPU full SFT | 🟡 Script ready, not yet executed | `scripts/run_sft_qwen35_122b_4node.sh` |
| F. Real internal multimodal data | ⏳ Pending — user has demo text JSONL only, no real images yet | needs spec |

## Open Questions (pending user)

1. Will the cluster scheduler always inject `RANK` as the **node** rank? If
   it instead injects per-process rank (e.g. via PyTorch's launcher), the
   scripts need to read `GROUP_RANK` or `SLURM_NODEID` instead. Verified
   today on the master node: `RANK=0 WORLD_SIZE=2` looks like *node-level*
   rank, but only one node was checked.
2. When does the 4-node allocation become available? The 4-node script is
   ready; only execution + parallelism tweaks remain.
3. Real multimodal data: format, fields, image storage path, total size?

## Legacy

Older bare-metal venv setup (cu128 + TE 2.7 + flash-attn 2.8.1, single-node
H20-141G, `/data/temp/...` paths), build-host preparation notes, Dockerfile
iteration history, conversion debugging — see **`memory_legacy.md`**.

_Last updated: 2026-04-24 (NGC 25.06 + cu12.9 cluster, 2-node LoRA demo + 4-node full SFT scripts ready)_
