# Qwen3.5-122B-A10B SFT — Operator Runbook

A 1-page runbook for launching the **2-node × 8-GPU LoRA SFT demo** (current
target) and the **4-node × 8-GPU full SFT** (later) inside the NGC 25.06
container.

## Pre-flight (once per fresh container, on **every** node)

```bash
cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge

# 1) Verify the container shipped what we expect (~5 s)
/opt/venv-mbridge/bin/python -c "
import torch, transformer_engine, flash_attn, fused_weight_gradient_mlp_cuda
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('TE', transformer_engine.__version__)
print('flash-attn', flash_attn.__version__)
print('apex cuda ext OK')
"

# 2) Install runtime-deferred deps (mamba-ssm / causal-conv1d / fla)
#    First call: 5–15 min nvcc compile. Subsequent calls: ~1 s no-op.
bash scripts/install_runtime_deps.sh

# 3) Sanity-check the cluster's distributed env
env | grep -E "^(MASTER_ADDR|MASTER_PORT|RANK|WORLD_SIZE|NCCL_)" | sort
```

Expected at step 3:
- `MASTER_ADDR` resolves on every node (try `getent hosts $MASTER_ADDR`)
- `WORLD_SIZE` = number of nodes you allocated (e.g. `2` for the LoRA demo)
- `RANK` should be the **node index** in `[0, WORLD_SIZE)`. If your
  scheduler doesn't set it, pass `RANK=<n>` on the command line.

## 2-node LoRA SFT — manual launch

Open `tmux` on each node so the run survives ssh disconnects.

**Node 0 (master)**
```bash
cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge
tmux new -s sft -d
tmux send-keys -t sft "RANK=0 bash scripts/run_sft_qwen35_122b_2node_lora.sh" C-m
tmux attach -t sft        # detach with Ctrl-B D
```

**Node 1 (worker)**
```bash
cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge
tmux new -s sft -d
tmux send-keys -t sft "RANK=1 bash scripts/run_sft_qwen35_122b_2node_lora.sh" C-m
tmux attach -t sft
```

That's it. The script:
- reads `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE` from the container env
- derives `--nproc_per_node=8`, `--nnodes=$WORLD_SIZE`, `--node_rank=$RANK`
- launches `scripts/training/run_recipe.py` with
  `qwen35_vl_122b_a10b_peft_config` + `--peft_scheme lora` +
  `--dataset vlm-preloaded --step_func vlm_step`
- loads the mcore base from
  `/mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore`
- uses the demo JSONL at
  `/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/demo_data/train_demo.jsonl`
- writes logs to `/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/logs/`
- writes any (disabled) checkpoints to
  `/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/qwen35_122b_lora_demo/`

### What you should see (rough timeline)

| When | Expected output |
|------|-----------------|
| 0–60 s | torchrun bootstrap, NCCL init, PyTorch DDP messages |
| 1–8 min | mcore checkpoint load — 234 GB sharded across 16 ranks |
| 8–12 min | iter 1 forward+backward (cold), `train loss=...` line |
| afterwards | one log line per `iter`, ~1–3 min/iter on 16 × L20Y for LoRA |

If you see `iter 1 ... loss=...` you're done — the pipeline runs. Default
`ITERS=10` so the whole demo wraps in ~30–60 min after warm-up.

## Quick overrides (env vars)

```bash
# Run more iters
ITERS=50 RANK=0 bash scripts/run_sft_qwen35_122b_2node_lora.sh

# Larger batch (need more memory)
GBS=32 MBS=2 RANK=0 bash scripts/run_sft_qwen35_122b_2node_lora.sh

# Longer sequence (default 2048; recipe default is 4096)
SEQ=4096 RANK=0 bash scripts/run_sft_qwen35_122b_2node_lora.sh

# Verbose NCCL for debugging hangs
NCCL_DEBUG=INFO RANK=0 bash scripts/run_sft_qwen35_122b_2node_lora.sh
```

## 4-node Full SFT — manual launch (when 4 nodes available)

Same shape, different script and 4 nodes:

```bash
# Node 0 (master)
cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge
tmux new -s full -d
tmux send-keys -t full "RANK=0 bash scripts/run_sft_qwen35_122b_4node.sh" C-m

# Node 1 / 2 / 3
RANK=1 bash scripts/run_sft_qwen35_122b_4node.sh   # on node 1
RANK=2 bash scripts/run_sft_qwen35_122b_4node.sh   # on node 2
RANK=3 bash scripts/run_sft_qwen35_122b_4node.sh   # on node 3
```

Default parallelism is **TP=2 PP=4 EP=8** (override `TP=`, `PP=`, `EP=`).
Full SFT on 32 × L20Y 80G is *tight* — expect to tune `MBS`, recompute, and
maybe `expert_tensor_parallel_size` further.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: mamba_ssm` | runtime-deferred deps not installed | `bash scripts/install_runtime_deps.sh` on this node |
| Hangs at "Initializing process group" | `MASTER_ADDR` not resolving from worker | `getent hosts $MASTER_ADDR` on worker; check NCCL_SOCKET_IFNAME |
| `NCCL ... Connection timed out` | wrong `MASTER_PORT` or firewall | confirm port matches across nodes |
| `RuntimeError: CUDA out of memory` mid-training | seq too long / MBS too large | drop `SEQ=2048` or `MBS=1` |
| `RuntimeError: 0 active drivers` at import | `import mamba_ssm` from a no-GPU shell | only run scripts in a `--gpus all` container |
| Loss is `nan` from iter 1 | likely TP/PP misconfigured for 122B | for the 4-node script, keep TP=2 PP=4 EP=8; do not change without verifying world size |

## File map

| Path | Purpose |
|------|---------|
| `scripts/run_sft_qwen35_122b_2node_lora.sh` | 2-node × 8-GPU LoRA demo (current target) |
| `scripts/run_sft_qwen35_122b_4node.sh` | 4-node × 8-GPU full SFT (later) |
| `scripts/install_runtime_deps.sh` | Idempotent installer for mamba-ssm/causal-conv1d/fla |
| `scripts/verify_mcore_ckpt.py` | Offline (no GPU) sanity for the mcore checkpoint |
| `scripts/run_smoke_qwen35_122b.sh` | Single-node 8-GPU inference smoke (real text generation) |
| `memory.md` | Single source of operational truth |
| `memory_legacy.md` | Older bare-metal venv notes |

_Updated: 2026-04-24._
