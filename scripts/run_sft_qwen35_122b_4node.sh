#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 4 nodes × 8 GPU (= 32 × H800 80G).
#
# Recipe: qwen35_vl_122b_a10b_sft_config
#   default TP=2, PP=6, EP=8, LR=2e-5, GBS=36, seq=2048
#
# IMPORTANT — parallelism on 32 GPU:
#   TP=2 × PP=6 × EP=8 (recipe default) needs 96 GPU. On 32 GPU override:
#     TP=2, PP=4, EP=4, DP=4.
#   Math: world_size=32, DP=32/(TP×PP)=32/(2×4)=4. EP must ≤ DP → EP=4.
#   EP=8 > DP=4 is INVALID (Megatron-Core asserts); do not use EP=8 on 32 GPU.
#
#   Layer alignment: model has 48 layers in groups of 4.
#   PP=4 → 12 layers/stage = 3 complete groups. Boundaries are aligned. ✓
#
# Memory note (TP=1 PP=4 EP=8 DP=8 SEQ=256 MBS=1, measured 2026-04-26 21:36 run):
#   after-setup    alloc=30.97 GB  reserved=36.68 GB
#   iter steady    peak alloc=37.72 GB  peak reserved=50.18 GB / 79.19 GB total
#   throughput     14 s / iter (cold-start iter 1 = 101 s)
#   loss           1.87 → 0.003 over 20 iters ✓ trains correctly
#
# Mitigations active (all required, all in OVERRIDES below):
#   - SEQ=256 + recompute_method=block recompute_num_layers=12  (Pitfall #23)
#   - mtp_num_layers=0  (Pitfall #19 — would add ~6 GB to last PP stage)
#   - optimizer.optimizer_cpu_offload=True + offload_fraction=1.0 +
#     overlap_cpu_optimizer_d2h_h2d=True  (Pitfall #28 — Adam m/v on CPU)
#   - optimizer.use_precision_aware_optimizer=True  (Pitfall #28 — bf16+resid)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (env, set below)
#   - triton.autotune monkey-patch in run_recipe.py  (Pitfall #22 — harmless,
#     keeps autotune from picking large bench configs)
# See memory.md Pitfalls #22-29 for the diagnostic history that led here.
#
# Container assumptions: same as run_sft_qwen35_122b_2node_lora.sh (NGC 25.06).
#
# Required env (per-node):
#   MASTER_ADDR, MASTER_PORT  cluster-injected
#   RANK or NODE_RANK         this node's index in [0, NNODES)
#   WORLD_SIZE                number of nodes (default 4)
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_4node.sh
#
# Override knobs:
#   TP=2 PP=4 EP=8 ITERS=20 SEQ=2048 GBS=32 MBS=1 ...

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo + paths
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge}"
HF_MODEL="${HF_MODEL:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B}"
MCORE_PATH="${MCORE_PATH:-/mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/demo_data/train_data_demo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/qwen35_122b_full_sft}"
LOG_DIR="${LOG_DIR:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_4node_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-4}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Parallelism overrides (32 GPU layout — see header for math)
# ---------------------------------------------------------------------------
TP="${TP:-1}"
PP="${PP:-4}"
EP="${EP:-8}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
# SEQ=256 chosen 2026-04-26 to fit GDN bwd activation in 80G (Pitfall #23).
# At SEQ=1024 OOM in chunk_gated_delta_rule_bwd even with TP=4 + full recompute.
SEQ="${SEQ:-256}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -d "$REPO_ROOT" ]]                          || { echo "[ERROR] missing REPO_ROOT: $REPO_ROOT" >&2; exit 1; }
[[ -d "$MCORE_PATH/iter_0000000" ]]            || { echo "[ERROR] missing mcore ckpt: $MCORE_PATH/iter_0000000" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]                         || { echo "[ERROR] missing train data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$HF_MODEL" ]]                           || { echo "[ERROR] missing HF model: $HF_MODEL" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Venv python
# ---------------------------------------------------------------------------
VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] venv python missing: $VENV_PY" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Install runtime deps + wire megatron.bridge → /mnt (ALL nodes, not just rank 0).
# install_runtime_deps.sh is idempotent: uv sync is a fast no-op if already done.
# Running on every node is required so that megatron.bridge resolves to /mnt,
# not /opt (Pitfall #15). Skipping on non-rank-0 nodes caused the /opt regression.
# ---------------------------------------------------------------------------
echo "[run_sft] node ${NODE_RANK}: running install_runtime_deps.sh..."
bash "$REPO_ROOT/scripts/install_runtime_deps.sh"

# ---------------------------------------------------------------------------
# Env (NGC container — patches off)
# ---------------------------------------------------------------------------
export MBRIDGE_PATCH_NVIDIA_FILE="${MBRIDGE_PATCH_NVIDIA_FILE:-0}"
export MBRIDGE_DISABLE_CUDNN="${MBRIDGE_DISABLE_CUDNN:-0}"
# expandable_segments=True is more memory-efficient than max_split_size_mb
# (no large fixed blocks that get fragmented).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MBRIDGE_PATCH_GRAD_FUSION="${MBRIDGE_PATCH_GRAD_FUSION:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
# Limit NCCL debug to init phase only, to avoid log explosion during training.
# Set NCCL_DEBUG_SUBSYS=ALL to see everything, but that will be very verbose.
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Do NOT set CUDA_LAUNCH_BLOCKING — it serializes all CUDA ops and breaks NCCL async.
# Do NOT override NCCL_SOCKET_IFNAME here — the cluster injects bond1 correctly.
# Pitfall #22: triton.autotune monkey-patch is in run_recipe.py (verified working).
# Keep FLA_AUTOTUNE=0 too as belt-and-suspenders.
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export HF_HOME="${HF_HOME:-/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run/hf_cache}"
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# Recipe + dataset overrides
# ---------------------------------------------------------------------------
OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    --step_func vlm_step
    --hf_path "$HF_MODEL"

    # Parallelism (32-GPU specific override)
    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.expert_model_parallel_size="$EP"

    # Activation recompute (full BLOCK — every layer of every PP stage is
    # recomputed end-to-end, freeing all per-layer GDN bwd workspace as soon
    # as the layer's bwd finishes). With PP=4 each stage has 12 layers.
    # uniform+1 (the previous setting) only recomputed in chunks of 1 layer
    # which still kept many layers' fwd activations alive at once during bwd
    # — contributed to Pitfall #23 OOM at SEQ=1024.
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers=12

    # Disable MTP (Multi-Token Prediction) to relieve last-pipeline-stage memory.
    # MTP adds ~1 extra transformer block + LM-head overhead only on the last PP stage,
    # causing rank3 OOM while rank0-2 are fine. MTP is for inference throughput, not
    # needed for SFT training correctness.
    model.mtp_num_layers=0

    # GDN constraint
    dataset.pack_sequences_in_batch=False

    # Optimizer CPU offload (Pitfall #28, 2026-04-26).
    # The 122B model on 32 GPU has baseline 48 GB / 80 GB, leaving 22 GB for
    # activations + optimizer-state lazy alloc.  But Adam's exp_avg + exp_avg_sq
    # buffers (~30 GB total fp32 unsharded, ~7.5 GB/GPU after DP=8 sharding)
    # are LAZILY allocated inside fused_adam.initialize_state on the first
    # optimizer.step().  Combined with iter-0 fwd/bwd transients
    # (~21 GB peak), the alloc crosses 80 G and OOMs.
    #
    # Fix: move all Adam state to CPU.  HybridDeviceOptimizer keeps Adam
    # math on CPU (slower step but compute is small fraction of total).
    # Skill cpu-offloading: optimizer offload is the only option for PP > 1.
    # PP=4 here, so activation offload is forbidden.
    #
    # Set OPTIM_OFFLOAD_FRAC=0 to disable; OPTIM_OFFLOAD_FRAC=1.0 for max savings.
    optimizer.optimizer_cpu_offload="${OPTIM_OFFLOAD:-True}"
    optimizer.optimizer_offload_fraction="${OPTIM_OFFLOAD_FRAC:-1.0}"
    optimizer.overlap_cpu_optimizer_d2h_h2d="${OPTIM_OFFLOAD_OVERLAP:-True}"
    optimizer.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPT:-True}"

    # Iters / batch
    train.train_iters="$ITERS"
    train.global_batch_size="$GBS"
    train.micro_batch_size="$MBS"
    train.eval_iters=0
    train.eval_interval=1000000

    # Checkpoint
    checkpoint.pretrained_checkpoint="$MCORE_PATH"
    checkpoint.save="$OUTPUT_DIR"
    checkpoint.save_interval="$SAVE_INTERVAL"
    checkpoint.load=null

    # Dataset
    dataset.train_data_path="$TRAIN_DATA"
    dataset.hf_processor_path="$HF_MODEL"
    dataset.seq_length="$SEQ"

    # Logging
    logger.log_interval="$LOG_INTERVAL"
    logger.tensorboard_dir=null
    logger.wandb_project=null
)

cat <<EOF
============================================================
Qwen3.5-122B-A10B FULL SFT (4-node)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  EP=${EP}
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  Log file     : $LOG_FILE
  Venv python  : $VENV_PY
============================================================
EOF

RDZV_TIMEOUT="${RDZV_TIMEOUT:-1800}"

"$VENV_PY" -u -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_conf "timeout=${RDZV_TIMEOUT}" \
    scripts/training/run_recipe.py \
    "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"
