#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 24 nodes × 8 GPU (= 192 × H800 80G).
#
# Recipe: qwen35_vl_122b_a10b_sft_config
#   default TP=2, PP=6, EP=8, LR=2e-5, GBS=36, seq=4096
#
# IMPORTANT — parallelism on 192 GPU with 96K context, CP=1:
#   Use TP=2, PP=12, CP=1, EP=4, DP=8.
#   Math: world_size=192, DP=192/(TP×PP×CP)=192/(2×12×1)=8.
#   EP must ≤ DP → EP=4 ≤ DP=8 ✓.
#   CP=1: each rank holds the full 96K sequence's activations — flash-attn /
#   TE attention is O(S) memory so attention itself is fine, but everything
#   else (LM head logits, GDN bwd workspace) is sized by the full SEQ.
#
#   Layer alignment: model has 48 layers in groups of 4 (3 GDN + 1 Attention).
#   PP=12 → 4 layers/stage = exactly 1 complete group per stage. ✓
#
# Memory safety strategy (CP=1 + SEQ=96K is the most aggressive shape we run):
#   1. cross_entropy_fusion_impl="te"
#        — chunked CE inside TE, never materializes the full
#          [B, 96K, V/TP] = ~23 GB logits tensor on the last PP stage.
#          Without this, last-stage rank OOMs at the LM head.
#   2. recompute_granularity=full, method=block, num_layers=4
#        — every layer of every stage is recomputed end-to-end (4 layers/stage
#          for PP=12), freeing per-layer GDN bwd workspace immediately.
#   3. optimizer_cpu_offload=True, fraction=1.0
#        — DP=8 already shards Adam state to ~10 GB/GPU vs ~30 GB at DP=4
#          (12-node), but SEQ went 32K→96K so transient activation peak grew.
#          Keep offload ON; flip OPTIM_OFFLOAD=False once a clean iter-1
#          shows headroom > 15 GB.
#   4. mtp_num_layers=0 — Pitfall #19, removes ~6-7 GB extra on last stage.
#   5. expandable_segments:True — Pitfall #28 anti-fragmentation.
#   6. sequence_parallel=True (recipe default for MoE) — TP=2 splits SEQ
#      across TP ranks for non-attention activations, halving most buffers.
#
# Estimated per-GPU peak (last PP stage, TP=2 PP=12 CP=1 EP=4 DP=8, SEQ=96K):
#   - Params bf16   ~5 GB     (sharded across TP × PP × EP)
#   - Grads  bf16   ~5 GB
#   - Adam state    ~0 GB     (offloaded to CPU)
#   - Activations   ~8 GB     (full block recompute, 4 layers/stage)
#   - LM head + CE  ~3 GB     (TE chunked CE, no full logits materialization)
#   - Workspace     ~5 GB
#   - Peak total    ~26 GB / 80 GB  → comfortable headroom for transients
#
# Container assumptions: same as run_sft_qwen35_122b_2node_lora.sh (NGC 25.06).
#
# Required env (per-node):
#   MASTER_ADDR, MASTER_PORT  cluster-injected
#   RANK or NODE_RANK         this node's index in [0, NNODES)
#   WORLD_SIZE                number of nodes (default 24)
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_24node_hade.sh
#
# Override knobs:
#   TP=2 PP=12 CP=1 EP=4 ITERS=20 SEQ=98304 GBS=32 MBS=1 ...

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo + paths
#
# Required (no defaults — must be set by the caller):
#   HF_MODEL    path to the Hugging Face model directory
#   MCORE_PATH  path to the converted Megatron-Core checkpoint directory
#
# Optional (derived from REPO_ROOT / MEG_RUN_DIR if not set):
#   REPO_ROOT   Megatron-Bridge repo root (auto-derived from this script's location)
#   MEG_RUN_DIR working root for data / outputs / logs / hf_cache
#               default: sibling of REPO_ROOT named "meg-run"
#   TRAIN_DATA  training JSONL file
#   OUTPUT_DIR  checkpoint save directory
#   LOG_DIR     log directory
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEG_RUN_DIR="${MEG_RUN_DIR:-$(dirname "$REPO_ROOT")/meg-run}"

# Mandatory path variables — fail fast if not set
if [[ -z "${HF_MODEL:-}" ]]; then
    echo "[ERROR] HF_MODEL is not set. Export the path to your HF model directory." >&2
    echo "  export HF_MODEL=/path/to/Qwen3.5-122B-A10B" >&2
    exit 1
fi
if [[ -z "${MCORE_PATH:-}" ]]; then
    echo "[ERROR] MCORE_PATH is not set. Export the path to your Megatron-Core checkpoint." >&2
    echo "  export MCORE_PATH=/path/to/Qwen3.5-122B-A10B-mcore" >&2
    exit 1
fi

TRAIN_DATA="${TRAIN_DATA:-${MEG_RUN_DIR}/demo_data/train_data_demo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_24node_seq98304}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs2}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_24node_seq98304_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-24}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Parallelism overrides (192 GPU / 96K layout — see header for math)
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-12}"
CP="${CP:-1}"
EP="${EP:-4}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-98304}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
DROP_OVERLENGTH="${DROP_OVERLENGTH:-False}"
VALID_SPLIT_RATIO="${VALID_SPLIT_RATIO:-0.05}"
VALID_SPLIT_SEED="${VALID_SPLIT_SEED:-1234}"
EVAL_ITERS="${EVAL_ITERS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

# Per-PP-stage layer count (used for recompute_num_layers — must match exactly,
# otherwise full-block recompute leaves some layers' fwd activations alive).
# 48 model layers / PP=12 = 4 layers per stage.
PP_STAGE_LAYERS="${PP_STAGE_LAYERS:-$(( 48 / PP ))}"

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
# Reduce allocator fragmentation (helps with optimizer-state OOM on 80G GPUs)
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
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export HF_HOME="${HF_HOME:-${MEG_RUN_DIR}/hf_cache}"
mkdir -p "$HF_HOME"
# Prevent thread exhaustion in DataLoader workers (8 GPU procs × 2 workers × rayon threads).
# HF fast tokenizer (Rust/PyO3) spawns a rayon thread pool per worker process;
# on a node with many GPU ranks this hits the OS thread limit (EAGAIN / errno 11),
# causing "The global thread pool has not been initialized" PanicException and
# DataLoader worker exit-code-1 crashes.
# TOKENIZERS_PARALLELISM=false: disables HF tokenizers internal thread pool.
# RAYON_RS_NUM_CPUS=1: caps rayon (Rust) thread pool to 1 thread per worker.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_RS_NUM_CPUS="${RAYON_RS_NUM_CPUS:-1}"

# ---------------------------------------------------------------------------
# Recipe + dataset overrides
# ---------------------------------------------------------------------------
OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    # Use qwen3_vl_step (Qwen3-VL-aware step). For CP=1 this is functionally
    # equivalent to vlm_step (get_batch_on_this_cp_rank is a no-op when cp=1),
    # but kept consistent with the 12-node CP>1 script so any future CP toggle
    # does not need a step_func swap. See Pitfall #24.
    --step_func qwen3_vl_step
    --hf_path "$HF_MODEL"

    # Parallelism (192-GPU / 96K specific override; CP=1)
    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.context_parallel_size="$CP"
    model.expert_model_parallel_size="$EP"
    model.seq_length="$SEQ"

    # Activation recompute — full BLOCK over every layer of every PP stage.
    # PP=12 → 4 layers/stage; recompute_num_layers must equal layers/stage so
    # each stage's full forward is freed before backward starts (any smaller
    # value leaves some layers' fwd activations alive, which is what
    # contributed to the earlier low-sequence OOM during 4-node bring-up).
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers="$PP_STAGE_LAYERS"

    # Disable MTP (Multi-Token Prediction) to relieve last-pipeline-stage memory.
    # MTP adds ~1 extra transformer block + LM-head overhead only on the last PP stage,
    # causing rank(PP-1) OOM while ranks 0..PP-2 are fine. MTP is for inference
    # throughput, not needed for SFT training correctness. (Pitfall #19)
    model.mtp_num_layers=0

    # Cross-entropy: switch to TE chunked impl to avoid materializing the full
    # [B, SEQ, V/TP] logits tensor on the last PP stage.
    #   * SEQ=96K, V=248320, TP=2 → full logits = 1 × 98304 × 124160 × 2B = 22.8 GB
    #     in a single allocation. With recipe default (impl="native" non-chunked)
    #     the last stage OOMs at the LM head before CE even starts.
    #   * impl="te" routes through transformer_engine.pytorch.cross_entropy
    #     which streams logits in chunks; peak is bounded by chunk size, not SEQ.
    #   * cross_entropy_loss_fusion stays True (recipe default) — that's the
    #     fusion ON/OFF switch; impl picks the kernel underneath.
    # CP=1 here so the native↔CP incompatibility (Pitfall #23) does not apply.
    model.cross_entropy_fusion_impl=te

    # GDN constraint (Pitfall #7): pack_sequences_in_batch=True crashes inside
    # the GDN kernel because GDN/linear-attention only supports BSHD, not THD.
    dataset.pack_sequences_in_batch=False

    # Optimizer CPU offload — kept ON for the 96K-bring-up.
    # DP=8 on 192 GPU shards Adam state to ~10 GB/GPU (vs ~30 GB on the 12-node
    # DP=4 layout where offload was load-bearing). In principle DP=8 alone might
    # fit on 80G, but SEQ went 32K → 96K so transient activation + LM-head peak
    # all grew. Keep offload ON for the first run; if iter-1 nvidia-smi shows
    # > 15 GB headroom on the last PP stage, flip OPTIM_OFFLOAD=False to
    # reclaim the CPU↔GPU copy time.
    #
    # Set OPTIM_OFFLOAD_FRAC=0 to disable; OPTIM_OFFLOAD_FRAC=1.0 for max savings.
    # Skill cpu-offloading: optimizer offload is the only option for PP > 1
    # (PP=12 here, so activation offload is forbidden).
    optimizer.optimizer_cpu_offload="${OPTIM_OFFLOAD:-True}"
    optimizer.optimizer_offload_fraction="${OPTIM_OFFLOAD_FRAC:-1.0}"
    optimizer.overlap_cpu_optimizer_d2h_h2d="${OPTIM_OFFLOAD_OVERLAP:-True}"
    optimizer.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPT:-True}"

    # Iters / batch
    train.train_iters="$ITERS"
    train.global_batch_size="$GBS"
    train.micro_batch_size="$MBS"
    validation.eval_iters="$EVAL_ITERS"
    validation.eval_interval="$EVAL_INTERVAL"

    # Checkpoint
    checkpoint.pretrained_checkpoint="$MCORE_PATH"
    checkpoint.save="$OUTPUT_DIR"
    checkpoint.save_interval="$SAVE_INTERVAL"
    checkpoint.load=null

    # Dataset
    dataset.train_data_path="$TRAIN_DATA"
    dataset.hf_processor_path="$HF_MODEL"
    dataset.seq_length="$SEQ"
    dataset.drop_overlength_samples="$DROP_OVERLENGTH"
    dataset.validation_split_ratio="$VALID_SPLIT_RATIO"
    dataset.validation_split_seed="$VALID_SPLIT_SEED"

    # Logging
    logger.log_interval="$LOG_INTERVAL"
    logger.tensorboard_dir=null
    logger.wandb_project=null
)

cat <<EOF
============================================================
Qwen3.5-122B-A10B FULL SFT (24-node, 96K, CP=1)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  CP=${CP}  EP=${EP}   (DP=$(( NNODES * NPROC / TP / PP / CP )))
  Recompute    : full / block / num_layers=${PP_STAGE_LAYERS}
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  CE impl      : te (chunked, prevents 23 GB logits OOM on last PP stage)
  Optim offload: ${OPTIM_OFFLOAD:-True}  (frac=${OPTIM_OFFLOAD_FRAC:-1.0})
  Drop long    : ${DROP_OVERLENGTH}
  Valid split  : ${VALID_SPLIT_RATIO}
  Eval         : every ${EVAL_INTERVAL}, iters=${EVAL_ITERS}
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
