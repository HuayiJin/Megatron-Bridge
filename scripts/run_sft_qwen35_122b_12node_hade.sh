#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 12 nodes × 8 GPU (= 96 × H800 80G).
#
# Recipe: qwen35_vl_122b_a10b_sft_config
#   default TP=2, PP=6, EP=8, LR=2e-5, GBS=36, seq=4096
#
# IMPORTANT — parallelism on 96 GPU with 32K context:
#   Use TP=2, PP=6, CP=2, EP=4, DP=4.
#   Math: world_size=96, DP=96/(TP×PP×CP)=96/(2×6×2)=4.
#   EP must ≤ DP → EP=4.
#   CP=2 keeps each rank's attention context shard at 16K tokens.
#
#   Layer alignment: model has 48 layers in groups of 4.
#   PP=6 → 8 layers/stage = 2 complete groups. Boundaries are aligned. ✓
#
# Memory note (122B BF16 + Adam, TP=2 PP=6 CP=2 EP=4 DP=4):
#   - Params bf16     ~244 GB total → ~7.6 GB/GPU (distributed across TP×PP×EP)
#   - Grads  bf16     same as params
#   - Adam (m,v) fp32, ZeRO-1 → sharded across DP=4 → ~30 GB/GPU
#   Per-GPU peak ≈ 70-78 GB on 80G GPUs — very tight.
#
# Container assumptions: same as run_sft_qwen35_122b_2node_lora.sh (NGC 25.06).
#
# Required env (per-node):
#   MASTER_ADDR, MASTER_PORT  cluster-injected
#   RANK or NODE_RANK         this node's index in [0, NNODES)
#   WORLD_SIZE                number of nodes (default 12)
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_12node.sh
#
# Override knobs:
#   TP=2 PP=6 CP=2 EP=4 ITERS=20 SEQ=32768 GBS=32 MBS=1 ...

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
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_12node_seq32768}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs2}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_12node_seq32768_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-12}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Parallelism overrides (96 GPU / 32K layout — see header for math)
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-6}"
CP="${CP:-2}"
EP="${EP:-4}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-32768}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
DROP_OVERLENGTH="${DROP_OVERLENGTH:-False}"
VALID_SPLIT_RATIO="${VALID_SPLIT_RATIO:-0.05}"
VALID_SPLIT_SEED="${VALID_SPLIT_SEED:-1234}"
EVAL_ITERS="${EVAL_ITERS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
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
    # IMPORTANT: use qwen3_vl_step (NOT the generic vlm_step) for any Qwen3-VL /
    # Qwen3.5-VL family run with CP > 1.
    #   * vlm_step.get_batch returns full-length labels and relies on the model
    #     to do CP slicing in its `if self.pre_process` branch — this only runs
    #     on PP rank 0. The last PP stage (where the LM head + CE live) then
    #     sees logits sliced to [B, SEQ/CP] but labels still [B, SEQ]; CE
    #     `logits_2d[arange_1d, masked_target_1d]` blows up (manifested as the
    #     dynamo broadcast error in fused CE; the unfused path silently
    #     mis-indexes).
    #   * qwen3_vl_step.forward_step calls get_batch_on_this_cp_rank on
    #     forward_args, slicing labels/loss_mask consistently with the model's
    #     own CP slice in *every* PP stage.
    # See Pitfall #24.
    --step_func qwen3_vl_step
    --hf_path "$HF_MODEL"

    # Parallelism (96-GPU / 32K specific override)
    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.context_parallel_size="$CP"
    model.expert_model_parallel_size="$EP"
    model.seq_length="$SEQ"

    # Activation recompute (full BLOCK — every layer of every PP stage is
    # recomputed end-to-end, freeing all per-layer GDN bwd workspace as soon
    # as the layer's bwd finishes). With PP=6 each stage has 8 layers.
    # uniform+1 (the previous setting) only recomputed in chunks of 1 layer
    # which still kept many layers' fwd activations alive at once during bwd
    # — contributed to Pitfall #23 OOM during earlier low-sequence bring-up.
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers=8

    # Disable MTP (Multi-Token Prediction) to relieve last-pipeline-stage memory.
    # MTP adds ~1 extra transformer block + LM-head overhead only on the last PP stage,
    # causing rank3 OOM while rank0-2 are fine. MTP is for inference throughput, not
    # needed for SFT training correctness.
    model.mtp_num_layers=0

    # CP > 1 required flags (config.py:1224-1228).
    # calculate_per_token_loss=True: avoids NaN loss on CP ranks whose
    # context shard contains only masked/padding tokens in SFT.
    # average_in_collective=False: incompatible with per-token loss scaling.
    model.calculate_per_token_loss=True
    ddp.average_in_collective=False

    # Disable fused (jit_fuser/torch.compile) cross-entropy under CP > 1.
    # The native fused path in mcore/fusions/fused_cross_entropy.py compiles
    # `logits_2d[arange_1d, masked_target_1d]` via dynamo. With CP, logits
    # are sharded on the sequence dim (len = SEQ/CP) but `target` keeps the
    # full sequence length (len = SEQ), so dynamo's fake-tensor shape check
    # fails with:
    #   "Attempting to broadcast a dimension of length <SEQ> at -1!
    #    Mismatching argument at index 1 had torch.Size([SEQ]); but
    #    expected shape should be broadcastable to [SEQ/CP]"
    # The training-config validator now auto-disables this when
    # context_parallel_size > 1 (see training/config.py around the
    # "Auto-disable native fused cross-entropy" block, plus Pitfall #23).
    # We still set it here explicitly so the per-run config is self-documenting
    # and survives any future validator change.
    model.cross_entropy_loss_fusion=False

    # GDN constraint
    dataset.pack_sequences_in_batch=False

    # Optimizer CPU offload (Pitfall #28, 2026-04-26).
    # The 122B model leaves limited 80G headroom for long-context activations
    # plus optimizer-state lazy alloc.  Adam's exp_avg + exp_avg_sq buffers
    # (~30 GB per GPU after DP=4 sharding for this 12-node TP=2 PP=6 CP=2 EP=4
    # layout — DP=4, not 8) are LAZILY allocated inside
    # fused_adam.initialize_state on the first optimizer.step().  Combined with
    # iter-0 fwd/bwd transients (~21 GB peak), the alloc crosses 80 G and OOMs.
    #
    # Fix: move all Adam state to CPU.  HybridDeviceOptimizer keeps Adam
    # math on CPU (slower step but compute is small fraction of total).
    # Skill cpu-offloading: optimizer offload is the only option for PP > 1.
    # PP=6 here, so activation offload is forbidden.
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
Qwen3.5-122B-A10B FULL SFT (12-node)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC}   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  CP=${CP}  EP=${EP}
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  Drop long     : ${DROP_OVERLENGTH}
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
