#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-122B-A10B FULL SFT — 32 nodes × 8 GPU (= 256 × H800 80G).
#
# Recipe: qwen35_vl_122b_a10b_sft_config
#   default TP=2, PP=6, EP=8, LR=2e-5, GBS=36, seq=4096
#
# IMPORTANT — parallelism on 256 GPU with 96K context, CP=1:
#   Use TP=2, PP=4, CP=1, EP=8, DP=32.
#   Math: world_size=256, DP=256/(TP×PP×CP)=256/(2×4×1)=32.
#   EP must ≤ DP → EP=8 ≤ DP=32 ✓.
#
#   Why PP=4 (not PP=12 as in the 24-node script)?
#   256 = 2^8 (pure power of two). PP must simultaneously:
#     1. Divide 256 exactly              → PP must be a power of 2
#     2. Divide 48 (model layers) exactly
#     3. Give an integer multiple of 4 layers/stage (GDN cycle = 3 GDN + 1 Attn)
#   Intersection:
#     PP=4  → 48/4=12 layers/stage, 12 mod 4=0 ✓   ← THE ONLY VALID CHOICE
#     PP=8  → 48/8=6  layers/stage, 6  mod 4=2 ✗ (breaks GDN alignment)
#     PP=16 → 48/16=3 layers/stage, 3  mod 4=3 ✗ (breaks GDN alignment)
#   PP=12 from the 24-node script contains factor 3 and does NOT divide 256.
#
#   Layer alignment: model has 48 layers in groups of 4 (3 GDN + 1 Attention).
#   PP=4 → 12 layers/stage = exactly 3 complete groups per stage. ✓
#
#   Pipeline bubble (GBS=32, MBS=1 → 32 microbatches):
#     bubble_fraction = (PP-1)/(PP-1+num_mb) = 3/35 ≈ 8.6%
#     Compare 24-node PP=12: 11/43 ≈ 25.6%  — 256-GPU bubble is much smaller.
#
# Memory safety strategy (CP=1 + SEQ=98304 + PP=4 + TP=2):
#   PP=4 means 12 layers/stage vs 4 layers/stage at PP=12 — 3× more LLM params
#   per GPU. Mitigation stack (identical to 24-node unless noted):
#   1. freeze_vision_model=True, freeze_vision_projection=True
#        — vision lives entirely on PP rank 0; freezing removes vision grads +
#          Adam state + bwd activations. (Same as 24-node.)
#   2. cross_entropy_fusion_impl="te"
#        — SEQ=98304, V=248320, TP=2 → logits = 1×98304×124160×2B = 22.8 GB
#          if materialized. TE chunked CE avoids this. (Same as 24-node.)
#   3. recompute_granularity=full, method=uniform, num_layers=1
#        — 12 layers/stage means 3× more layers to recompute per stage.
#          Still ~5-10% throughput cost but essential given more layers/stage.
#   4. optimizer_cpu_offload=True, fraction=1.0
#        — DP=32 shards Adam state to ~2.5 GB/GPU (vs ~10 GB at DP=8).
#          More headroom than 24-node; consider flipping OFF after iter-1 check.
#   5. mtp_num_layers=0 — Pitfall #19.
#   6. expandable_segments:True — Pitfall #28.
#   7. sequence_parallel=True — TP=2 splits SEQ across 2 ranks.
#   8. NCCL eager init + 60-min timeouts — same rationale as 24-node.
#
# Estimated per-GPU peak (PP=4, TP=2, DP=32, SEQ=98304):
#   PP rank 0 (ranks 0..63, vision encoder + LLM layers 0-11):
#     - LLM params bf16 (12 layers)  ~15 GB  (3× vs PP=12/4-layer stage)
#     - LLM grads  bf16              ~15 GB
#     - LLM Adam                      ~0 GB  (CPU offload)
#     - LLM activations (uniform/1)   ~8 GB
#     - Vision params (frozen fwd)    ~2 GB
#     - Vision activations (uniform/1)~2 GB
#     - Workspace                     ~5 GB
#     - Peak total                   ~47 GB / 80 GB  → safe
#   PP rank 1..2 (LLM only, 12 layers):  ~38 GB / 80 GB
#   PP rank 3 (LM head + CE):            ~41 GB / 80 GB
#   If PP rank 0 OOMs: switch to TP=4, EP=8 (DP=16 ≥ EP=8 ✓).
#
# MoE GroupedGEMM: disabled by default (E1, same as 24-node).
#   logs6/logs7 证伪 TE GroupedLinear 在本环境（torch 2.8, TE 2.4, CUDA 12.9）。
#   MBRIDGE_DISABLE_MOE_GROUPED_GEMM=1 追加 model.moe_grouped_gemm=False。
#
# Required env (per-node):
#   MASTER_ADDR, MASTER_PORT  cluster-injected
#   RANK or NODE_RANK         this node's index in [0, NNODES)
#   WORLD_SIZE                number of nodes (default 32)
#
# Required user-set env:
#   HF_MODEL    path to HuggingFace model directory
#   MCORE_PATH  path to converted Megatron-Core checkpoint directory
#
# Usage (run on EACH node):
#   bash scripts/run_sft_qwen35_122b_32node_hade.sh
#
# Override knobs:
#   TP=2 PP=4 CP=1 EP=8 ITERS=20 SEQ=98304 GBS=32 MBS=1
#   TP=4 EP=8   → DP=16, tighter memory per GPU, useful if PP rank 0 OOMs
#   OPTIM_OFFLOAD=False          flip once iter-1 headroom > 15 GB
#   MOE_Z_LOSS_COEFF=1e-3        router z-loss (default)
#   LR=1e-5                      peak LR (ms-swift reference)
#   LR_WARMUP_ITERS=200
#   MOE_AUX_LOSS_COEFF=          override recipe load-balance coeff
#   MBRIDGE_DISABLE_MOE_GROUPED_GEMM=1   E1 旁路 (default ON)
#   DIST_TIMEOUT_MIN=60
#   DIST_TIMEOUT_SEC=3600

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo + paths
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEG_RUN_DIR="${MEG_RUN_DIR:-$(dirname "$REPO_ROOT")/meg-run}"

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

TRAIN_DATA="${TRAIN_DATA:-${MEG_RUN_DIR}/demo_data/total.v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_122b_full_sft_32node_seq98304}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs_32node}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_full_32node_seq98304_$(date +%Y%m%d_%H%M%S)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-32}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Parallelism (256 GPU / 96K layout — see header for math)
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-4}"
CP="${CP:-1}"
EP="${EP:-8}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_122b_a10b_sft_config}"
ITERS="${ITERS:-20}"
SEQ="${SEQ:-98304}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"

LR="${LR:-1e-5}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-}"
MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-}"
MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-1e-3}"

DROP_OVERLENGTH="${DROP_OVERLENGTH:-False}"
VALID_SPLIT_RATIO="${VALID_SPLIT_RATIO:-0.05}"
VALID_SPLIT_SEED="${VALID_SPLIT_SEED:-1234}"
EVAL_ITERS="${EVAL_ITERS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

# PP=4 → 48/4 = 12 layers/stage (3 complete GDN groups of 4 per stage)
PP_STAGE_LAYERS="${PP_STAGE_LAYERS:-$(( 48 / PP ))}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -d "$REPO_ROOT" ]]               || { echo "[ERROR] missing REPO_ROOT: $REPO_ROOT" >&2; exit 1; }
[[ -d "$MCORE_PATH/iter_0000000" ]] || { echo "[ERROR] missing mcore ckpt: $MCORE_PATH/iter_0000000" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]              || { echo "[ERROR] missing train data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$HF_MODEL" ]]                || { echo "[ERROR] missing HF model: $HF_MODEL" >&2; exit 1; }

_world=$(( NNODES * NPROC ))
_dp=$(( _world / TP / PP / CP ))
if (( _world % (TP * PP * CP) != 0 )); then
    echo "[ERROR] world_size=$_world not divisible by TP×PP×CP=$(( TP * PP * CP ))" >&2; exit 1
fi
if (( EP > _dp )); then
    echo "[ERROR] EP=$EP > DP=$_dp — EP must be ≤ DP" >&2; exit 1
fi
if (( 48 % PP != 0 )); then
    echo "[ERROR] PP=$PP does not divide 48 model layers" >&2; exit 1
fi
if (( (48 / PP) % 4 != 0 )); then
    echo "[WARNING] PP=$PP gives $((48/PP)) layers/stage — not a multiple of 4 (GDN cycle broken)." >&2
fi

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
cd "$REPO_ROOT"

exec > >(tee "$LOG_FILE") 2>&1

# ---------------------------------------------------------------------------
# Venv python
# ---------------------------------------------------------------------------
VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] venv python missing: $VENV_PY" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Install runtime deps (ALL nodes, idempotent)
# ---------------------------------------------------------------------------
echo "[run_sft] node ${NODE_RANK}: running install_runtime_deps.sh..."
bash "$REPO_ROOT/scripts/install_runtime_deps.sh"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export MBRIDGE_PATCH_NVIDIA_FILE="${MBRIDGE_PATCH_NVIDIA_FILE:-0}"
export MBRIDGE_DISABLE_CUDNN="${MBRIDGE_DISABLE_CUDNN:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MBRIDGE_PATCH_GRAD_FUSION="${MBRIDGE_PATCH_GRAD_FUSION:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# NCCL watchdog / heartbeat — 60-min/3600-s same as 24-node (logs5 fix).
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_NCCL_ENABLE_TIMING="${TORCH_NCCL_ENABLE_TIMING:-1}"
export TORCH_NCCL_TRACE_CPP_STACK="${TORCH_NCCL_TRACE_CPP_STACK:-1}"

# Eager NCCL communicator creation (proven fix in logs5→logs6).
export NCCL_RUNTIME_CONNECT="${NCCL_RUNTIME_CONNECT:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_USE_COMM_NONBLOCKING="${TORCH_NCCL_USE_COMM_NONBLOCKING:-0}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"

export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_RS_NUM_CPUS="${RAYON_RS_NUM_CPUS:-1}"
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export HF_HOME="${HF_HOME:-${MEG_RUN_DIR}/hf_cache}"
mkdir -p "$HF_HOME"

# E1: GroupedGEMM 旁路（默认 ON）
: "${MBRIDGE_DISABLE_MOE_GROUPED_GEMM:=1}"
export MBRIDGE_DISABLE_MOE_GROUPED_GEMM

# ---------------------------------------------------------------------------
# Recipe + dataset overrides
# ---------------------------------------------------------------------------
OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    --step_func qwen3_vl_step
    --hf_path "$HF_MODEL"

    # Parallelism (256-GPU / 96K, CP=1)
    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.context_parallel_size="$CP"
    model.expert_model_parallel_size="$EP"
    model.seq_length="$SEQ"

    # NCCL / process-group timeouts (same as 24-node logs5 fix)
    dist.distributed_timeout_minutes="${DIST_TIMEOUT_MIN:-60}"
    dist.distributed_timeout_seconds_after_init="${DIST_TIMEOUT_SEC:-3600}"

    # Activation recompute — full/uniform/1 (same as 24-node)
    # PP=4 means 12 layers/stage; every layer is its own checkpoint window.
    model.recompute_granularity=full
    model.recompute_method=uniform
    model.recompute_num_layers=1

    # Disable MTP (Pitfall #19)
    model.mtp_num_layers=0

    # Freeze vision encoder + projection (same as 24-node)
    model.freeze_vision_model=True
    model.freeze_vision_projection=True

    # Cross-entropy: TE chunked (same need as 24-node: SEQ=98K, TP=2)
    model.cross_entropy_fusion_impl=te

    # GDN constraint (Pitfall #7): no THD support
    dataset.pack_sequences_in_batch=False

    # MoE numerical stability (proven in logs5)
    model.moe_z_loss_coeff="${MOE_Z_LOSS_COEFF}"

    # Optimizer CPU offload
    # DP=32 → Adam ~2.5 GB/GPU; offload is less critical than at DP=8 but
    # keep ON for first run. Flip OPTIM_OFFLOAD=False after iter-1 check.
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

if [[ -n "$LR" ]]; then
    OVERRIDES+=(optimizer.lr="$LR")
fi
if [[ -n "$LR_WARMUP_ITERS" ]]; then
    OVERRIDES+=(scheduler.lr_warmup_iters="$LR_WARMUP_ITERS")
fi
if [[ -n "$MOE_AUX_LOSS_COEFF" ]]; then
    OVERRIDES+=(model.moe_aux_loss_coeff="$MOE_AUX_LOSS_COEFF")
fi
if [[ "$MBRIDGE_DISABLE_MOE_GROUPED_GEMM" == "1" ]]; then
    OVERRIDES+=(model.moe_grouped_gemm=False)
fi

print_launch_diagnostics() {
    local world_gpus=$(( NNODES * NPROC ))
    local dp=$(( world_gpus / TP / PP / CP ))
    local ranks_per_pp=$(( world_gpus / PP ))
    local node_first_rank=$(( NODE_RANK * NPROC ))
    local node_last_rank=$(( node_first_rank + NPROC - 1 ))
    local node_first_pp=$(( node_first_rank / ranks_per_pp ))
    local node_last_pp=$(( node_last_rank / ranks_per_pp ))

    printf '[launch-diag] timestamp=%s\n' "$(date -Is 2>/dev/null || date)"
    printf '[launch-diag] host=%s user=%s pwd=%s\n' "$(hostname 2>/dev/null || printf unknown)" "${USER:-unknown}" "$PWD"
    printf '[launch-diag] node_rank=%s local_procs=%s global_rank_range=%s-%s\n' "$NODE_RANK" "$NPROC" "$node_first_rank" "$node_last_rank"
    printf '[launch-diag] world_gpus=%s tp=%s pp=%s cp=%s ep=%s dp=%s ranks_per_pp=%s node_pp_range=%s-%s\n' \
        "$world_gpus" "$TP" "$PP" "$CP" "$EP" "$dp" "$ranks_per_pp" "$node_first_pp" "$node_last_pp"
    printf '[launch-diag] pp_rank_ranges:'
    local pp_rank
    for (( pp_rank = 0; pp_rank < PP; pp_rank++ )); do
        local pp_first=$(( pp_rank * ranks_per_pp ))
        local pp_last=$(( pp_first + ranks_per_pp - 1 ))
        printf ' pp%s=%s-%s' "$pp_rank" "$pp_first" "$pp_last"
    done
    printf '\n'
    printf '[launch-diag] nccl_env NCCL_DEBUG=%s NCCL_DEBUG_SUBSYS=%s CUDA_DEVICE_MAX_CONNECTIONS=%s PYTORCH_CUDA_ALLOC_CONF=%s\n' \
        "${NCCL_DEBUG:-}" "${NCCL_DEBUG_SUBSYS:-}" "${CUDA_DEVICE_MAX_CONNECTIONS:-}" "${PYTORCH_CUDA_ALLOC_CONF:-}"
    printf '[launch-diag] timeout_env DIST_TIMEOUT_MIN=%s DIST_TIMEOUT_SEC=%s TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=%s\n' \
        "${DIST_TIMEOUT_MIN:-60}" "${DIST_TIMEOUT_SEC:-3600}" "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
    printf '[launch-diag] eager_nccl NCCL_RUNTIME_CONNECT=%s NCCL_NVLS_ENABLE=%s NONBLOCK=%s HIPRIO=%s\n' \
        "${NCCL_RUNTIME_CONNECT:-}" "${NCCL_NVLS_ENABLE:-}" "${TORCH_NCCL_USE_COMM_NONBLOCKING:-}" "${TORCH_NCCL_HIGH_PRIORITY:-}"
    printf '[launch-diag] moe_e1 MBRIDGE_DISABLE_MOE_GROUPED_GEMM=%s\n' "${MBRIDGE_DISABLE_MOE_GROUPED_GEMM:-1}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total --format=csv,noheader,nounits \
            | while IFS= read -r gpu_line; do printf '[launch-diag] gpu %s\n' "$gpu_line"; done
    fi
}

if [[ "${MBRIDGE_LAUNCH_DIAG:-1}" == "1" ]]; then
    print_launch_diagnostics
fi

cat <<EOF
============================================================
Qwen3.5-122B-A10B FULL SFT (32-node / 256-GPU, 96K, CP=1)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC} = $(( NNODES * NPROC )) GPU   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  CP=${CP}  EP=${EP}   (DP=$(( NNODES * NPROC / TP / PP / CP )))
  Layers/stage : ${PP_STAGE_LAYERS}   (48 total, GDN cycle=4; ${PP_STAGE_LAYERS}/4=$((PP_STAGE_LAYERS/4)) groups/stage)
  Bubble frac  : $(( PP - 1 ))/$((PP - 1 + GBS / MBS)) ≈ $(echo "scale=1; (${PP}-1)*100/(${PP}-1+${GBS}/${MBS})" | bc)%   (24-node was ~25.6%)
  Recompute    : full / uniform / num_layers=1
  Vision       : freeze_vision_model=True  freeze_vision_projection=True
  MoE stab     : moe_z_loss_coeff=${MOE_Z_LOSS_COEFF}  aux_coeff=${MOE_AUX_LOSS_COEFF:-recipe-default}
  LR           : lr=${LR:-recipe-default}  warmup_iters=${LR_WARMUP_ITERS:-recipe-default}
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  CE impl      : te (chunked, prevents ~23 GB logits OOM on last PP stage)
  Optim offload: ${OPTIM_OFFLOAD:-True}  (frac=${OPTIM_OFFLOAD_FRAC:-1.0})
  NCCL timeout : init=${DIST_TIMEOUT_MIN:-60}min  steady=${DIST_TIMEOUT_SEC:-3600}s  watchdog=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}s
  MoE bypass   : DISABLE_GROUPED_GEMM=${MBRIDGE_DISABLE_MOE_GROUPED_GEMM}
  Drop long    : ${DROP_OVERLENGTH}
  Valid split  : ${VALID_SPLIT_RATIO}
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
    "${OVERRIDES[@]}"
