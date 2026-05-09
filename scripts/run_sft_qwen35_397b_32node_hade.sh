#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Qwen3.5-397B-A17B FULL SFT — 32 nodes × 8 GPU (= 256 × H800 80G).
#
# Fit-first 96K context layout for the 397B MoE model.
#
# IMPORTANT — parallelism on 256 GPU with 96K context:
#   Use TP=2, PP=16, CP=4, EP=16, ETP=1.
#   Attention/non-expert DP: 256/(TP×PP×CP)=256/(2×16×4)=2.
#   Expert DP: 256/(ETP×EP×PP)=256/(1×16×16)=1.
#   CP=4 shards 96K to 24K local context per attention rank.
#
#   397B has 60 language layers. PP=16 does not divide 60 uniformly, so this
#   script uses first/last pipeline stages with 2 layers each:
#     2 + 14×4 + 2 = 60.
#   This keeps the embedding and loss stages lighter while keeping middle stages
#   at 4 layers/stage.
#
# Keep the same memory-saving settings as the verified 122B 32-node baseline:
#   1. freeze_vision_model=True, freeze_vision_projection=True
#   2. full block activation recompute over each pipeline stage
#   3. optimizer CPU offload
#   4. MTP disabled
#   5. fused cross entropy disabled under CP
#   6. FLA autotune disabled
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
#   bash scripts/run_sft_qwen35_397b_32node_hade.sh
#
# Override knobs:
#   TP=2 PP=16 CP=4 EP=16 ETP=1 ITERS=20 SEQ=98304 GBS=32 MBS=1
# python /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge/run/run_on_all_nodes.py /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge/scripts/run_sft_qwen35_397b_32node_hade.sh --env-file /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge/run/env-0509-397.txt --master-port 23456 --master-addr $(hostname -i)

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo + paths
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEG_RUN_DIR="${MEG_RUN_DIR:-$(dirname "$REPO_ROOT")/meg-run}"

if [[ -z "${HF_MODEL:-}" ]]; then
    echo "[ERROR] HF_MODEL is not set. Export the path to your HF model directory." >&2
    echo "  export HF_MODEL=/path/to/Qwen3.5-397B-A17B" >&2
    exit 1
fi
if [[ -z "${MCORE_PATH:-}" ]]; then
    echo "[ERROR] MCORE_PATH is not set. Export the path to your Megatron-Core checkpoint." >&2
    echo "  export MCORE_PATH=/path/to/Qwen3.5-397B-A17B-mcore" >&2
    exit 1
fi

TRAIN_DATA="${TRAIN_DATA:-${MEG_RUN_DIR}/demo_data/total.v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEG_RUN_DIR}/qwen35_397b_full_sft_32node_seq98304}"
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs_397b_32node}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_397b_full_32node_seq98304_$(date +%Y%m%d_%H%M)_rank${RANK:-0}.log}"

# ---------------------------------------------------------------------------
# Distributed config
# ---------------------------------------------------------------------------
NPROC="${NPROC:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-32}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Parallelism (256-GPU / 96K layout — see header for math)
# ---------------------------------------------------------------------------
TP="${TP:-2}"
PP="${PP:-16}"
CP="${CP:-4}"
EP="${EP:-16}"
ETP="${ETP:-1}"

TOTAL_LAYERS="${TOTAL_LAYERS:-60}"
FIRST_PP_LAYERS="${FIRST_PP_LAYERS:-2}"
LAST_PP_LAYERS="${LAST_PP_LAYERS:-2}"
MIDDLE_PP_LAYERS="${MIDDLE_PP_LAYERS:-}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RECIPE="${RECIPE:-qwen35_vl_397b_a17b_sft_config}"
ITERS="${ITERS:-500}"
SEQ="${SEQ:-98304}"
GBS="${GBS:-32}"
MBS="${MBS:-1}"
DROP_OVERLENGTH="${DROP_OVERLENGTH:-True}"

VALID_SPLIT_RATIO="${VALID_SPLIT_RATIO:-0.05}"
VALID_SPLIT_SEED="${VALID_SPLIT_SEED:-1234}"
EVAL_ITERS="${EVAL_ITERS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

OPTIM_OFFLOAD="${OPTIM_OFFLOAD:-True}"
OPTIM_OFFLOAD_FRAC="${OPTIM_OFFLOAD_FRAC:-1.0}"
OPTIM_OFFLOAD_OVERLAP="${OPTIM_OFFLOAD_OVERLAP:-True}"
USE_PRECISION_AWARE_OPT="${USE_PRECISION_AWARE_OPT:-True}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
[[ -d "$REPO_ROOT" ]]               || { echo "[ERROR] missing REPO_ROOT: $REPO_ROOT" >&2; exit 1; }
[[ -d "$MCORE_PATH/iter_0000000" ]] || { echo "[ERROR] missing mcore ckpt: $MCORE_PATH/iter_0000000" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]              || { echo "[ERROR] missing train data: $TRAIN_DATA" >&2; exit 1; }
[[ -d "$HF_MODEL" ]]                || { echo "[ERROR] missing HF model: $HF_MODEL" >&2; exit 1; }
[[ -f "$HF_MODEL/config.json" ]]    || { echo "[ERROR] missing HF config: $HF_MODEL/config.json" >&2; exit 1; }

_world=$(( NNODES * NPROC ))
_attn_dp=$(( _world / TP / PP / CP ))
_expert_dp=$(( _world / ETP / EP / PP ))
if (( _world % (TP * PP * CP) != 0 )); then
    echo "[ERROR] world_size=$_world not divisible by TP×PP×CP=$(( TP * PP * CP ))" >&2; exit 1
fi
if (( _world % (ETP * EP * PP) != 0 )); then
    echo "[ERROR] world_size=$_world not divisible by ETP×EP×PP=$(( ETP * EP * PP ))" >&2; exit 1
fi
if (( SEQ % (2 * CP) != 0 )); then
    echo "[ERROR] SEQ=$SEQ must be divisible by 2×CP=$(( 2 * CP ))" >&2; exit 1
fi
if (( PP <= 2 )); then
    echo "[ERROR] PP=$PP must be greater than 2 when first/last PP layers are set" >&2; exit 1
fi
if (( TOTAL_LAYERS - FIRST_PP_LAYERS - LAST_PP_LAYERS <= 0 )); then
    echo "[ERROR] invalid first/last layer split for TOTAL_LAYERS=$TOTAL_LAYERS" >&2; exit 1
fi
if (( (TOTAL_LAYERS - FIRST_PP_LAYERS - LAST_PP_LAYERS) % (PP - 2) != 0 )); then
    echo "[ERROR] middle layers $(( TOTAL_LAYERS - FIRST_PP_LAYERS - LAST_PP_LAYERS )) not divisible by middle PP stages $(( PP - 2 ))" >&2; exit 1
fi
if [[ -z "$MIDDLE_PP_LAYERS" ]]; then
    MIDDLE_PP_LAYERS=$(( (TOTAL_LAYERS - FIRST_PP_LAYERS - LAST_PP_LAYERS) / (PP - 2) ))
fi
if (( MIDDLE_PP_LAYERS <= 0 )); then
    echo "[ERROR] MIDDLE_PP_LAYERS=$MIDDLE_PP_LAYERS must be greater than 0" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Venv python
# ---------------------------------------------------------------------------
VENV_PY="${VENV_PY:-/opt/venv-mbridge/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] venv python missing: $VENV_PY" >&2; exit 1
fi

CHECK_MODEL_CONFIG="${CHECK_MODEL_CONFIG:-1}"
if [[ "$CHECK_MODEL_CONFIG" == "1" ]]; then
    "$VENV_PY" - <<PY
import json
import re
import sys
from pathlib import Path

hf_config_path = Path("$HF_MODEL") / "config.json"
mcore_run_config_path = Path("$MCORE_PATH") / "iter_0000000" / "run_config.yaml"

with hf_config_path.open() as f:
    hf_config = json.load(f)
text_config = hf_config.get("text_config", {})

hf_layers = text_config.get("num_hidden_layers")
hf_experts = text_config.get("num_experts")
hf_topk = text_config.get("num_experts_per_tok")
model_type = hf_config.get("model_type")

errors = []
if model_type != "qwen3_5_moe":
    errors.append(f"model_type={model_type!r}, expected 'qwen3_5_moe'")
if hf_layers != $TOTAL_LAYERS:
    errors.append(f"HF num_hidden_layers={hf_layers}, expected $TOTAL_LAYERS for Qwen3.5-397B-A17B")
if hf_experts != 512:
    errors.append(f"HF num_experts={hf_experts}, expected 512 for Qwen3.5-397B-A17B")
if hf_topk != 10:
    errors.append(f"HF num_experts_per_tok={hf_topk}, expected 10 for Qwen3.5-397B-A17B")

if mcore_run_config_path.exists():
    run_config = mcore_run_config_path.read_text()
    mcore_layers_match = re.search(r"(?m)^  num_layers: (\d+)$", run_config)
    mcore_experts_match = re.search(r"(?m)^  num_moe_experts: (\d+)$", run_config)
    mcore_layers = int(mcore_layers_match.group(1)) if mcore_layers_match else None
    mcore_experts = int(mcore_experts_match.group(1)) if mcore_experts_match else None
    if mcore_layers != $TOTAL_LAYERS:
        errors.append(f"MCore num_layers={mcore_layers}, expected $TOTAL_LAYERS for Qwen3.5-397B-A17B")
    if mcore_experts != 512:
        errors.append(f"MCore num_moe_experts={mcore_experts}, expected 512 for Qwen3.5-397B-A17B")

if errors:
    print("[ERROR] model config preflight failed; this script is for Qwen3.5-397B-A17B:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(f"  HF_MODEL={hf_config_path.parent}", file=sys.stderr)
    print(f"  MCORE_PATH={mcore_run_config_path.parents[1]}", file=sys.stderr)
    sys.exit(1)
PY
fi

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
cd "$REPO_ROOT"

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
export FLA_AUTOTUNE="${FLA_AUTOTUNE:-0}"
export HF_HOME="${HF_HOME:-${MEG_RUN_DIR}/hf_cache}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_RS_NUM_CPUS="${RAYON_RS_NUM_CPUS:-1}"

# ---------------------------------------------------------------------------
# Recipe + dataset overrides
# ---------------------------------------------------------------------------
OVERRIDES=(
    --recipe "$RECIPE"
    --dataset vlm-preloaded
    --step_func qwen3_vl_step
    --hf_path "$HF_MODEL"

    # Parallelism (256-GPU / 96K, PP=16, CP=4; local context = 24K)
    model.tensor_model_parallel_size="$TP"
    model.pipeline_model_parallel_size="$PP"
    model.context_parallel_size="$CP"
    model.expert_model_parallel_size="$EP"
    model.expert_tensor_parallel_size="$ETP"
    model.virtual_pipeline_model_parallel_size=null
    model.num_layers_in_first_pipeline_stage="$FIRST_PP_LAYERS"
    model.num_layers_in_last_pipeline_stage="$LAST_PP_LAYERS"
    model.seq_length="$SEQ"

    dist.distributed_timeout_minutes="${DIST_TIMEOUT_MINUTES:-30}"

    # Activation memory.
    model.recompute_granularity=full
    model.recompute_method=block
    model.recompute_num_layers="$MIDDLE_PP_LAYERS"

    model.mtp_num_layers=0

    # Vision memory: keep frozen for long-context SFT.
    model.freeze_vision_model=True
    model.freeze_vision_projection=True

    # CP > 1 required flags copied from the verified 122B path.
    model.calculate_per_token_loss=True
    ddp.average_in_collective=False

    # Disable fused CE under CP; qwen3_vl_step slices labels per CP rank.
    model.cross_entropy_loss_fusion=False
    model.moe_permute_fusion=False

    # GDN constraint: no THD support.
    dataset.pack_sequences_in_batch=False

    # Optimizer memory.
    optimizer.optimizer_cpu_offload="$OPTIM_OFFLOAD"
    optimizer.optimizer_offload_fraction="$OPTIM_OFFLOAD_FRAC"
    optimizer.overlap_cpu_optimizer_d2h_h2d="$OPTIM_OFFLOAD_OVERLAP"
    optimizer.use_precision_aware_optimizer="$USE_PRECISION_AWARE_OPT"

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
    dataset.dataloader_type=cyclic

    # Logging
    logger.log_interval="$LOG_INTERVAL"
    logger.tensorboard_dir=null
    logger.wandb_project=null
)

cat <<EOF
============================================================
Qwen3.5-397B-A17B FULL SFT (32-node / 256-GPU, 96K, PP=16, CP=4)
  Recipe       : $RECIPE
  HF model dir : $HF_MODEL
  Mcore base   : $MCORE_PATH
  Train data   : $TRAIN_DATA
  Output       : $OUTPUT_DIR
  Nodes/GPUs   : ${NNODES} × ${NPROC} = $(( NNODES * NPROC )) GPU   (this is rank ${NODE_RANK})
  Master       : ${MASTER_ADDR}:${MASTER_PORT}
  Parallelism  : TP=${TP}  PP=${PP}  CP=${CP}  EP=${EP}  ETP=${ETP}
  Attention DP : ${_attn_dp}
  Expert DP    : ${_expert_dp}
  Layers/stage : first=${FIRST_PP_LAYERS}, middle=${MIDDLE_PP_LAYERS}, last=${LAST_PP_LAYERS}  (${TOTAL_LAYERS} total)
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  Drop long     : ${DROP_OVERLENGTH}
  Local context: $(( SEQ / CP )) per CP rank
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
    "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"
