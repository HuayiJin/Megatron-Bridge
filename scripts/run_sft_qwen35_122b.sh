#!/usr/bin/env bash
# Qwen3.5-122B-A10B SFT launcher.
#
# Modes:
#   smoke   1 node × 8 GPU, very few iters, mock-ish, just verifies SFT can step
#   full    Multi-node SFT (the recipe default expects 4 × 8 = 32 GPUs TP=2 PP=6 EP=8)
#
# Single-node 8-GPU is NOT enough for full bf16 SFT of 122B; for development
# we use TP=1 PP=8 EP=1 + recompute=full + small GBS as a shape-correctness smoke.
#
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT and run the script
# on each node.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/temp/Megatron-Bridge}"
HF_MODEL="${HF_MODEL:-/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B}"
MCORE_PATH="${MCORE_PATH:-/data/temp/workspace/models/Qwen3.5-122B-A10B-mcore}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/temp/workspace/runs/qwen35_122b_sft}"
LOG_DIR="${LOG_DIR:-/data/temp/workspace/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sft_$(date +%Y%m%d_%H%M%S).log}"

MODE="${1:-smoke}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC="${NPROC:-8}"

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"

# Verify ckpt exists
if [[ ! -d "$MCORE_PATH" ]]; then
    echo "[ERROR] mcore checkpoint dir missing: $MCORE_PATH"
    echo "        Run scripts/run_convert_qwen35_122b.sh first."
    exit 1
fi

cd "$REPO_ROOT"

# cuDNN sublib workaround: prepend venv libs (also covered by Python-side patch)
VENV_LIB="$REPO_ROOT/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$VENV_LIB/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

# Mode-dependent overrides
case "$MODE" in
smoke)
    # Single-node 8-GPU SFT smoke. Use TP=1 PP=8 EP=1 with HEAVY recompute,
    # tiny iters, GBS=8, MBS=1. Goal: verify the train step runs and loss decreases.
    # NOTE: 122B bf16 + Adam is well over 1 TB; this WILL OOM. Use this only
    #       as a sanity step on a small dummy mcore ckpt or accept OOM and
    #       jump straight to multi-node.
    OVERRIDES=(
        --recipe qwen35_vl_122b_a10b_sft_config
        --dataset vlm-hf
        --step_func vlm_step
        --hf_path "$HF_MODEL"
        # Parallelism overrides for single-node (fits worse than the default 4-node)
        model.tensor_model_parallel_size=1
        model.pipeline_model_parallel_size=8
        model.expert_model_parallel_size=1
        model.expert_tensor_parallel_size=1
        model.sequence_parallel=False
        # Workarounds
        model.gradient_accumulation_fusion=false
        model.cuda_graph_impl=none
        # Memory
        model.recompute_granularity=full
        model.recompute_method=uniform
        model.recompute_num_layers=1
        # Iters / batch
        train.train_iters=5
        train.global_batch_size=8
        train.micro_batch_size=1
        train.eval_iters=0
        train.eval_interval=1000000
        # Checkpoint
        checkpoint.pretrained_checkpoint="$MCORE_PATH"
        checkpoint.save="$OUTPUT_DIR"
        checkpoint.save_interval=1000000
        checkpoint.load=null
        # Logging
        logger.log_interval=1
        logger.tensorboard_dir=null
        logger.wandb_project=null
        # Dataset (small public OCR dataset, fits in HF cache)
        # NOTE: full maker function name (recipe default = make_cord_v2_dataset)
        dataset.maker_name=make_cord_v2_dataset
        dataset.hf_processor_path="$HF_MODEL"
        dataset.seq_length=2048
        # Mixed precision (bf16, no fp8)
        mixed_precision=bf16_mixed
    )
    ;;
full)
    # Multi-node full SFT with the recipe default TP=2 PP=6 EP=8 (4 nodes × 8 = 32 GPU).
    OVERRIDES=(
        --recipe qwen35_vl_122b_a10b_sft_config
        --dataset vlm-hf
        --step_func vlm_step
        --hf_path "$HF_MODEL"
        # Workarounds (still required even on multi-node)
        model.gradient_accumulation_fusion=false
        model.cuda_graph_impl=none
        # Iters / batch (recipe defaults: GBS=36, train_iters=300000, MBS=4)
        train.global_batch_size="${GBS:-32}"
        train.micro_batch_size="${MBS:-1}"
        train.train_iters="${ITERS:-1000}"
        # Checkpoint
        checkpoint.pretrained_checkpoint="$MCORE_PATH"
        checkpoint.save="$OUTPUT_DIR"
        checkpoint.save_interval="${SAVE_INTERVAL:-500}"
        checkpoint.load=null
        # Logging
        logger.log_interval=1
        # Dataset (full maker function name)
        dataset.maker_name="${DATASET:-make_cord_v2_dataset}"
        dataset.hf_processor_path="$HF_MODEL"
    )
    ;;
*)
    echo "Usage: $0 {smoke|full}"
    exit 1
    ;;
esac

echo "============================================================"
echo "Qwen3.5-122B-A10B SFT  (mode=$MODE)"
echo "  HF       : $HF_MODEL"
echo "  Mcore    : $MCORE_PATH"
echo "  Output   : $OUTPUT_DIR"
echo "  Nodes    : $NNODES (rank $NODE_RANK), nproc=$NPROC"
echo "  Master   : $MASTER_ADDR:$MASTER_PORT"
echo "  Log      : $LOG_FILE"
echo "============================================================"

PYTHONUNBUFFERED=1 \
NCCL_DEBUG=WARN \
TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
.venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/sft_qwen35_122b.py \
    "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"
