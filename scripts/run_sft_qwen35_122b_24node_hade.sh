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
#   1. freeze_vision_model=True, freeze_vision_projection=True
#        — Qwen3-VL ViT lives entirely on PP rank 0 (model.py:170-194,
#          pipeline_model_parallel_size=1 for vision). Without freezing, vision
#          grads + Adam state + bwd-retained activations push PP rank 0 (global
#          ranks [0..15]) to ~77 GB / 80 GB; ranks 8..11 OOM first, ranks 0..7
#          observe ~99% memory. Freezing eliminates vision grads + Adam state +
#          bwd activations entirely. Parity with ms-swift --freeze_vit
#          --freeze_aligner that fits the same model on the same hardware.
#   2. cross_entropy_fusion_impl="te"
#        — chunked CE inside TE, never materializes the full
#          [B, 96K, V/TP] = ~23 GB logits tensor on the last PP stage.
#          Without this, last-stage rank OOMs at the LM head.
#   3. recompute_granularity=full, method=uniform, num_layers=1
#        — every single layer is its own checkpoint window (finest possible).
#          uniform/1 covers BOTH the LLM (4 layers/stage) and the vision
#          encoder (~27-32 layers, all on PP rank 0). The previous block/4
#          setting only covered LLM but left vision activations alive — a
#          contributor to PP rank 0 OOM. Cost: ~5-10% extra forward time.
#   4. optimizer_cpu_offload=True, fraction=1.0
#        — DP=8 already shards Adam state to ~10 GB/GPU vs ~30 GB at DP=4
#          (12-node), but SEQ went 32K→96K so transient activation peak grew.
#          Keep offload ON; flip OPTIM_OFFLOAD=False once a clean iter-1
#          shows headroom > 15 GB.
#   5. mtp_num_layers=0 — Pitfall #19, removes ~6-7 GB extra on last stage.
#   6. expandable_segments:True — Pitfall #28 anti-fragmentation.
#   7. sequence_parallel=True (recipe default for MoE) — TP=2 splits SEQ
#      across TP ranks for non-attention activations, halving most buffers.
#   8. NCCL eager init + 60-min process-group timeout (2026-04-30, third
#      round). MoE all-to-all NCCL groups are LAZILY created on the first
#      collective that uses them. In the PP=12 + EP=4 layout the last PP
#      stage triggers lazy MoE-comm init only after finishing forward of all
#      microbatches, while mid PP stages have already entered recv_backward
#      on PIPELINE_MODEL_PARALLEL_GROUP — at SeqNum=9 the 20 min watchdog
#      tripped on mid stages while last stage was still inside
#      ncclCommInitRankConfig (logs5 evidence). Mitigation:
#        * NCCL_RUNTIME_CONNECT=0 eagerly connects all transports at comm
#          init time, shrinking the lazy-init race window
#        * NCCL_NVLS_ENABLE=0 cuts another lazy-allocation surface
#        * timeouts pushed to 60 min / 3600 s so any residual race still has
#          enough headroom to finish the first iter
#
# Estimated per-GPU peak after freeze + uniform/1:
#   PP rank 0 (ranks 0..15, holds vision encoder + LLM stage 0):
#     - LLM params bf16     ~5 GB
#     - LLM grads  bf16     ~5 GB
#     - LLM Adam            ~0 GB     (CPU offload)
#     - LLM activations     ~3 GB     (uniform/1 recompute, 4 layers in flight)
#     - Vision params bf16  ~2-3 GB   (frozen — fwd only, no grad/Adam)
#     - Vision activations  ~2-4 GB   (uniform/1 recompute on vision too)
#     - Workspace           ~5 GB
#     - Peak total          ~25 GB / 80 GB  → safe margin
#   PP rank 1..10 (LLM only):
#     - Total               ~15 GB / 80 GB  (matches observed ~30 GB before fix)
#   PP rank 11 (LM head + CE):
#     - LM head + CE        ~3 GB     (TE chunked, no full logits tensor)
#     - Total               ~18 GB / 80 GB
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
#   MOE_Z_LOSS_COEFF=1e-3   router-logit z-loss; bound router logit norm.
#                           default 1e-3 (mcore-recommended starting value).
#   LR=2e-5   peak learning rate. set 1e-5 if router z-loss alone doesn't
#             stop the iter-2 NaN.
#   LR_WARMUP_ITERS=200   default 200; set 500 if still unstable.
#   MOE_AUX_LOSS_COEFF=  override the recipe's load-balance loss coeff.
#   MBRIDGE_LOSS_DIAG_VERBOSE=1   force the loss/fwd diag to print every call.
#   MBRIDGE_LOSS_DIAG_FIRST_N=4   first-N calls per rank to fingerprint
#                                 (default 4; raise to debug iter 2+).
#   TORCH_NCCL_TRACE_BUFFER_SIZE=1048576  enable NCCL flight recorder for
#                                        timeout stack/collective dumps.
#   MBRIDGE_LAUNCH_DIAG=1       print per-node launch/NCCL/rank-map diagnostics.
#   DIST_TIMEOUT_MIN=60          init-phase process-group timeout (default 60).
#   DIST_TIMEOUT_SEC=3600        steady-phase process-group timeout (default 3600).
#                                Both bumped 2026-04-30 after a SeqNum=9 hang in
#                                PIPELINE_MODEL_PARALLEL_GROUP triggered by
#                                first-iter MoE all-to-all NCCL group lazy init.
#   NCCL_RUNTIME_CONNECT=0       eager NCCL transport connect at comm init
#                                time (default ON; set =1 to revert to lazy).
#   NCCL_NVLS_ENABLE=0           disable NVLink-Sharp during init storm
#                                (default OFF; set =1 to re-enable).
#   TORCH_NCCL_USE_COMM_NONBLOCKING=0  keep init_process_group blocking
#                                (default; do NOT change unless debugging).
#
# === MoE GroupedGEMM 故障与当前推荐（详见 memory0430.md §9）===
# logs6 / logs7 都炸在同一栈：
#   experts.py:369 → transformer_engine.py:1905 → grouped_linear.py:158
#   → te_general_grouped_gemm → cublaslt_gemm.cu:543
#   "cuBLAS Error: the function failed to launch on the GPU" → 异步 IMA
#
# 已被 logs7 证伪的方向（不要再试）：
#   * 加大 cuBLAS workspace（NVTE_CUBLAS_WORKSPACE_SIZE_BYTES=128MiB +
#     CUBLAS_WORKSPACE_CONFIG=:4096:8）：rank48 仍崩在完全相同的栈
#   * 关 fused bias-gelu nvfuser（NVTE_BIAS_GELU_NVFUSION=0）：同样无效
#
# 当前推荐的唯一下一步：E1 — 直接绕开 TE GroupedLinear
#   MBRIDGE_DISABLE_MOE_GROUPED_GEMM=1 LOG_DIR=/mnt/.../logs8 \
#     bash scripts/run_sft_qwen35_122b_24node_hade.sh
# 这会追加 model.moe_grouped_gemm=False，每个 expert 走普通 GEMM；慢约
# 1.3-1.5×，但完全消除 grouped GEMM 这条故障路径。

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
LOG_DIR="${LOG_DIR:-${MEG_RUN_DIR}/logs7}"
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
# Optional optimizer / MoE numerical-stability overrides. Empty string ""
# means "use recipe default" (no override emitted).
LR="${LR:-}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-}"
MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-}"
DROP_OVERLENGTH="${DROP_OVERLENGTH:-False}"
VALID_SPLIT_RATIO="${VALID_SPLIT_RATIO:-0.05}"
VALID_SPLIT_SEED="${VALID_SPLIT_SEED:-1234}"
EVAL_ITERS="${EVAL_ITERS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

# Per-PP-stage layer count (informational only; printed in the run header).
# Previously fed into recompute_num_layers under method=block; we now use
# method=uniform + recompute_num_layers=1 (every layer is its own checkpoint
# window) so this value is no longer wired into recompute config. 48 / PP=12 = 4.
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

# Capture the entire launcher, not just torchrun. This keeps shell-side
# diagnostics, dependency setup output, and torchrun output in one rank log.
exec > >(tee "$LOG_FILE") 2>&1

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
# NCCL watchdog / heartbeat timeouts. Default torch values (10 min comm, 8 min
# heartbeat) are too tight for 192-GPU init + first-iter compile + cold caches.
#
# 2026-04-30 third-round bump: 20 min was NOT enough. logs5 evidence:
#   * iter 1 forward completed cleanly on last PP stage (call#1..#4 all ok)
#   * mid PP stage (node 6/7, ranks 48-63) timed out at SeqNum=9 in
#     PIPELINE_MODEL_PARALLEL_GROUP after 1240s waiting on recv_backward
#   * Stack: recv_backward → _communicate → batch_isend_irecv → endCoalescing
#   * Last PP stage was busy initializing a brand-new nranks=8 NCCL EP group
#     (commId 0x7e9a..., 0xa964..., 0xe379...) — these are LAZILY created the
#     first time the MoE all-to-all is executed, and were created in different
#     orders on different PP stages, racing with PP recv_backward
# → Fix is two-fold:
#   (1) push every timeout to 60 min so the first-iter NCCL group init storm
#       (compile + cold-cache + lazy MoE comm creation) cannot trigger watchdog
#   (2) eager-create NCCL communicators wherever possible (NCCL_RUNTIME_CONNECT
#       below) so the lazy-init race window shrinks
#
# Two layers of timeout must agree:
#   1. process-group timeout passed to init_process_group (set via the
#      dist.distributed_timeout_minutes / distributed_timeout_seconds_after_init
#      overrides further down — those are the source of truth)
#   2. torch's NCCL watchdog timeouts below — must match or exceed (1) so the
#      watchdog doesn't kill a still-valid collective.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Enable PyTorch NCCL flight recorder. Without a nonzero trace buffer, timeout
# dumps only report the watchdog failure and lose the collective stack/context.
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_NCCL_ENABLE_TIMING="${TORCH_NCCL_ENABLE_TIMING:-1}"
export TORCH_NCCL_TRACE_CPP_STACK="${TORCH_NCCL_TRACE_CPP_STACK:-1}"

# ---------------------------------------------------------------------------
# Eager NCCL communicator creation (mitigates lazy-init deadlock seen in logs5):
#   * NCCL_RUNTIME_CONNECT=0  — eagerly establish all NCCL transport
#     connections at communicator init time, instead of on first send/recv.
#     Without this, MoE all-to-all groups in PP > 1 + EP > 1 setups race with
#     PP send/recv first-iter and can deadlock.
#   * NCCL_NVLS_ENABLE=0     — disable NVLink-Sharp; the multi-stage init it
#     does adds another lazy-allocation surface in PP+EP setups.
#   * TORCH_NCCL_USE_COMM_NONBLOCKING=0 — keep init_process_group calls
#     blocking so we never enter "init pending" state across ranks.
#   * TORCH_NCCL_HIGH_PRIORITY=1 — give NCCL streams higher priority than
#     compute streams during the init storm so dummy collectives complete
#     promptly during warmup.
# All four are env overridable so we can A/B them off without script edits.
# ---------------------------------------------------------------------------
export NCCL_RUNTIME_CONNECT="${NCCL_RUNTIME_CONNECT:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_USE_COMM_NONBLOCKING="${TORCH_NCCL_USE_COMM_NONBLOCKING:-0}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"

# 异常时打全 C++ 栈（无害；只在 RuntimeError 时激活）。logs6/logs7 的 cuBLAS
# 报错栈完整度依赖此项。
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

# E1 — 唯一保留的 grouped-GEMM 旁路开关。默认 OFF；置 1 时在 OVERRIDES 段
# 追加 model.moe_grouped_gemm=False，让每个 expert 走普通 GEMM，绕开 logs6/
# logs7 中崩在 te_general_grouped_gemm 的故障路径。详情见 memory0430.md §9。
: "${MBRIDGE_DISABLE_MOE_GROUPED_GEMM:=0}"
export MBRIDGE_DISABLE_MOE_GROUPED_GEMM

# NCCL kernel launch timeout (legacy NCCL_TIMEOUT removed in 2.x; use the
# per-pg timeout instead, which we set via dist.distributed_timeout_minutes).
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

    # NCCL / process-group timeouts (default torch is 10 min, too tight for
    # 192-GPU init + first-iter compile + cold caches at SEQ=98304):
    #   * distributed_timeout_minutes (= init_process_group timeout):
    #       used during init for the initial rendezvous and any collective
    #       before update_pg_timeout fires.
    #   * distributed_timeout_seconds_after_init:
    #       train.py:350-352 calls update_pg_timeout(timedelta(seconds=...))
    #       at the start of training, replacing the init-time timeout. This is
    #       what protects long collectives during steady-state training (e.g.
    #       a slow CPU offload optimizer step or a stragger straggler in DP).
    # 2026-04-30: bumped from 20 min / 600 s to 60 min / 3600 s after the
    # mid-PP-stage hang at SeqNum=9 (logs5). 20 min was not enough to absorb
    # the first-iter dynamo trace + lazy MoE all-to-all NCCL group init storm
    # on the last PP stage; 60 min gives ~3× margin while still failing fast
    # on real deadlocks. The watchdog/heartbeat env vars above match (3600 s).
    # These large timeouts are first-iter only; from iter 2 onward each
    # collective should complete in < 60 s, so the larger window does not
    # weaken NaN/timeout detection mid-training.
    dist.distributed_timeout_minutes="${DIST_TIMEOUT_MIN:-60}"
    dist.distributed_timeout_seconds_after_init="${DIST_TIMEOUT_SEC:-3600}"

    # Activation recompute — full UNIFORM over every single layer (parity with
    # ms-swift run that fits the same model on the same hardware).
    # Why uniform/1 instead of block/4 (the previous setting):
    #   * recompute_method=block + recompute_num_layers=N recomputes only the
    #     FIRST N layers of a transformer_block. For the LLM (4 layers/stage at
    #     PP=12) block/4 covers all of them, so it was fine. For the *vision
    #     encoder* (Qwen3-VL ViT, ~27-32 layers, lives entirely on PP rank 0,
    #     pipeline_model_parallel_size=1 — see model.py:184), block/4 only
    #     recomputed the first 4 layers and kept the remaining ~24 layers'
    #     forward activations alive. This was the dominant contributor to the
    #     PP rank 0 OOM (ranks 0..15) at SEQ=98304.
    #   * recompute_method=uniform + recompute_num_layers=1 recomputes every
    #     single layer (each layer is its own checkpoint window). This is the
    #     finest possible granularity and is what ms-swift uses.
    #   * Cost: ~5-10% throughput hit for the extra forward of each layer.
    #     Acceptable trade-off vs OOM.
    model.recompute_granularity=full
    model.recompute_method=uniform
    model.recompute_num_layers=1

    # Disable MTP (Multi-Token Prediction) to relieve last-pipeline-stage memory.
    # MTP adds ~1 extra transformer block + LM-head overhead only on the last PP stage,
    # causing rank(PP-1) OOM while ranks 0..PP-2 are fine. MTP is for inference
    # throughput, not needed for SFT training correctness. (Pitfall #19)
    model.mtp_num_layers=0

    # ---------------------------------------------------------------------------
    # Freeze vision encoder + projection/aligner (parity with ms-swift
    # --freeze_vit / --freeze_aligner). Root cause of the 24-node OOM at
    # SEQ=98304:
    #
    #   * vision_model is constructed only when `pre_process and add_encoder`
    #     are both true (model.py:170-194). With Megatron's default rank order
    #     PP rank 0 occupies global ranks [0..15] (24 nodes × 8 GPU / PP=12).
    #     Hence ALL of: vision params + vision grads + vision forward acts +
    #     vision Adam state live on those 16 GPUs.
    #   * vision_transformer_config.pipeline_model_parallel_size = 1
    #     (model.py:184) — vision encoder cannot be PP-sharded.
    #   * vision_model has no CP path either (CP=1 here, and even at CP>1 only
    #     vision_dp_when_cp would split it).
    #   * The OOM stack-tops were inside vision_model.forward (vision_model.py:353
    #     → transformer_block.py:318 → bias_dropout_add). At 77.18 GB allocated
    #     of 80 GB, ranks 8..11 (node 1) tipped first; ranks 0..7 (node 0)
    #     observed at ~99% — the entire PP rank 0 group was on the cliff.
    #
    # Freezing both the ViT backbone and the projection/aligner removes:
    #   - vision grads (bf16, ~ViT params bytes)
    #   - vision Adam state (offloaded but still touches GPU on copy)
    #   - vision activations' backward retention
    # Vision params (forward only) remain on PP rank 0; that fraction is small.
    #
    # Trade-off: vision encoder weights stay at the pretrained values; only the
    # LLM is SFT-tuned. This is the standard SFT pattern for multimodal models
    # and matches the ms-swift run that fit on the same hardware.
    # ---------------------------------------------------------------------------
    model.freeze_vision_model=True
    model.freeze_vision_projection=True

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

    # ---------------------------------------------------------------------------
    # MoE numerical-stability hardening (added 2026-04-30 after iter-2 NaN on
    # ranks 188-191 = ep ∈ {2,3}). Evidence chain (memory0430.md §4 / §8)：
    #   * iter 1 forward ok across ALL ranks (call#1 fwd-diag clean, max ~26)
    #   * iter 2 forward NaN ONLY on ep ∈ {2,3}; ep ∈ {0,1} stayed clean on
    #     the same data and same DP rank
    #   * 一次性 ckpt expert weight fingerprint 全 ok（已确认无 ckpt 污染）
    # → iter 1 backward put NaN/Inf into the expert weights of ep ∈ {2,3}.
    #
    # The recipe has moe_router_topk=8, moe_aux_loss_coeff=1e-3,
    # moe_router_load_balancing_type=global_aux_loss, but moe_z_loss_coeff
    # was unset (default None → no router-logits regularization). With topk=8
    # and no z-loss, router logits are free to grow unboundedly during the
    # first backward pass, which dominates expert grad magnitude unevenly
    # across EP ranks (some experts attract all the heat, others receive
    # ~zero traffic) → the heavily-routed experts on ep ∈ {2,3} blew up first.
    #
    # Hardening (override-able from outside via env var):
    #   * moe_z_loss_coeff = 1e-3 — the "good start value" mentioned in the
    #     mcore TransformerConfig comment for this very field. Bounds router
    #     logit norm by penalising log-sum-exp(logits)^2 in the loss.
    # If after enabling z-loss the run still NaNs, the next levers (NOT
    # enabled here so we keep the variable count low for diagnosis) are:
    #   * --lr 1e-5  (halve LR) and --lr_warmup_iters 500
    #   * --moe_aux_loss_coeff 1e-2 (10× stronger balance push)
    #   * --moe_router_pre_softmax True if available
    # ---------------------------------------------------------------------------
    model.moe_z_loss_coeff="${MOE_Z_LOSS_COEFF:-1e-3}"

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

# Conditional optimizer / MoE overrides — only emitted when caller sets the
# env var. Keeps the override list short and grep-friendly when defaults work.
if [[ -n "$LR" ]]; then
    OVERRIDES+=(optimizer.lr="$LR")
fi
if [[ -n "$LR_WARMUP_ITERS" ]]; then
    OVERRIDES+=(scheduler.lr_warmup_iters="$LR_WARMUP_ITERS")
fi
if [[ -n "$MOE_AUX_LOSS_COEFF" ]]; then
    OVERRIDES+=(model.moe_aux_loss_coeff="$MOE_AUX_LOSS_COEFF")
fi

# E1: 绕开 TE GroupedLinear，每 expert 走普通 GEMM。仅在
# MBRIDGE_DISABLE_MOE_GROUPED_GEMM=1 时启用。
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
    printf '[launch-diag] torch_nccl TORCH_NCCL_TRACE_BUFFER_SIZE=%s TORCH_NCCL_DUMP_ON_TIMEOUT=%s TORCH_NCCL_DESYNC_DEBUG=%s TORCH_NCCL_ENABLE_TIMING=%s TORCH_NCCL_TRACE_CPP_STACK=%s\n' \
        "${TORCH_NCCL_TRACE_BUFFER_SIZE:-}" "${TORCH_NCCL_DUMP_ON_TIMEOUT:-}" "${TORCH_NCCL_DESYNC_DEBUG:-}" "${TORCH_NCCL_ENABLE_TIMING:-}" "${TORCH_NCCL_TRACE_CPP_STACK:-}"
    printf '[launch-diag] timeout_env DIST_TIMEOUT_MIN=%s DIST_TIMEOUT_SEC=%s TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=%s RDZV_TIMEOUT=%s\n' \
        "${DIST_TIMEOUT_MIN:-60}" "${DIST_TIMEOUT_SEC:-3600}" "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}" "${RDZV_TIMEOUT:-1800}"
    printf '[launch-diag] eager_nccl_env NCCL_RUNTIME_CONNECT=%s NCCL_NVLS_ENABLE=%s TORCH_NCCL_USE_COMM_NONBLOCKING=%s TORCH_NCCL_HIGH_PRIORITY=%s\n' \
        "${NCCL_RUNTIME_CONNECT:-}" "${NCCL_NVLS_ENABLE:-}" "${TORCH_NCCL_USE_COMM_NONBLOCKING:-}" "${TORCH_NCCL_HIGH_PRIORITY:-}"
    printf '[launch-diag] diag_env MBRIDGE_LOSS_DIAG_FIRST_N=%s MBRIDGE_LOSS_DIAG_VERBOSE=%s\n' \
        "${MBRIDGE_LOSS_DIAG_FIRST_N:-4}" "${MBRIDGE_LOSS_DIAG_VERBOSE:-0}"
    # MoE 旁路开关（详见 memory0430.md §9）。
    printf '[launch-diag] moe_e1 MBRIDGE_DISABLE_MOE_GROUPED_GEMM=%s TORCH_SHOW_CPP_STACKTRACES=%s\n' \
        "${MBRIDGE_DISABLE_MOE_GROUPED_GEMM:-0}" "${TORCH_SHOW_CPP_STACKTRACES:-}"
    printf '[launch-diag] grep_hints="Watchdog caught|Flight recorder|desync|SeqNum|CollectiveFingerPrint|Stack trace|PIPELINE_MODEL_PARALLEL_GROUP|cuBLAS Error|cublaslt_gemm|illegal memory access|EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP|EXPERT_MODEL_PARALLEL_GROUP|te_general_grouped_gemm"\n'
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total --format=csv,noheader,nounits \
            | while IFS= read -r gpu_line; do printf '[launch-diag] gpu %s\n' "$gpu_line"; done
    else
        printf '[launch-diag] gpu nvidia-smi not found\n'
    fi
}

if [[ "${MBRIDGE_LAUNCH_DIAG:-1}" == "1" ]]; then
    print_launch_diagnostics
fi

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
  Recompute    : full / uniform / num_layers=1   (PP_STAGE_LAYERS=${PP_STAGE_LAYERS} unused)
  Vision       : freeze_vision_model=True  freeze_vision_projection=True
  MoE stab     : moe_z_loss_coeff=${MOE_Z_LOSS_COEFF:-1e-3}  aux_coeff=${MOE_AUX_LOSS_COEFF:-recipe-default}
  LR override  : lr=${LR:-recipe-default}  warmup_iters=${LR_WARMUP_ITERS:-recipe-default}
  Iters/GBS    : ${ITERS} / ${GBS}    MBS=${MBS}
  Seq length   : ${SEQ}
  CE impl      : te (chunked, prevents 23 GB logits OOM on last PP stage)
  Optim offload: ${OPTIM_OFFLOAD:-True}  (frac=${OPTIM_OFFLOAD_FRAC:-1.0})
  NCCL timeout : init=${DIST_TIMEOUT_MIN:-60}min  steady=${DIST_TIMEOUT_SEC:-3600}s  watchdog=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}s
  NCCL eager   : RUNTIME_CONNECT=${NCCL_RUNTIME_CONNECT}  NVLS=${NCCL_NVLS_ENABLE}  NONBLOCK=${TORCH_NCCL_USE_COMM_NONBLOCKING}  HIPRIO=${TORCH_NCCL_HIGH_PRIORITY}
  NCCL trace   : buffer=${TORCH_NCCL_TRACE_BUFFER_SIZE}  dump=${TORCH_NCCL_DUMP_ON_TIMEOUT}  desync=${TORCH_NCCL_DESYNC_DEBUG}
  MoE bypass   : DISABLE_GROUPED_GEMM=${MBRIDGE_DISABLE_MOE_GROUPED_GEMM}  CPP_STACK=${TORCH_SHOW_CPP_STACKTRACES}
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
    "${OVERRIDES[@]}"
