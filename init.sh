#!/usr/bin/env bash
# 环境初始化脚本：在当前镜像（Python 3.11）上安装 megatron-bridge 并执行转换
set -e

# ============================================================
# Step 1: 升级 transformers（需 >= 5.2.0 支持 qwen3_5_moe）
# ============================================================
pip install "transformers>=5.2.0"

# ============================================================
# Step 2: 安装额外缺失依赖（系统镜像未自带）
# ============================================================
pip install nvidia-modelopt --no-deps -q
pip install pulp -q

# ============================================================
# Step 3: 放开 pyproject.toml Python 版本限制（镜像是 3.11，包要求 >=3.12）
# ============================================================
sed -i 's/requires-python = ">=3.12,<3.13"/requires-python = ">=3.11"/' \
    /data/temp/Megatron-Bridge/pyproject.toml

sed -i 's/requires-python = ">=3.12"/requires-python = ">=3.11"/' \
    /data/temp/Megatron-Bridge/3rdparty/Megatron-LM/pyproject.toml

# ============================================================
# Step 4: 安装 megatron-core（submodule 版本，替换系统 0.13.0）
#   系统自带 megatron-core 0.13.0，但 Megatron-Bridge 需要更新版本
#   （缺少 unwrap_model 等符号）
# ============================================================
pip install -e /data/temp/Megatron-Bridge/3rdparty/Megatron-LM/ --no-deps -q

# ============================================================
# Step 5: 安装 megatron-bridge 本体（不拉依赖，系统已有全套）
# ============================================================
pip install -e /data/temp/Megatron-Bridge --no-deps -q

# ============================================================
# Step 6: 执行 HF → Megatron checkpoint 转换
# ============================================================
export WORKSPACE=/data/temp/ckpts
mkdir -p "${WORKSPACE}"

python /data/temp/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
    --hf-model /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B \
    --megatron-path "${WORKSPACE}/Qwen3.5-122B-A10B" \
    --torch-dtype bfloat16
