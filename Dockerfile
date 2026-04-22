# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Qwen3.5-122B-A10B VLM 训练镜像
# 基于 nvcr.io/nvidia/pytorch:25.06-py3
# 功能：HF → Megatron ckpt 转换 + SFT 训练（含 MTP）
#
# Base 环境（来自 pytorch:25.06-py3）：
#   Python 3.11.13 / torch 2.9.1 / CUDA 12.8
#   flash-attn 2.7.3 / transformer-engine 2.10.0
#   transformers 5.5.4 / megatron-core 0.13.0（将被替换）

FROM nvcr.io/nvidia/pytorch:25.06-py3

# ============================================================
# 基础环境变量
# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================================
# Step 1: 拷贝 Megatron-Bridge 代码（含 submodule）
# ============================================================
COPY Megatron-Bridge /opt/Megatron-Bridge

# ============================================================
# Step 2: 放开 Python 版本限制（代码要求 >=3.12，镜像是 3.11）
# ============================================================
RUN sed -i 's/requires-python = ">=3.12,<3.13"/requires-python = ">=3.11"/' \
        /opt/Megatron-Bridge/pyproject.toml && \
    sed -i 's/requires-python = ">=3.12"/requires-python = ">=3.11"/' \
        /opt/Megatron-Bridge/3rdparty/Megatron-LM/pyproject.toml

# ============================================================
# Step 3: 升级 transformers（需 >= 5.2.0 支持 qwen3_5_moe）
# ============================================================
RUN pip install "transformers>=5.2.0"

# ============================================================
# Step 4: 安装额外缺失依赖
#   - nvidia-modelopt: AutoBridge 的量化感知模块依赖
#   - pulp: modelopt 内部依赖（LP solver）
# ============================================================
RUN pip install nvidia-modelopt --no-deps && \
    pip install pulp

# ============================================================
# Step 5: 安装 megatron-core（submodule 版本）
#   替换系统自带的 0.13.0，Megatron-Bridge 需要更新版本
#   （缺少 unwrap_model 等新增符号）
# ============================================================
RUN pip install -e /opt/Megatron-Bridge/3rdparty/Megatron-LM/ --no-deps

# ============================================================
# Step 6: 安装 megatron-bridge 本体（不拉依赖，系统已有全套）
# ============================================================
RUN pip install -e /opt/Megatron-Bridge --no-deps

# ============================================================
# Step 7: 验证安装
# ============================================================
RUN python -c "from megatron.bridge import AutoBridge; print('megatron-bridge OK')"

WORKDIR /opt/Megatron-Bridge
