#!/usr/bin/env bash
# =============================================================================
# Megatron-Bridge — 容器启动命令速查
# =============================================================================

# 1. 必须设置的环境变量（未设置会立即报错退出）
export HF_MODEL=/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B   # HF 原始模型目录
export MCORE_PATH=/mnt/tidal-alsh01/dataset/redone/hade/data/Qwen3.5-122B-A10B-mcore        # MCore 格式检查点目录

# 2. 可选配置（均有合理默认值，按需覆盖）
export MEG_RUN_DIR=/mnt/tidal-alsh01/dataset/redone/hade/dd/meg-run                   # 按实际路径修改
export TRAIN_DATA="${MEG_RUN_DIR}/demo_data/train_data_demo.jsonl"                    # 按实际路径修改
export OUTPUT_DIR="${MEG_RUN_DIR}/qwen35_122b_full_sft"
export WHEEL_DIR="${MEG_RUN_DIR}/wheels"

# 网络代理（按集群实际情况设置或删除）
export https_proxy=http://10.7.4.2:3128
export HTTPS_PROXY=http://10.7.4.2:3128

# 3. 安装运行时依赖
cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge
bash scripts/install_runtime_deps.sh

# 4. 执行
bash run/start_4node.sh
