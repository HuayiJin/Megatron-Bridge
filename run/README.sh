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

# 3. 每个 node/container 上启动或加入 Ray 集群。
#    rank 0 会继续下发训练任务；其他 rank 必须 hold 住容器，否则 Ray worker 会随容器退出而掉线。
cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge
bash run/start_ray.sh

NODE_RANK_VALUE="${NODE_RANK:-${RANK:-0}}"
if [[ "$NODE_RANK_VALUE" -eq 0 ]]; then
    /opt/venv-mbridge/bin/python run/run_on_all_nodes.py scripts/run_sft_qwen35_122b_24node_hade.sh
else
    sleep infinity
fi

# 96 节点 demo
python run/run_on_all_nodes.py scripts/run_sft_qwen35_122b_12node_hade.sh --env-file run/env20-10.txt

# 192 节点 demo
/opt/venv-mbridge/bin/python /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/run/run_on_all_nodes.py /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/scripts/run_sft_qwen35_122b_24node_hade.sh --env-file /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/run/env20-10.txt --master-port 23456 --master-addr $(hostname -i)

# 256 节点 demo
/opt/venv-mbridge/bin/python /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/run/run_on_all_nodes.py /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/scripts/run_sft_qwen35_122b_32node_hade.sh --env-file /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/run/env5k-30-20.txt --master-port 23456 --master-addr $(hostname -i)

# rushb
/opt/venv-mbridge/bin/python /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/run/run_on_all_nodes.py /mnt/tidal-alsh01/dataset/pai/zhaoxiang02/run_rushb.sh

# clean all
/opt/venv-mbridge/bin/python /mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge-dbg/run/run_on_all_nodes.py --cleanup