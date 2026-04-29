# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import subprocess
import sys

import ray

logger = logging.getLogger(__name__)

# ── 脚本级别注入的环境变量（优先级高于节点本地环境变量）──────────────────────
# 这些变量会在每个节点上与该节点自身的环境变量合并，
# 若 key 重复，以下值会覆盖节点本地的值。
INJECTED_ENV_VARS: dict[str, str] = {
    "HF_MODEL": "/mnt/tidal-alsh01/dataset/pai/models/Qwen3.5/Qwen3.5-122B-A10B",
    "MCORE_PATH": "/mnt/tidal-alsh01/dataset/pai/hade/data/Qwen3.5-122B-A10B-mcore",
    "MEG_RUN_DIR": "/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run",
    "TRAIN_DATA": "/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/demo_data/train200.jsonl",
    "OUTPUT_DIR": "/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/qwen35_122b_full_sft",
    "WHEEL_DIR": "/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/wheels",
}

# 启动脚本的绝对路径
START_SCRIPT = "/mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge/run/start.sh"


@ray.remote
def run_bash_script(injected_env: dict[str, str]) -> dict[str, str]:
    """在远程节点上执行 bash 脚本。

    环境变量合并策略（优先级从低到高）：
      1. 节点本地环境变量（os.environ）
      2. injected_env 中指定的变量（调用方注入，会覆盖同名本地变量）

    Args:
        injected_env: 从调用方传入的额外环境变量字典。

    Returns:
        包含 status、stdout、stderr 的结果字典。
    """
    # 以节点本地环境变量为基础
    merged_env = os.environ.copy()

    # 移除 Ray 对 GPU 的限制，让子进程能看到机器上所有 GPU
    merged_env.pop("CUDA_VISIBLE_DEVICES", None)

    # 将注入的变量覆盖合并进去（注入变量优先级更高）
    merged_env.update(injected_env)

    try:
        result = subprocess.run(
            ["bash", START_SCRIPT],
            env=merged_env,
            capture_output=True,
            text=True,
            check=True,
        )
        return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "stdout": e.stdout, "stderr": e.stderr}


@ray.remote
def cleanup_python_processes() -> dict[str, str]:
    """在远程节点上杀死所有 Python 进程。

    Returns:
        包含 status 和 message 的结果字典。
    """
    try:
        subprocess.run(["pkill", "-9", "-f", "python"], capture_output=True, text=True)
        return {"status": "success", "message": "Killed python processes"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}


def cleanup_all_nodes(alive_nodes: list[dict]) -> None:
    """在所有节点上杀死 Python 进程。

    Args:
        alive_nodes: 活跃节点信息列表，每项包含 NodeManagerAddress 等字段。
    """
    logger.info("========================================")
    logger.info("开始清理所有节点上的 Python 进程...")
    logger.info("========================================")

    cleanup_futures = []
    for node in alive_nodes:
        node_ip = node["NodeManagerAddress"]
        future = cleanup_python_processes.options(resources={f"node:{node_ip}": 0.01}).remote()
        cleanup_futures.append((node_ip, future))

    for ip, future in cleanup_futures:
        result = ray.get(future)
        logger.info("Node %s: %s", ip, result["message"])

    logger.info("清理完成！")


def main() -> None:
    """主入口：连接 Ray 集群，并在所有存活节点上下发训练任务。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 连接到当前已经运行的 Ray 集群；'auto' 会自动寻找本地启动的 Ray 进程
    ray.init(address="auto")

    nodes = ray.nodes()
    alive_nodes = [node for node in nodes if node["Alive"]]
    logger.info("检测到 %d 个活跃节点，准备下发任务...", len(alive_nodes))

    # 若传入 --cleanup 参数，则只执行清理后退出
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_all_nodes(alive_nodes)
        ray.shutdown()
        return

    # 遍历每个节点，将任务强制调度到该节点上。
    # Ray 默认为每个节点分配了一个名为 "node:<IP>" 的资源；
    # 指定 0.01 个自定义资源可将任务绑定到特定节点，同时不占用真实算力。
    futures = []
    for node in alive_nodes:
        node_ip = node["NodeManagerAddress"]
        future = run_bash_script.options(
            resources={f"node:{node_ip}": 0.01},
        ).remote(INJECTED_ENV_VARS)
        futures.append((node_ip, future))

    # 收集并打印所有节点的执行结果
    for ip, future in futures:
        result = ray.get(future)
        logger.info("========================================")
        logger.info("Node IP: %s | Status: %s", ip, result["status"])
        if result["stdout"]:
            logger.info("--- STDOUT ---\n%s", result["stdout"])
        if result["stderr"]:
            logger.info("--- STDERR ---\n%s", result["stderr"])

    ray.shutdown()


if __name__ == "__main__":
    main()
