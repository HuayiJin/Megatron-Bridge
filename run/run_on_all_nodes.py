import ray
import os
import subprocess
import sys

# 1. 连接到当前已经运行的 Ray 集群
# 如果在 master 节点运行，'auto' 会自动寻找本地启动的 Ray 进程
ray.init(address="auto")

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


# 2. 定义一个 Ray 远程任务，使用 subprocess 执行 bash 脚本
@ray.remote
def run_bash_script(injected_env: dict[str, str]):
    """在远程节点上执行 bash 脚本。

    环境变量合并策略（优先级从低到高）：
      1. 节点本地环境变量（os.environ）
      2. injected_env 中指定的变量（调用方注入，会覆盖同名本地变量）

    Args:
        injected_env: 从调用方传入的额外环境变量字典。
    """
    try:
        # 以节点本地环境变量为基础
        merged_env = os.environ.copy()

        # 移除 Ray 对 GPU 的限制，让子进程能看到机器上所有 GPU
        merged_env.pop("CUDA_VISIBLE_DEVICES", None)

        # 将注入的变量覆盖合并进去（注入变量优先级更高）
        merged_env.update(injected_env)

        # 执行 bash 脚本，并捕获标准输出和标准错误
        result = subprocess.run(
            ["bash", "/mnt/tidal-alsh01/dataset/pai/hade/dd/Megatron-Bridge/run/start.sh"],
            env=merged_env,
            capture_output=True,
            text=True,
            check=True,
        )
        return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "stdout": e.stdout, "stderr": e.stderr}


# 2.1 定义清理任务：杀死所有 Python 进程
@ray.remote
def cleanup_python_processes():
    try:
        # 杀死所有 python 进程（除了 Ray worker 本身）
        result = subprocess.run(["pkill", "-9", "-f", "python"], capture_output=True, text=True)
        return {"status": "success", "message": "Killed python processes"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 3. 获取集群中所有节点的信息
nodes = ray.nodes()

# 筛选出当前存活（Alive）的节点
alive_nodes = [node for node in nodes if node["Alive"]]
print(f"检测到 {len(alive_nodes)} 个活跃节点，准备下发任务...")

# 4. 遍历每个节点，将任务强制调度到该节点上
futures = []
for node in alive_nodes:
    node_ip = node["NodeManagerAddress"]

    # Ray 默认为每个节点分配了一个名为 "node:<IP>" 的资源
    # 利用这个特性，我们可以通过指定资源需求，将任务绑定到特定的 IP 节点上
    # 分配 0.01 个自定义资源是为了防止占用真实算力，确保任务能顺利下发
    future = run_bash_script.options(
        resources={f"node:{node_ip}": 0.01},
    ).remote(INJECTED_ENV_VARS)

    futures.append((node_ip, future))

# 5. 收集并打印所有节点的执行结果
for ip, future in futures:
    result = ray.get(future)
    print(f"========================================")
    print(f"Node IP: {ip} | Status: {result['status']}")
    if result["stdout"]:
        print(f"--- STDOUT ---\n{result['stdout']}")
    if result["stderr"]:
        print(f"--- STDERR ---\n{result['stderr']}")


# 6. 清理函数：杀死所有远程 Python 进程
def cleanup_all_nodes():
    """在所有节点上杀死 Python 进程"""
    print("\n========================================")
    print("开始清理所有节点上的 Python 进程...")
    print("========================================\n")

    cleanup_futures = []
    for node in alive_nodes:
        node_ip = node["NodeManagerAddress"]
        future = cleanup_python_processes.options(resources={f"node:{node_ip}": 0.01}).remote()
        cleanup_futures.append((node_ip, future))

    for ip, future in cleanup_futures:
        result = ray.get(future)
        print(f"Node {ip}: {result['message']}")

    print("\n清理完成！")


# 7. 如果传入 --cleanup 参数，则只执行清理
if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
    cleanup_all_nodes()
    ray.shutdown()
    sys.exit(0)
