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

"""Launch a chosen training script on every Ray node.

Run this manually on rank 0 after every node has executed run/start_ray.sh:

    /opt/venv-mbridge/bin/python run/run_on_all_nodes.py scripts/run_sft_qwen35_122b_24node_hade.sh

Ray is only used as the fan-out layer.  By default, this program first kills
stale training processes on every Ray node, then submits exactly one task to each
node.  That task runs the specified shell script, and the shell script is
responsible for launching torchrun with all local GPUs.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import ray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "run_sft_qwen35_122b_24node_hade.sh"
DEFAULT_MASTER_PORT = "23456"


def _resolve_path(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


def _parse_env_assignment(assignment: str) -> tuple[str, str]:
    if "=" not in assignment:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got: {assignment}")
    key, value = assignment.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"empty env key in: {assignment}")
    return key, value


def _load_env_file(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    with open(path, encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = _parse_env_assignment(line)
            env[key] = value
    return env


@ray.remote
def run_train_script(train_script: str, injected_env: dict[str, str]) -> dict[str, str]:
    """Run the selected training script on one Ray node."""
    merged_env = os.environ.copy()

    # Ray may set CUDA_VISIBLE_DEVICES for its worker.  torchrun must see all
    # GPUs on the node because the training scripts derive nproc_per_node from
    # the local device count/defaults.
    merged_env.pop("CUDA_VISIBLE_DEVICES", None)
    merged_env.update(injected_env)

    try:
        result = subprocess.run(
            ["bash", train_script],
            cwd=merged_env.get("REPO_ROOT", str(REPO_ROOT)),
            env=merged_env,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "status": "success",
            "hostname": socket.gethostname(),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "hostname": socket.gethostname(),
            "stdout": exc.stdout,
            "stderr": exc.stderr,
            "returncode": str(exc.returncode),
        }


@ray.remote
def cleanup_training_processes() -> dict[str, str]:
    """Kill stale non-Ray Python processes on this node.

    This cleanup task itself is running inside a Ray Python worker, so a literal
    `pkill -9 -f python` would also kill Ray and make it impossible for rank 0 to
    submit the follow-up training task.  We therefore kill all Python processes
    except the Ray control/worker processes that are required for this fan-out.
    """
    current_pid = os.getpid()
    ps_result = subprocess.run(
        ["ps", "-eo", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if ps_result.returncode != 0:
        return {
            "status": "error",
            "hostname": socket.gethostname(),
            "message": f"ps failed: {ps_result.stderr.strip()}",
        }

    protected_markers = ("ray", "run_on_all_nodes.py")
    target_pids: list[str] = []
    skipped: list[str] = []
    for line in ps_result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid, command = parts
        if not pid.isdigit() or int(pid) == current_pid:
            continue
        if "python" not in command:
            continue
        if any(marker in command for marker in protected_markers):
            skipped.append(f"{pid}:{command}")
            continue
        target_pids.append(pid)

    if not target_pids:
        return {
            "status": "success",
            "hostname": socket.gethostname(),
            "message": f"no stale non-Ray python processes found; skipped {len(skipped)} Ray/control python processes",
        }

    kill_result = subprocess.run(
        ["kill", "-9", *target_pids],
        capture_output=True,
        text=True,
        check=False,
    )
    message = f"killed python pids: {', '.join(target_pids)}; skipped {len(skipped)} Ray/control python processes"
    if kill_result.returncode not in (0,):
        message += f"; kill rc={kill_result.returncode} stderr={kill_result.stderr.strip()}"
    return {"status": "success", "hostname": socket.gethostname(), "message": message}


def _alive_nodes() -> list[dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes found")
    return nodes


def _node_ip(node: dict[str, Any]) -> str:
    return str(node["NodeManagerAddress"])


def _ordered_nodes(alive_nodes: list[dict[str, Any]], master_addr: str) -> list[dict[str, Any]]:
    """Return a deterministic node order with the torchrun master first."""
    return sorted(alive_nodes, key=lambda node: (0 if _node_ip(node) == master_addr else 1, _node_ip(node)))


def cleanup_all_nodes(alive_nodes: list[dict[str, Any]]) -> None:
    logger.info("========================================")
    logger.info("开始清理所有节点上的历史训练进程...")
    logger.info("========================================")

    cleanup_futures = []
    for node in alive_nodes:
        node_ip = _node_ip(node)
        future = cleanup_training_processes.options(resources={f"node:{node_ip}": 0.01}).remote()
        cleanup_futures.append((node_ip, future))

    for node_ip, future in cleanup_futures:
        result = ray.get(future)
        logger.info("Node %s (%s): %s", node_ip, result.get("hostname", "?"), result["message"])

    logger.info("清理完成！")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_script",
        nargs="?",
        default=os.environ.get("TRAIN_SCRIPT", str(DEFAULT_TRAIN_SCRIPT)),
        help="Training script to run on every Ray node.",
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR") or os.environ.get("RAY_HEAD_IP"),
        help="torchrun rendezvous address. Defaults to MASTER_ADDR/RAY_HEAD_IP, then the first Ray node.",
    )
    parser.add_argument(
        "--master-port",
        default=os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT),
        help="torchrun rendezvous port.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Only kill stale training processes on all Ray nodes, then exit.",
    )
    parser.add_argument(
        "--no-cleanup-before-run",
        action="store_true",
        help="Skip the default cleanup step before launching the training script.",
    )
    parser.add_argument(
        "--env",
        action="append",
        type=_parse_env_assignment,
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable injected into every node. Can be repeated.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="File with KEY=VALUE lines to inject into every node. Later files/--env override earlier values.",
    )
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        metavar="KEY",
        help="Copy an environment variable from this rank-0 process into every node. Can be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)

    ray.init(address="auto")
    try:
        alive_nodes = _alive_nodes()
        if args.cleanup:
            cleanup_all_nodes(alive_nodes)
            return 0

        train_script = _resolve_path(args.train_script)
        if not Path(train_script).is_file():
            logger.error("training script not found: %s", train_script)
            return 2

        user_env: dict[str, str] = {}
        for env_file in args.env_file:
            user_env.update(_load_env_file(env_file))
        for key in args.pass_env:
            if key in os.environ:
                user_env[key] = os.environ[key]
            else:
                logger.warning("--pass-env %s ignored because it is not set in the driver environment", key)
        user_env.update(dict(args.env))

        master_addr = args.master_addr or _node_ip(alive_nodes[0])
        ordered_nodes = _ordered_nodes(alive_nodes, master_addr)
        world_size = len(ordered_nodes)

        if not args.no_cleanup_before_run:
            cleanup_all_nodes(alive_nodes)

        logger.info("检测到 %d 个活跃 Ray 节点，准备下发训练任务...", world_size)
        logger.info("train_script=%s", train_script)
        logger.info("MASTER=%s:%s", master_addr, args.master_port)

        futures = []
        for node_rank, node in enumerate(ordered_nodes):
            node_ip = _node_ip(node)
            injected_env = {
                **user_env,
                "REPO_ROOT": str(REPO_ROOT),
                "RANK": str(node_rank),
                "NODE_RANK": str(node_rank),
                "WORLD_SIZE": str(world_size),
                "NNODES": str(world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(args.master_port),
            }
            future = run_train_script.options(resources={f"node:{node_ip}": 0.01}).remote(
                train_script,
                injected_env,
            )
            futures.append((node_rank, node_ip, future))
            logger.info("submitted node_rank=%d node_ip=%s", node_rank, node_ip)

        failed = False
        for node_rank, node_ip, future in futures:
            result = ray.get(future)
            status = result["status"]
            failed = failed or status != "success"
            logger.info("========================================")
            logger.info(
                "Node rank: %d | IP: %s | Host: %s | Status: %s",
                node_rank,
                node_ip,
                result.get("hostname", "?"),
                status,
            )
            if result.get("returncode"):
                logger.info("Return code: %s", result["returncode"])
            if result.get("stdout"):
                logger.info("--- STDOUT ---\n%s", result["stdout"])
            if result.get("stderr"):
                logger.info("--- STDERR ---\n%s", result["stderr"])

        return 1 if failed else 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
