# Megatron Bridge Skills 索引

本文档汇总了 `skills/` 目录下所有可用技能的说明与用途，供开发者快速查阅。

---

## 目录

- [顶层技能](#顶层技能)
  - [adding-model-support](#adding-model-support)
  - [recipe-recommender](#recipe-recommender)
  - [mlm-bridge-training](#mlm-bridge-training)
  - [parity-testing](#parity-testing)
  - [resiliency](#resiliency)
  - [test-system](#test-system)
  - [multi-node-slurm](#multi-node-slurm)
  - [developer-guide](#developer-guide)
  - [code-style](#code-style)
- [性能优化技能 (perf-techniques)](#性能优化技能-perf-techniques)
- [推荐阅读顺序](#推荐阅读顺序)

---

## 顶层技能

### adding-model-support

**文件**: [`skills/adding-model-support/SKILL.md`](skills/adding-model-support/SKILL.md)  
**适用场景**: 为新 LLM 或 VLM 模型添加支持，包含 bridge、provider、recipe、tests、docs 和 examples 的完整流程。当需要接入、支持、上线或集成新模型时使用。

**流程概览**:

| 阶段 | 内容 |
|------|------|
| Phase 1: Discovery | 获取 HF config.json，分析模型结构（LLM/VLM/MoE/MLA），检查 FP8 量化 |
| Phase 2: Bridge | 实现 `_bridge.py`（config + weight mappings），VLM 需额外 `_provider.py` 和模型类 |
| Phase 3: Recipe | 创建预训练/SFT/PEFT recipe 函数 |
| Phase 4: Tests | 编写功能测试（round-trip、forward parity、training parity） |
| Phase 5: Docs & Examples | 补充文档和转换示例 |

**关键注意事项**:
- FP8 量化模型需在转换前手动 dequantize：`weight.to(bfloat16) * weight_scale_inv`
- VLM 的 `tie_word_embeddings` 在顶层 config，不在 `text_config`
- 模型专属代码放在 `modeling_<model>/` 目录，不污染共享代码

---

### recipe-recommender

**文件**: [`skills/recipe-recommender/SKILL.md`](skills/recipe-recommender/SKILL.md)  
**适用场景**: 根据模型名称/大小、GPU 数量与类型、训练目标（pretrain/SFT/PEFT）推荐合适的 Bridge recipe。

**入口命令**:
```bash
# 使用 mock 数据预训练
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe <recipe_function_name> --dataset llm-pretrain-mock

# CLI 覆盖参数
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe llama3_8b_pretrain_config \
    'model.tensor_model_parallel_size=2' 'training.global_batch_size=64'
```

**已收录的主要模型 Recipe**:

| 模型系列 | 可用 size | 模式 |
|---------|-----------|------|
| Llama 2/3/3.1 | 7B, 8B, 70B, 405B | Pretrain / SFT / PEFT |
| Qwen2 / Qwen2.5 | 500M–72B | Pretrain / SFT / PEFT |
| Qwen3 Dense | 600M–32B | Pretrain / SFT / PEFT |
| Qwen3 MoE | 30B-A3B, 235B-A22B | Pretrain / SFT / PEFT |
| Qwen3-Next MoE | 80B-A3B | Pretrain / SFT / PEFT |
| DeepSeek V2/V3 | Lite, V2, V3 (2048 GPU), 32-node | Pretrain |
| Gemma 2/3 | 1B–27B | Pretrain / SFT / PEFT |
| GLM-4.5 | Air-106B, 355B | Pretrain / SFT / PEFT |
| NemotronH | 4B, 8B, 47B, 56B | Pretrain / SFT / PEFT |
| Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL | 各 size | SFT / PEFT |
| Gemma3-VL | 4B, 12B, 27B | SFT / PEFT |
| Kimi-K2 | 1T MoE | Pretrain |

> **注意**: 性能 recipe（perf recipes）仅用于吞吐量基准测试，不保证训练正确性，须单独验证 loss 曲线。

---

### mlm-bridge-training

**文件**: [`skills/mlm-bridge-training/SKILL.md`](skills/mlm-bridge-training/SKILL.md)  
**适用场景**: 使用 mock 或真实数据运行 MLM（Megatron-LM）和 Megatron Bridge 训练，进行 loss 对比测试，翻译配置。

**核心用途**:
- 验证 MLM 与 Bridge 的 loss 数值一致性（使用 `vanilla_gpt_pretrain_config`）
- 查询 MLM ↔ Bridge 参数映射表（见 `docs/megatron-lm-to-megatron-bridge.md`）
- 管理 Megatron-Core 子模块版本（`./scripts/switch_mcore.sh status/dev/main`）

**常见陷阱**:
- 每次相关测试前必须 `rm -rf nemo_experiments`（Bridge 会自动续训旧 checkpoint）
- 必须用 `uv run`，不能用裸 `torchrun` 或 `python`
- 覆盖 `train_iters` 时同步设置 `scheduler.lr_warmup_iters` 和 `lr_decay_iters`
- MoE 大模型通常需要全量 activation recompute 和多节点 EP，TP **不能**减少单 GPU 的 expert 内存

---

### parity-testing

**文件**: [`skills/parity-testing/SKILL.md`](skills/parity-testing/SKILL.md)  
**适用场景**: 验证 HF ↔ MCore 权重转换的数值一致性，选择正确的验证工具，调试权重不匹配。

**工具选择速查**（所有工具位于 `examples/conversion/`）:

| 验证目标 | 工具 | 需要 GPU? |
|---------|------|-----------|
| 所有权重精确 round-trip（单 GPU） | `hf_megatron_roundtrip.py` | 否 |
| TP/PP/EP 下权重 round-trip | `hf_megatron_roundtrip_multi_gpu.py` | 是 |
| 前向传播 logit 等价性 | `compare_hf_and_megatron/compare.py` | 是 |
| 文本生成 sanity check | `hf_to_megatron_generate_text.py` | 是 |
| VLM 生成 sanity check | `hf_to_megatron_generate_vlm.py` | 是 |
| 代码内权重验证 | `weights_verification_table()` | 是 |

**3 级测试策略**:

| 级别 | 验证内容 | 通过标准 |
|------|---------|---------|
| Level 1 | State Dict Round-Trip | `max_diff == 0.0`（精确匹配） |
| Level 2 | Forward-Pass Parity（bfloat16） | cosine similarity > 0.9999 |
| Level 3 | Training Parity（可选） | loss 持续下降 |

**调试顺序**: 单 GPU round-trip → 多 GPU round-trip → forward pass → 检查 `provider_bridge()` config 映射

---

### resiliency

**文件**: [`skills/resiliency/SKILL.md`](skills/resiliency/SKILL.md)  
**适用场景**: 配置容错、straggler 检测、in-process 重启、抢占（preemption）和 checkpoint 恢复。

**核心功能**:

| 功能 | 配置类 / Plugin | 主要参数 |
|------|----------------|---------|
| 故障容错 (Slurm) | `FaultTolerancePlugin` | `num_in_job_restarts=3`, `num_job_retries_on_failure=2`, `rank_heartbeat_timeout=300` |
| Straggler 检测 | `NVRxStragglerDetectionConfig` | `gpu_relative_perf_threshold=0.7`, `stop_if_detected=False` |
| 抢占处理 | `PreemptionPlugin` | `preempt_time=60`（作业结束前 N 秒发信号） |
| Re-run 状态机 | `RerunStateMachineConfig` | `check_for_nan_in_loss=True`, `check_for_spiky_loss=False` |
| In-process 重启（实验性） | `InProcessRestartConfig` | `granularity="node"`, `soft_timeout=60`, `hard_timeout=90` |

**参考文档**: `docs/training/resiliency.md`、`docs/training/checkpointing.md`

---

### test-system

**文件**: [`skills/test-system/SKILL.md`](skills/test-system/SKILL.md)  
**适用场景**: 了解 CI 测试目录布局、tier 语义，以及如何添加、移动、禁用测试。

**目录结构**:
```
tests/functional_tests/launch_scripts/
  h100/active/    # H100 自动运行（阻塞 CI）
  h100/flaky/     # H100 隔离（不阻塞 CI）
  gb200/active/   # GB200 自动运行
  gb200/flaky/    # GB200 隔离
```

**Tier 说明**:

| Tier | 触发条件 | 是否阻塞 PR |
|------|---------|------------|
| L0 | 每个 PR、每次 push to main、定时 | 是 |
| L1 | push to main 或 `needs-more-tests` label | 是 |
| L2 | 定时或手动触发 | 是（触发时） |
| flaky | 手动 `test_suite=all` | 否 |

**操作速查**:
- **添加新测试**: 在 `active/` 下新建 `{Tier}_{Description}.sh`，首行写 `# CI_TIMEOUT=<minutes>`，`chmod +x`
- **移动到 flaky**: `git mv tests/.../active/L0_Foo.sh tests/.../flaky/L0_Foo.sh`
- **删除测试**: 直接删文件，无需修改 workflow

---

### multi-node-slurm

**文件**: [`skills/multi-node-slurm/SKILL.md`](skills/multi-node-slurm/SKILL.md)  
**适用场景**: 将单节点脚本转为多节点 Slurm sbatch 作业，调试多节点故障（NCCL 超时、OOM、端口冲突等）。

**两种方式对比**:

| 方式 | ntasks-per-node | 适用场景 |
|------|-----------------|---------|
| **srun-native**（推荐） | 8 | Bridge 训练、转换、推理脚本 |
| **uv run torch.distributed**（旧方式） | 1 | MLM `pretrain_gpt.py` |

**关键要点**:
- `NEMO_HOME` 必须指向共享文件系统（Lustre），否则多节点 SFT/PEFT 的 packed-sequence 数据不可见
- 两阶段 srun：Phase 1 单进程热身 uv cache，Phase 2 全量运行（Phase 2 的 `uv sync` 为 no-op）
- Bridge 通过 `common_utils.py` 自动从 SLURM 环境变量推导 `RANK`、`WORLD_SIZE`、`MASTER_ADDR`，无需手动设置
- 旧方式中 `ntasks-per-node` 必须为 1，否则 8 tasks × 8 procs = 64 进程/节点，导致端口冲突

**参考示例**: `examples/models/vlm/glm_45v/slurm_sft.sh`

---

### developer-guide

**文件**: [`skills/developer-guide/SKILL.md`](skills/developer-guide/SKILL.md)  
**适用场景**: 开发环境搭建、CI/CD 流程、uv 包管理、pre-commit hooks、CI 失败排查，以及 uv.lock 问题处理。

**核心原则**: 在容器内开发，始终使用 `uv run`。

**常用命令**:

| 任务 | 命令 |
|------|------|
| 从 lockfile 安装所有依赖 | `uv sync --locked` |
| 运行脚本 | `uv run python script.py` |
| 运行分布式训练 | `uv run python -m torch.distributed.run --nproc_per_node=N script.py` |
| 添加依赖 | `uv add <package>` |
| 重新生成 lockfile | `uv lock`（必须在容器内 Linux 上，macOS 不可用） |
| 运行 lint | `uv run ruff check --fix . && uv run ruff format .` |
| 安装 pre-commit hooks | `uv run --group dev pre-commit install` |

**容器选项**:
- NeMo Framework 容器（开箱即用，推荐）: `nvcr.io/nvidia/nemo:latest`
- 自建 CI 镜像: `docker build -f docker/Dockerfile.ci --target megatron_bridge -t megatron-bridge:latest .`

---

### code-style

**文件**: [`skills/code-style/SKILL.md`](skills/code-style/SKILL.md)  
**适用场景**: 编写新代码或 code review 时自动应用，确保代码风格符合规范。

**主要规范**:

| 类别 | 规范 |
|------|------|
| Python 版本 | 3.10+，Google Python Style Guide |
| 格式化 | `ruff`，行长 119，双引号 |
| 类型注解 | 所有 public API 必须有；用 `X \| Y`（不用 `Union`）；用内置泛型 `list`/`dict` |
| 命名 | 类 PascalCase，函数/变量 snake_case，全局变量加 `G_` 前缀，常量 UPPER_SNAKE |
| 日志 | `logging.getLogger(__name__)`，分布式场景用 `print_rank_0` |
| 文档字符串 | Google style，public 函数和类必须有 |
| 关键字参数 | 有多个同类型参数时用 `*` 强制 keyword-only |

---

## 性能优化技能 (perf-techniques)

所有子技能位于 [`skills/perf-techniques/<name>/SKILL.md`](skills/perf-techniques/)。

| 子技能 | 用途 | 关键参数/概念 |
|--------|------|-------------|
| [**activation-recompute**](skills/perf-techniques/activation-recompute/SKILL.md) | 选择性或全量 activation recompute，减少显存 | `recompute_method`, `recompute_granularity` |
| [**cpu-offloading**](skills/perf-techniques/cpu-offloading/SKILL.md) | CPU 卸载（activation、optimizer state） | `cpu_offloading`, `HybridDeviceOptimizer`, `optimizer_offload_fraction` |
| [**cuda-graphs**](skills/perf-techniques/cuda-graphs/SKILL.md) | CUDA graph 捕获，减少 kernel launch overhead | local full-iteration graphs, TE scoped graphs |
| [**expert-parallel-overlap**](skills/perf-techniques/expert-parallel-overlap/SKILL.md) | MoE 专家并行通信 overlap | `overlap_moe_expert_parallel_comm`, `delay_wgrad_compute`, DeepEP/HybridEP |
| [**hybrid-context-parallel**](skills/perf-techniques/hybrid-context-parallel/SKILL.md) | 层级 context parallelism（A2A + P2P） | `hierarchical_context_parallel_sizes` |
| [**megatron-fsdp**](skills/perf-techniques/megatron-fsdp/SKILL.md) | Megatron FSDP 配置与验证 | FSDP sharding 策略 |
| [**memory-tuning**](skills/perf-techniques/memory-tuning/SKILL.md) | 峰值显存优化，OOM 修复 | expandable segments，并行策略调整，recompute，CPU offload |
| [**moe-comm-overlap**](skills/perf-techniques/moe-comm-overlap/SKILL.md) | MoE dispatch overlap，flex dispatcher | `overlap_moe_expert_parallel_comm` |
| [**moe-dispatcher-selection**](skills/perf-techniques/moe-dispatcher-selection/SKILL.md) | 选择合适的 MoE token dispatcher | `alltoall` / DeepEP / HybridEP，按 EP 度和硬件选择 |
| [**moe-hardware-configs**](skills/perf-techniques/moe-hardware-configs/SKILL.md) | 各硬件平台（H100/GB200）MoE 训练 playbook | 吞吐量参考，并行配置模式 |
| [**moe-long-context**](skills/perf-techniques/moe-long-context/SKILL.md) | 长上下文 MoE 训练指南 | CP sizing，selective recompute，DSV3/Qwen3 实践 |
| [**moe-optimization-workflow**](skills/perf-techniques/moe-optimization-workflow/SKILL.md) | 系统性 MoE 优化工作流 | Three Walls 框架（内存墙、计算墙、通信墙） |
| [**moe-vlm-training**](skills/perf-techniques/moe-vlm-training/SKILL.md) | MoE VLM 训练（Qwen3-VL 等） | FSDP vs 3D-parallel 选择，Qwen3-VL/Next 经验 |
| [**parallelism-strategies**](skills/perf-techniques/parallelism-strategies/SKILL.md) | TP/PP/EP/CP/DP 并行策略选型与组合 | sizing 规则，硬件拓扑映射 |
| [**sequence-packing**](skills/perf-techniques/sequence-packing/SKILL.md) | Packed sequence 与长上下文训练 | LLM 离线 packing vs VLM in-batch packing，CP 约束 |
| [**tp-dp-comm-overlap**](skills/perf-techniques/tp-dp-comm-overlap/SKILL.md) | TP/DP/PP 通信 overlap 配置 | `overlap_grad_reduce`, `overlap_param_gather` |

---

## 推荐阅读顺序

### 新成员入门
1. [`developer-guide`](skills/developer-guide/SKILL.md) — 搭建开发环境
2. [`code-style`](skills/code-style/SKILL.md) — 了解代码规范
3. [`mlm-bridge-training`](skills/mlm-bridge-training/SKILL.md) — 跑通第一个训练

### 接入新模型
1. [`adding-model-support`](skills/adding-model-support/SKILL.md) — 完整接入流程
2. [`parity-testing`](skills/parity-testing/SKILL.md) — 验证权重正确性
3. [`recipe-recommender`](skills/recipe-recommender/SKILL.md) — 配置合适的训练 recipe

### 优化训练性能
1. [`perf-techniques/moe-optimization-workflow`](skills/perf-techniques/moe-optimization-workflow/SKILL.md) — 系统性优化入口
2. [`perf-techniques/parallelism-strategies`](skills/perf-techniques/parallelism-strategies/SKILL.md) — 并行策略选型
3. [`perf-techniques/memory-tuning`](skills/perf-techniques/memory-tuning/SKILL.md) — OOM 问题排查
4. [`perf-techniques/moe-dispatcher-selection`](skills/perf-techniques/moe-dispatcher-selection/SKILL.md) — MoE dispatcher 选择

### 生产部署
1. [`multi-node-slurm`](skills/multi-node-slurm/SKILL.md) — 多节点 Slurm 脚本
2. [`resiliency`](skills/resiliency/SKILL.md) — 容错配置
3. [`test-system`](skills/test-system/SKILL.md) — CI 测试管理
