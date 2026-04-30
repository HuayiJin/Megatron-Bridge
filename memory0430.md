# memory0430.md — Qwen3.5-122B-A10B 24-node SFT 全流程问题与修复记录

> **覆盖范围**：2026-04-30 一整天围绕 24-node × 8 GPU = 192 H800 跑
> Qwen3.5-VL-122B-A10B 全参 SFT (SEQ=98304, CP=1) 的问题诊断、脚本修改、
> 数据清洗、显存优化、NaN 调试。
>
> **关联文档**：
> - `memory.md`：全部 Pitfall #1~#28 + 标准配置表
> - `memory_cp_qwen35vl.md`：CP 严格分析（本文 24-node 走 CP=1，未触发）
> - `memory_legacy.md`：历史 bare-metal 决策

---

## 0. 一句话总结

| 阶段 | 事件 | 修复 |
|------|------|------|
| 1 | 数据 jsonl jinja TemplateError "System message must be at the beginning" | 写 `normalize_qwenvl_conversation.py` 重整 conversation |
| 2 | 24-node OOM @ PP rank 0 (ranks 0..15)，vision encoder 未冻结 | freeze vit + aligner，recompute 改 uniform/1 |
| 3 | NCCL 默认超时太紧 | 全部改到 20min（init + steady + watchdog） |
| 4 | iter 2 在 last PP stage (ranks 188-191) 出现 NaN | 加 loss/forward 诊断日志（本次任务） |

---

## 1. 数据清洗（demo.json / total.jsonl → *.fixed.jsonl）

### 1.1 报错栈
```
qwen2_5_collate_fn → _apply_chat_template_for_example
  → processor.apply_chat_template
    → jinja2.exceptions.TemplateError:
        System message must be at the beginning.
```

### 1.2 数据格式真实结构（examined demo.json）
- 顶层是**单个 JSON dict**（不是 jsonl），keys = `messages, images, tools`
- `messages[i].content` 是**裸字符串**，不是 Qwen-VL 期望的 `[{"type":"text","text":...}]` 列表
- 出现非标准 role：`tool_call`、`tool`
- 出现 `<image>` 和 `<ref_image>...</ref_image>` 占位符
- `images` 是顶层 flat list (20 张图)，靠**位置顺序**对应占位符
- `tools` 是 OpenAI function-calling schema

### 1.3 5 个不兼容点（全部需处理）
| # | 问题 | 修复策略 |
|---|------|---------|
| A | 多条 system，且不在 [0] | 只保留第一条，强制移到 [0] |
| B | role=tool_call 不合法 | 折叠到上一条 assistant 的 `tool_calls` 字段 (OpenAI/Qwen-Agent schema) |
| C | content 是裸字符串 | 升级为 list of `{"type":"text","text":...}` 部分 |
| D | `<image>` 占位与全局 images 池靠位置对应 | 按占位顺序消费 image 池，emit `{"type":"image","image":path}` |
| E | 顶层 tools | 原样透传（bridge collate.py:79-81 已支持） |

### 1.4 脚本

**位置**：`/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/demo_data/normalize_qwenvl_conversation.py`

**特性**：
- 纯 stdlib，无外部依赖
- 自动判别输入是单 JSON / JSON 数组 / JSONL
- 同时匹配 `<ref_image>...</ref_image>` 和裸 `<image>` 占位
- tool_call → assistant.tool_calls 折叠（OpenAI schema）；上一条非 assistant 时合成空 assistant 承载
- 相邻同 role 自动合并（可关）
- 启发式校验：role 集合、首条不能是 assistant/tool、content 非空、image 池足够
- 可选 `--validate-with-processor PATH` 用真实 transformers 跑 jinja dry-run
- 输出 `*.fixed.jsonl` (好) + `*.bad.jsonl` (坏，附 _reason/_stage)

**统计输出格式**：
```
multiple_system_collapsed / system_moved_to_front /
tool_call_merged_to_assistant / tool_call_with_synthetic_assistant /
text_split_for_image_tokens / consecutive_same_role_merged /
unknown_role_dropped[role] / image_pool_underflow /
image_pool_overflow_unused / jinja_validated_ok / jinja_validated_fail
```

### 1.5 实际跑的统计
**demo.json (1 条)**：1/1 通过
- multiple_system_collapsed: 1
- text_split_for_image_tokens: 20
- tool_call_merged_to_assistant: 2

**total.jsonl (102 条)**：102/102 通过，0 bad
- multiple_system_collapsed: 44
- **system_moved_to_front: 10** ← 这就是 jinja 报错的直接根因
- text_split_for_image_tokens: 1135
- tool_call_merged_to_assistant: 110
- tool_call_with_synthetic_assistant: 55

**输出路径**：`/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/demo_data/total.fixed.jsonl`

### 1.6 用法示例
```bash
# 干跑只看统计
python3 normalize_qwenvl_conversation.py --input total.jsonl --output total.fixed.jsonl --dry-run

# 实际写出 + jinja 真校验
python3 normalize_qwenvl_conversation.py \
  --input  /mnt/tidal-alsh01/.../total.jsonl \
  --output /mnt/.../total.fixed.jsonl \
  --validate-with-processor /path/to/Qwen3.5-VL-snapshot
```

---

## 2. 24-node OOM 诊断 (PP rank 0 vision encoder)

### 2.1 关键事实链
- **拓扑**：TP=2, PP=12, CP=1, EP=4, DP=8 → world=192
- **OOM rank**：8/9/10/11 (节点 1)，**全部属于 PP rank 0**（global ranks 0..15）
- **栈顶**：`vision_model.py:353 → transformer_block.py:318 → bias_dropout_add line 46  x = x + bias`
- **77.18 GB allocated / 80 GB**
- vision_model **只挂在 PP rank 0**（model.py:170 `if pre_process and add_encoder`）
- vision config 写死 `pipeline_model_parallel_size=1` (model.py:184)
- vision config 第 86-88 行 `num_moe_experts=None` → EP 切不动它
- `recompute_*` 字段会从 LLM config 复制到 vision config（transformer_config.py:74-76）
  - 但 `recompute_method=block + num_layers=4` 在 vision (~27-32 层) 上**只 recompute 前 4 层**，其余 ~28 层 forward activation 全留 → 显存爆点

### 2.2 ms-swift 跑通的对照（同硬件、同数据）
ms-swift 命令：
```
megatron sft --tensor_model_parallel_size 2 --pipeline_model_parallel_size 16
  --expert_model_parallel_size 8 --context_parallel_size 1
  --freeze_llm false --freeze_vit true --freeze_aligner true
  --packing true --recompute_granularity full --recompute_method uniform
  --recompute_num_layers 1 --max_length 100000 --micro_batch_size 1
  --global_batch_size 32 --torch_dtype bfloat16 ...
```

| 维度 | ms-swift（跑通） | Bridge（OOM） | 影响排序 |
|---|---|---|---|
| **freeze_vit** | **true** | false | ★★★★★ 决定性 |
| **freeze_aligner** | **true** | false | ★★★ |
| recompute_method | **uniform** | block | ★★ |
| recompute_num_layers | **1** | 4 | ★★ |
| PP | 16 | 12 | 不直接相关 |
| EP | 8 | 4 | 间接 |
| packing | true | **false** | 无法对齐（GDN 不支持 THD，Pitfall #7） |
| dtype | bfloat16（CLI 写死） | bf16（recipe 默认） | **不需要写死** |

### 2.3 `--torch_dtype bfloat16` 是否必要写死？
**不必要**。
- Bridge 的 dtype 由 recipe 自动设为 bf16
- ms-swift 的 `--torch_dtype bfloat16` 是它自己 CLI 习惯，覆盖 HF model 的 `torch_dtype` 字段
- Bridge 走 mcore checkpoint，dtype 由 model config 决定
- OOM 报错 77.18 GB 的数值也吻合 bf16，证明已经是 bf16

### 2.4 修复方案（已落地）
**唯一改动文件**：`scripts/run_sft_qwen35_122b_24node_hade.sh`

#### 改动 1-2：替换 recompute 段
```bash
# 原（block/4）：
model.recompute_method=block
model.recompute_num_layers="$PP_STAGE_LAYERS"   # = 4

# 改为（uniform/1，对齐 ms-swift）：
model.recompute_method=uniform
model.recompute_num_layers=1
```

#### 改动 3-4：新增 freeze
```bash
model.freeze_vision_model=True
model.freeze_vision_projection=True
```
对照表：
- ms-swift `--freeze_vit true` → Bridge `model.freeze_vision_model=True`
- ms-swift `--freeze_aligner true` → Bridge `model.freeze_vision_projection=True`
- ms-swift `--freeze_llm false` → Bridge 默认（recipe 已是 False）

#### 文件头注释、Header banner 同步更新
- 把 freeze_vit 列为 #1 显存策略
- 注释 `PP_STAGE_LAYERS` 现已仅供展示
- Banner 加 `Vision: freeze_vision_model=True freeze_vision_projection=True` 行

### 2.5 不动项
- TP/PP/CP/EP = 2/12/1/4 — freeze_vit 是大杠杆，不必同时引入 PP=16 (48/16=3 破坏 GDN 4 层周期对齐) 风险
- `optimizer_cpu_offload=True` — LLM Adam state 仍需要
- `cross_entropy_fusion_impl=te` — last stage CE 防御
- `mtp_num_layers=0` — Pitfall #19
- `dataset.pack_sequences_in_batch=False` — Pitfall #7 GDN 不支持 THD

### 2.6 验证步骤
1. PP rank 0 (ranks 0..15) 不再 OOM
2. nvidia-smi 看 PP rank 0 显存比之前下降 10+ GB
3. iter 0 / iter 1 跑出 loss
4. 若仍 OOM：`EP=4 → EP=8`（一行 override，DP=8 ≥ EP=8 仍满足）

---

## 3. NCCL 超时调到 20 min

### 3.1 字段背景
- Bridge 的字段是 `cfg.dist.distributed_timeout_minutes`（init_process_group 超时）
- 训练中通信超时是 `cfg.dist.distributed_timeout_seconds_after_init`（train.py:350-352 用 `update_pg_timeout()` 替换 init 时的超时）
- 字段定义在 mcore `common_config.py:71` `DistributedInitConfig`
  - `distributed_timeout_minutes: int = 10`（默认 10 分钟）
  - `distributed_timeout_seconds_after_init: int | None = None`
- NCCL 2.x 已删除 `NCCL_TIMEOUT` 环境变量，改用 PyTorch per-pg 超时

### 3.2 实际改动（脚本）

#### env 段新增（line 205-207 附近）：
```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
```

#### OVERRIDES 段新增（line 257-258 附近）：
```bash
dist.distributed_timeout_minutes="${DIST_TIMEOUT_MIN:-20}"
dist.distributed_timeout_seconds_after_init="${DIST_TIMEOUT_SEC:-1200}"
```

#### Header banner 加：
```
NCCL timeout : init=20min  steady=1200s  watchdog=1200s
```

### 3.3 时间线
```
t=0       init_process_group()         ← distributed_timeout_minutes (20min) 保护
t=A       train.py:350 触发
          update_pg_timeout(1200s)     ← steady 超时接管
t > A     每次 collective              ← 1200s 保护
          torch watchdog               ← TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200 监控
```
**关键约束**：watchdog 必须 ≥ pg 超时，否则 watchdog 会 kill 还在合法跑的 collective。

---

## 4. NaN 问题（iter 2 last PP stage）

### 4.1 报错
```
RuntimeError: Rank 189, node lshb-reservedpool-ftji-3, device 5,
  iteration 2: Unexpected result nan
  (message='found NaN in local forward loss calculation')
```
- **位置**：`losses.py:73 → rerun_state_machine.validate_result`
- **触发函数**：`masked_next_token_loss`，loss = sum(losses * loss_mask)
- **NaN 触发 ranks**：188, 189, 190, 191（节点 23）
- **iter**：2

### 4.2 拓扑分解
默认 mcore rank order = `tp-cp-ep-dp-pp`：
```
rank 184: tp=0 ep=0 dp=1 pp=11
rank 185: tp=1 ep=0 dp=1 pp=11
rank 186: tp=0 ep=1 dp=1 pp=11
rank 187: tp=1 ep=1 dp=1 pp=11
rank 188: tp=0 ep=2 dp=1 pp=11   ← NaN
rank 189: tp=1 ep=2 dp=1 pp=11   ← NaN
rank 190: tp=0 ep=3 dp=1 pp=11   ← NaN
rank 191: tp=1 ep=3 dp=1 pp=11   ← NaN
```
**线索**：184-187 (ep ∈ {0,1}) 没 NaN，188-191 (ep ∈ {2,3}) 全 NaN。同 DP rank，差别只在 EP。
→ **强烈暗示 NaN 与 EP 切分的某些 expert 相关**，不是数据本身。

### 4.3 NaN 可能来源
`loss = sum(losses * loss_mask)`：
1. `losses` 本身 NaN（model forward 出 NaN）
2. `losses` 有 Inf 且对应位置 mask=0 → `0 * inf = NaN` 经典坑
3. `loss_mask` 全 0 + losses 含特殊值
4. `output_tensor` 是 tuple（LLaVA 风格）时 mask 被覆盖且对齐错

### 4.4 时序
- iter 1 没有任何 loss 输出（rank23 是 last-stage 的 log，mcore 的 iteration log 通常只在 rank 0 打）
- iter 2 forward 出 NaN
- **可能**：iter 1 的 backward 让 weights 变 NaN/inf；或 iter 2 数据/EP 路由触发新 case

### 4.5 collective timeout 连带
其他节点 (e.g. node 4, ranks 32-39) 后续也报 `[E430 ProcessGroupNCCL] Received a dump signal due to a collective timeout from rank 63`，但**它们没报 NaN**——是被 rank23 节点 NaN 退出连累等不到 collective。

### 4.6 已加的诊断日志（本任务核心改动）

#### 改动文件 1：`src/megatron/bridge/training/losses.py`

**新增 imports + 模块级常量**：
```python
import logging
import os

_LOSS_DIAG_FIRST_N: int = int(os.environ.get("MBRIDGE_LOSS_DIAG_FIRST_N", "4"))
_LOSS_DIAG_VERBOSE: bool = os.environ.get("MBRIDGE_LOSS_DIAG_VERBOSE", "0") == "1"
_LOSS_DIAG_CALL_COUNT: int = 0

def _rank_tag() -> str: ...
def _tensor_fingerprint(name, t) -> str: ...   # NaN-safe min/max/n_nan/n_inf
```

**在 `masked_next_token_loss` 里**：
- 区分 `output_was_tuple`，记录 model 自带的 loss_mask 是否覆盖了绑定时的
- 在 `validate_result` 之前打印：
  - 单行 fingerprint：`loss / output_was_tuple / loss_mask_sum / loss_mask_nonzero`
  - 异常时 + verbose 时打全量：`losses` 和 `loss_mask` 的详细 fingerprint
  - **关键**：检测 `nonfinite & (mask>0)` 的 overlap → 区分"NaN 在 mask=0 区域被静默吞掉"vs"NaN 真的污染了 sum"
- 用 `print(..., flush=True)` 而非 logger，绕过 dataloader/scheduler 缓冲

**触发条件**：
- 默认前 4 次调用每 rank 必打
- `MBRIDGE_LOSS_DIAG_VERBOSE=1` → 每次都打
- 任何 NaN/Inf 检测 → 强制全量打

#### 改动文件 2：`src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py`

**在 `forward_step` 的 `output_tensor = model(**forward_args)` 之后**加：
- 仅在 `is_last`（last PP stage）触发
- 用函数属性 `forward_step._diag_call_count` 计数
- 打印 model 输出 tensor 的 fingerprint（shape/dtype/n_nan/n_inf/n_finite/min/max/mean）
- 异常时强制打；前 4 次必打；`MBRIDGE_LOSS_DIAG_VERBOSE=1` 时每次打
- 用同一组环境变量控制（与 losses.py 一致）

### 4.7 通过日志能区分的失败模式
| fwd-diag | loss-diag | 含义 |
|----------|-----------|------|
| n_nan>0 | losses n_nan>0 | model forward 已经出 NaN（vision/MoE/CE 上游 bug） |
| n_inf>0 | losses n_inf>0, mask 在该位置=0 | 0*inf 经典坑（CE 在 padding 区出 inf） |
| OK | losses OK 但 loss=NaN | 罕见，mask 全 0 之类边界 |
| OK | output_was_tuple=True 且 mask 替换前后形状不一致 | LLaVA-style step 对齐错 |

### 4.8 用法
```bash
# 默认前 4 次每 rank 打 fingerprint，异常时全量打
bash scripts/run_sft_qwen35_122b_24node_hade.sh

# 全量打（每次都打，调试时用）
MBRIDGE_LOSS_DIAG_VERBOSE=1 bash scripts/run_sft_...

# 改前 N 次窗口
MBRIDGE_LOSS_DIAG_FIRST_N=10 bash scripts/run_sft_...

# grep 时
grep -E "loss-diag.*ABNORMAL|fwd-diag.*ABNORMAL" *.log
grep -E "loss-diag rank188|loss-diag rank189|loss-diag rank190|loss-diag rank191" *.log
```

---

## 5. 跑批/日志注意事项（本次踩到的）

- 日志在 11:27 才 ready（用户提示）— 部分节点日志会延迟/缺失，logs4 目录里 24 节点只有 12 个 log 文件
- last PP stage 的日志（rank8..rank23 文件）才能看到 loss-diag / fwd-diag 输出
- `mcore` 的 iteration log（lm loss / lr / grad_norm）只在 rank 0 打 — 排查 last stage 问题需要 rank0 + last-stage 文件交叉看
- `print(..., flush=True)` 不走 logger，所以 logger 配置不影响诊断输出

---

## 6. 关键修改文件清单（本次会话累计）

| 文件 | 类型 | 作用 |
|------|------|------|
| `meg-run/demo_data/normalize_qwenvl_conversation.py` | 新建 | 数据 jsonl 清洗，仓外不入 git |
| `scripts/run_sft_qwen35_122b_24node_hade.sh` | 修改 | freeze vit + uniform/1 recompute + NCCL 20min 超时 |
| `src/megatron/bridge/training/losses.py` | 修改 | 加 NaN 诊断日志（loss-diag） |
| `src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py` | 修改 | 加 model 输出诊断日志（fwd-diag） |
| `src/megatron/bridge/training/train.py` | 修改 | 加 expert weight one-shot 诊断（expert-fp） |
| `memory0430.md` | 新建（本文件） | 全程记录 |

### 6.1 expert-fp 诊断（train.py）使用方法

**插入位置**：`train()` 函数内、`Starting training loop` 打印之后、while 循环之前。
保证 ckpt load 已完成、第一个 train_step 还未发生时打一次。

**输出格式**（每 rank 每个 expert/shared_expert 参数一行）：
```
[expert-fp] rank{R} tp={T} pp={P} cp={C} dp={D} ep={E} chunk={K}
  ok|ABNORMAL kind=expert|param name={qualified_name}
  shape=(...) dtype=torch.bfloat16
  n_nan=... n_inf=... n_finite=...
  min=... max=... mean=... abs_max=...
```
末尾会有 `[expert-fp] rank{R} done (diag_all=False)` 表示完成。

**默认行为**：开启，仅打 routed/shared experts（`.experts.` 或 `.shared_experts.`）。
**全量打**：`MBRIDGE_EXPERT_WEIGHT_DIAG_ALL=1`（每个 param 一行，量大）。
**关闭**：`MBRIDGE_EXPERT_WEIGHT_DIAG=0`。

**诊断 NaN 的关键 grep**（与 4.6 节配套）：
```bash
LOG_DIR=/mnt/.../logs5

# (1) 关键判定：是否 ep ∈ {2,3} 的 expert weight 一开始就 ABNORMAL?
grep "expert-fp.*ABNORMAL" $LOG_DIR/*.log | head -50

# (2) ep=0/1 vs ep=2/3 expert weight 的 abs_max / mean 是否分布一致?
grep "expert-fp.*ep=2" $LOG_DIR/*.log | head -20
grep "expert-fp.*ep=3" $LOG_DIR/*.log | head -20
grep "expert-fp.*ep=0" $LOG_DIR/*.log | head -20

# (3) NaN 4 个 rank 的 expert 列表
grep "expert-fp rank188" $LOG_DIR/*rank{8..23}*.log
```

**判定矩阵**：
| expert-fp ep=2/3 | 结论 | 下一步 |
|---|---|---|
| 任何 ABNORMAL（n_nan>0 或 n_inf>0） | ckpt load / conversion bug | 停训查 `qwen35_vl_bridge.py` 的 expert mapping，特别是 `linear_fc1.weight*` 切片路径 |
| 全 ok 但 abs_max / mean 与 ep=0/1 显著不同（>10×） | ckpt 切片不均，可能 routing imbalance 早期触发数值漂移 | 看 router top-k 选择，跑 EP=8 对照 |
| 全 ok 且分布一致 | ckpt 加载无问题，问题在 forward/backward | 走 fwd-diag/loss-diag 分流 |

---

---

## 7. Follow-up（未做、留给下一轮）

1. **NaN 真因**：依赖下次跑的 expert-fp / fwd-diag / loss-diag 三件套定位
   - **第一道闸**：iter 0 之前的 `expert-fp` —— 排除 ckpt load bug
   - 优先怀疑：MoE 路由/grouped_gemm/某 expert 在 ep ∈ {2,3} 上权重异常
   - 次要怀疑：CE TE chunked 在 padding 区出 inf
   - 验证手段：
     1. `grep "expert-fp.*ABNORMAL"` —— 直接显示 ckpt 已经异常
     2. `grep "fwd-diag.*ABNORMAL"` —— 找 model forward 第一次出 NaN 的 call#
     3. `grep "loss-diag.*ABNORMAL"` —— 区分 0×inf vs 真 NaN

2. **PP=16 + GDN 周期对齐**：ms-swift PP=16 跑通暗示能跑，但 48/16=3 破坏 4 层 GDN 周期。需研究 ms-swift 的 layer split 策略。

3. **EP=4 → EP=8**：如下次还 OOM 的备选；DP=8 仍 ≥ EP=8。

4. **CP=2 + vision_dp_when_cp**：终极方案，但要先过 `memory_cp_qwen35vl.md` 的 4 条 CP 硬约束。

---

## 8. 2026-04-30 第二轮诊断结论（基于 logs5）

### 8.1 三件套日志的关键证据
- **`expert-fp` 全部 ok**（grep ABNORMAL 0 命中）→ ckpt 加载没问题
- **iter 1 `fwd-diag` 全部 ok**：rank188-191 (ep ∈ {2,3}) 的 logits max=26.7，
  rank184-187 (ep ∈ {0,1}) max=23.8 — 数值健康，但 ep=2/3 已经稍高 ~12%
- **iter 2 `fwd-diag` ABNORMAL**：rank188-191 全部 NaN，n_nan = loss_mask_nonzero
  → NaN 与监督位置 100% 重合，是真 NaN，不是 0×inf
- **rank0 没产 iteration log**：被 last-stage NaN 退出连累，没 grad_norm 数据

### 8.2 锁定根因
**iter 1 backward 时 ep ∈ {2,3} 的某些 expert 收到了过大 grad，Adam step 后
weight 变 NaN，iter 2 forward 立即崩**。

证据链：
1. ckpt 没问题 (expert-fp ok)
2. iter 1 forward 没问题
3. ep=0/1 在 iter 2 没事，ep=2/3 在 iter 2 崩 → 同样数据，差别仅在 EP 切分
4. **MoE 配置 `moe_z_loss_coeff=None`**（recipe 默认）→ router logits 无界
5. `topk=8`（每 token 路由 8 个 expert）+ `aux_loss=0.001`（极弱）
   → router 容易把热点全压到几个 expert 上

### 8.3 数据分析（不是根因，但是关键放大器）
脚本：`meg-run/demo_data/analyze_sample_pressure.py` (新建，**已含 visual token**)

**第一次分析（仅文本）漏算了 visual tokens**，统计严重低估。
Qwen3.5-VL 公式：`vis_tokens = ceil(H/16) × ceil(W/16) / spatial_merge_size² ≈ H×W/1024`，
processor smart-resize cap 到 max_pixels (default ≈ 1280 tokens/img)。

**重跑（含 PIL probe 真实图片尺寸，1135 image refs / 303 unique）**：

| 指标 | 旧（仅文本） | 新（含 visual） | 倍数 |
|---|---|---|---|
| est_tokens p50 | 535 | **17,267** | 32× |
| est_tokens max | 10,688 | **57,722** (idx=99) | 5.4× |
| visual_tokens p50 | 0 | **15,360** | — |
| visual_tokens max | 0 | **51,200** (idx=99: 40 张图×1280) | — |

**iter 1 (GBS=32, idx 0..31) vs iter 2 总压力**：
- iter 1: pressure sum=**868k**, 5 条 50k+ tokens 的怪物 (idx 2/4/5/22/15, 都是 37-40 张图)
- iter 2: pressure sum=**117k** (iter 1 是 7.4×)
→ iter 1 backward 时 router 被 visual-token 砸爆 → ep=2/3 expert grad 失衡 → 崩

### 8.3.1 为什么超量样本"没被截断"？
- `dataset.drop_overlength_samples=False`（脚本默认）
- `qwen3_vl_step.py:391` `force_to_pad_to_seq_len=True` (PP=12 > 1) → **pad** 到 SEQ=98304
- 57k token 样本 < 98304，不会触发"超长丢弃"，**也不会被截断**
- 真问题：**batch 内不均** — 大样本激活几乎全部 expert，小样本被淹没

### 8.4 新数据集 total.v2.jsonl（已生成）

阈值：`max_est_tokens=30000, max_images=25, max_visual_tokens=28000`
- 命令：
  ```bash
  python3 filter_high_pressure.py --input total.fixed.jsonl \
      --output total.v2.jsonl --dropped-out total.v2.jsonl.dropped.jsonl \
      --max-est-tokens 30000 --max-images 25 --max-visual-tokens 28000 \
      --probe-images
  ```
- **保留 85/102 (16.7% 丢)**，丢的 17 条都是 30+ 张图的极端样本（idx 2/4/5/15/22 等怪物全去掉）
- 仍保留 12-20 张图的样本，VLM 训练价值不丢
- 新分布：max est_tok=29458（差不多减半），iter 1/2 总压力比从 7.4× → **1.5×**

| 维度 | total.fixed.jsonl | total.v2.jsonl |
|---|---|---|
| 样本数 | 102 | 85 |
| est_tok max | 57722 | 29458 |
| iter 1 总压力 | 868k | 427k |
| iter 1/iter 2 比 | 7.4× | 1.5× |

### 8.5 本轮落地的修复

| 文件 | 改动 | 目的 |
|---|---|---|
| `scripts/run_sft_qwen35_122b_24node_hade.sh` | `model.moe_z_loss_coeff=1e-3` | bound router logit 范数，**主修复** |
| 同上 | 加 LR/LR_WARMUP_ITERS/MOE_AUX_LOSS_COEFF env knob | 备选实验，默认沿用 recipe |
| `meg-run/demo_data/analyze_sample_pressure.py` | 新建 + 升级 | 含 visual token 的离线压力分析 |
| `meg-run/demo_data/filter_high_pressure.py` | 新建 + 升级 | visual-token-aware 高压样本过滤 |
| `meg-run/demo_data/total.v2.jsonl` | 新建 | 85/102 条样本（去掉 30+ 图怪物） |
| `meg-run/demo_data/total.v2.jsonl.dropped.jsonl` | 新建 | 17 条被丢样本 + 原因 + stats |

### 8.6 下次跑的优先矩阵（**已升级**）

新认知：visual tokens 是 router 的**主要压力源**，比 z-loss 缺失更"可见"。
推荐先用 **A+B 组合**（z-loss + v2 数据），同时打掉两个根因。

| 实验 | 命令 | 预期 |
|---|---|---|
| **A+B：z-loss + v2 数据**（推荐） | `TRAIN_DATA=/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/demo_data/total.v2.jsonl bash scripts/run_sft_qwen35_122b_24node_hade.sh` | iter 2 不再 NaN，可跑到 iter 5+ |
| A：仅 z-loss（原数据） | `bash scripts/run_sft_qwen35_122b_24node_hade.sh` | 验证 z-loss 能否独立扛住 50k token 样本 |
| C：A+B + 降 LR | `LR=1e-5 LR_WARMUP_ITERS=500 TRAIN_DATA=...total.v2.jsonl bash scripts/...` | 若 A+B 仍 NaN |
| D：A+B + 强 aux | `MOE_AUX_LOSS_COEFF=1e-2 TRAIN_DATA=...total.v2.jsonl bash scripts/...` | 若 A+B 仍 NaN |
| E：全开 | A+B+C+D | 兜底 |

如 A+B 跑通，**iter 2 fwd-diag 和 loss-diag 应该全 ok**，rank0 有 lm loss / grad_norm 输出。

### 8.7 数据集要不要处理？（**已修订**）
- 第一轮判断"NaN 根因不在数据"是基于错误的压力统计（漏算 visual tokens）
- **新认知**：iter 1 喂了 5 条 50k token 样本（每条 ~50000 visual tokens），是 router 失衡的**主要驱动**
- ep=0/1 不崩仅说明它们持有的 expert 群组运气好；实际所有 expert 都在被压
- **结论**：A+B 同时上才是正解 —
  - z-loss 防止后续训练数值漂移
  - v2 数据消除 iter 1 的极端冲击

---

---

## 9. 第四轮 / 第五轮诊断（logs5 → logs6 → logs7，MoE 通信链与 GroupedGEMM）

### 9.1 时间线总览

| 轮次 | 日志 | 触发改动 | 现象 | 结论 |
|---|---|---|---|---|
| 4 | logs5 | A+B 上线（z-loss + v2 数据） | 24 节点全部跑过 expert-fp，last PP stage call#1..#4 forward+loss 全 ok 无 NaN；mid PP stage 在 SeqNum=9 PIPELINE_MODEL_PARALLEL_GROUP 上 watchdog timeout 1240s | A+B 修好了 NaN；新瓶颈是 **MoE all-to-all NCCL group 惰性初始化与 PP recv_backward 抢跑** |
| 5a | logs6 | NCCL eager init + 60min timeout 上线 | 3 个 rank 崩在完全相同栈：`te_general_grouped_gemm → cublaslt_gemm.cu:543 cuBLAS Error: function failed to launch on the GPU` → 异步 IMA on EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP / EXPERT_MODEL_PARALLEL_GROUP | NCCL hang 解决；新故障下沉到 cuBLAS grouped GEMM kernel |
| 5b | logs7 | M1+M2+M3 三件套默认开（NVTE_CUBLAS_WORKSPACE_SIZE_BYTES=128MiB + CUBLAS_WORKSPACE_CONFIG=:4096:8 + NVTE_BIAS_GELU_NVFUSION=0） | rank48 仍崩在**完全相同的栈**；崩溃 rank 数从 3→1，但本质未除 | **workspace 与 fused gelu 假设全部证伪**；下一步唯一推荐 = E1 旁路 grouped GEMM |

### 9.2 logs5 锁定的真因（NCCL group 惰性初始化死锁）

- 拓扑：TP=2 PP=12 EP=4 CP=1 DP=8，GBS=32 MBS=1，每 DP rank 4 个 microbatch
- 观察到的 hang：节点 6/7（mid PP stage, ranks 48-63）在 `PIPELINE_MODEL_PARALLEL_GROUP` SeqNum=9 上等 1240s 后 watchdog kill
- stack：`recv_backward → _communicate → batch_isend_irecv → endCoalescing`
- last PP stage（节点 22/23）**没有任何 [E430**：它们没崩，是它们**还没轮到 send_backward**，因为它们正在 init 一个新的 nranks=8 NCCL group（commId 0x7e9a... / 0xa964... / 0xe379...）
- nranks=8 group = TP×EP=2×4 平面，是 MoE token-permute all-to-all 的子集，**第一次进入 backward 时才被惰性创建**
- 节点 14-23 (PP rank 7-11) 看到 `nranks=8` group，节点 0-13 没看到 → 不同 PP stage 在不同时刻调用 `ncclCommInitRankConfig`，跨 stage 不对称 → 死锁

**修复（已落地，logs6+ 已验证不再 hang）**：
- `NCCL_RUNTIME_CONNECT=0`：comm init 时 eager 建立所有 NCCL transport 连接，缩小惰性 init 竞争窗口
- `NCCL_NVLS_ENABLE=0`：去掉 NVLink-Sharp 的多阶段惰性分配
- `TORCH_NCCL_USE_COMM_NONBLOCKING=0`、`TORCH_NCCL_HIGH_PRIORITY=1`
- `dist.distributed_timeout_minutes=60` + `dist.distributed_timeout_seconds_after_init=3600` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`：从 20min/600s 拉到 60min/3600s

### 9.3 logs6 暴露的真因（cuBLAS grouped GEMM launch fail）

崩溃 rank 与拓扑：

| rank | 节点 | 拓扑 (tp/pp/cp/ep/dp) | 错误类型 |
|---|---|---|---|
| rank47 | node 5 | tp=1 pp=2 cp=0 ep=3 dp=7 | IMA on PG GUID 812 (EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP) |
| rank112 | node 14 | tp=0 pp=7 cp=0 ep=0 dp=0 | **Python-level RuntimeError**（最权威） |
| rank132 | node 16 | tp=0 pp=8 cp=0 ep=0 dp=2 | IMA on PG 834 (EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP) + PG 594 (EXPERT_MODEL_PARALLEL_GROUP) |

**统一的 Python 调用栈**（rank112 提供）：
```
src/megatron/bridge/training/train.py:906   forward_backward_func(...)
3rdparty/.../pipeline_parallel/schedules.py:2237   forward_step(...)
3rdparty/.../pipeline_parallel/schedules.py:437    forward_step_func(data_iterator, model)
src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py:456   output_tensor = model(**forward_args)
  ...
src/.../qwen_vl/modelling_qwen3_vl/transformer_block.py:592 _checkpointed_forward
  → tensor_parallel/random.py:642 checkpoint                ← Megatron activation checkpoint
3rdparty/.../transformer/transformer_layer.py:818  _forward_mlp(...)
3rdparty/.../transformer/moe/moe_layer.py:634 → 598 → 507 routed_experts_compute(...)
3rdparty/.../transformer/moe/experts.py:369  fc1_output, bias_parallel = apply_module(self.linear_fc1)(...)
3rdparty/.../extensions/transformer_engine.py:1905  out = super().forward(x, m_splits, is_first_microbatch=...)
[TE 已用 @no_torch_dynamo() 装饰，eval_frame.py:850 只是 disable trampoline，不是真在 trace]
transformer_engine/pytorch/module/grouped_linear.py:751  out = linear_fn(*args)
transformer_engine/pytorch/module/grouped_linear.py:158  general_grouped_gemm(...)
transformer_engine/pytorch/cpp_extensions/gemm.py:164  tex.te_general_grouped_gemm(...)
  ↓
RuntimeError: /workspace/TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:543
              in function cublas_gemm:
              cuBLAS Error: the function failed to launch on the GPU
随后 → CUDA error: an illegal memory access was encountered（异步 IMA）
```

**MoE 维度（来自 expert-fp 自带 tag）**：
- 每 rank 持有 64 个 routed expert（fc1 weight0..weight63，shape=(2048, 3072)）
- recipe：`moe_grouped_gemm=True`、`moe_token_dispatcher_type="alltoall"`、`moe_permute_fusion=True`、`moe_router_topk=8`
- 单 microbatch 路由 token 数 = 98304 × 8 / EP=4 = 196,608，平均每 expert ~3072 token
- 环境：torch 2.8.0a0+nv25.06、TE 2.4.0+3cd6870、cuBLAS/cuda 12.9、cudnn 9.10.2、NCCL 2.27.3、GPU 是 H800 80G（hostname/型号字符串里的 "L20Y" 是 H800 的别名，仍是 sm_90）

### 9.4 logs7 的关键负面证据（M1+M2+M3 默认开后）

logs7 一手 grep（已排除 launch-diag 横幅噪音）：

| 关键字 | 真实命中节点数 |
|---|---|
| cuBLAS Error / cublaslt_gemm / te_general_grouped_gemm | **1**（rank48 = node 6, host lshb-qs-vwik-22, tp=0 pp=3 ep=0） |
| illegal memory access | 2（rank48 主因 + 一个 IMA 二级） |
| RuntimeError | 1（rank48） |
| ChildFailedError / SIGABRT | 2（node 6 主退 + node 8 被连累退） |
| Watchdog caught | 0 |

崩溃栈与 logs6 完全相同：`te_general_grouped_gemm → cublaslt_gemm.cu:543 → cuBLAS Error → 异步 IMA`。

**结论**：
1. M1（NVTE_CUBLAS_WORKSPACE_SIZE_BYTES=128 MiB）→ **证伪**：workspace 翻 4 倍没救
2. M2（CUBLAS_WORKSPACE_CONFIG=:4096:8）→ **证伪**：global cuBLAS pool 不是瓶颈
3. M3（NVTE_BIAS_GELU_NVFUSION=0）→ **证伪**：fused gelu 不是诱因
4. 崩溃 rank 数从 3 → 1：mitigation 让一些原本边缘崩的 rank 撑过去了，但**最受压的那一个仍打不过 cuBLAS**，本质问题不是 workspace
5. 唯一未试且证据指向最直接的下一步：**E1 = `model.moe_grouped_gemm=False`**，完全绕开 TE GroupedLinear

### 9.5 已落地的脚本最终态（清理后）

清理掉 logs7 证伪的 M1/M2/M3 与从未启用的 E2 (`model.moe_permute_fusion=False`) / E3 (cuBLASLt 详细日志)。**保留**：

- 所有 NCCL 修复（eager init、60min timeout、flight recorder）—— logs5→logs6 已验证
- MoE z-loss=1e-3 与 v2 数据 —— logs5 已验证消除 NaN
- freeze vit + uniform/1 recompute —— 第二节已验证消除 OOM
- `TORCH_SHOW_CPP_STACKTRACES=1`（无害；仅异常时激活，logs6/logs7 富栈依赖此项）
- **唯一新开关 E1**：`MBRIDGE_DISABLE_MOE_GROUPED_GEMM=1` → 自动追加 `model.moe_grouped_gemm=False`

脚本从 749 行精简到 635 行。文档头里**显式标注被证伪的方向**，防止下一轮 agent 再次发明它们。

### 9.6 下一轮跑命令（推荐 E1）

```bash
LOG_DIR=/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/logs8 \
TRAIN_DATA=/mnt/tidal-alsh01/dataset/pai/hade/dd/meg-run/demo_data/total.v2.jsonl \
MBRIDGE_DISABLE_MOE_GROUPED_GEMM=1 \
bash scripts/run_sft_qwen35_122b_24node_hade.sh
```

**预期**：每 expert 走普通 GEMM，慢 ~1.3-1.5×，但 `te_general_grouped_gemm` 整条栈消失，cuBLAS Error 不再出现。能跑出 iter 1 的 lm loss / grad_norm 才算真正过关。

### 9.7 跑后判定矩阵

| logs8 grep 结果 | 含义 | 下一步 |
|---|---|---|
| 任意 rank 仍 `cuBLAS Error \| cublaslt_gemm` 命中 | 不可能；E1 已经把 grouped GEMM 关了，普通 GEMM 还崩说明 GPU/cuBLAS 本身坏 | 单卡复现，疑硬件/驱动 |
| 0 个 rank `cuBLAS Error`，但出现 NaN | router 在 grouped→ungrouped 切换后行为变化，z-loss 仍需调强 | 走 §8.6 的 C/D（降 LR 或加 aux_loss） |
| 0 个 rank `cuBLAS Error`，rank0 出现 `iteration 1/` 且无 NaN | **彻底通关** | 跑长 step 验证收敛 |
| 在 `_forward_mlp` 之外的某个新位置崩 | grouped GEMM 不是真因，是另一个 kernel 巧合先炸 | 看新栈，针对性修 |

### 9.8 给未来 agent 的一段话

- **不要再加 cuBLAS workspace** 类的 env：logs7 已证伪
- **不要假设 dynamo 在 trace TE GroupedLinear**：TE 已用 `@no_torch_dynamo()` 装饰，栈里的 `_dynamo/eval_frame.py:850 _fn` 是 `torch._dynamo.disable()` 的 trampoline，不是真在 trace
- **不要被 `EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP` watchdog 报错误导**：这是异步 IMA 在下一次 watchdog 轮询被发现的位置，**不是真正的 collective hang**。真因永远是 Python-level `RuntimeError`，看 `RuntimeError` 而不是 watchdog 行
- 24 节点跑代价巨大，每次跑前先列 hypothesis-evidence-mitigation 表，**单次跑要能区分多个假设**

---

_Last updated: 2026-04-30 第五轮（logs7）。
M1+M2+M3 全部证伪，下一步唯一推荐 E1=MBRIDGE_DISABLE_MOE_GROUPED_GEMM=1。
脚本已清理：749→635 行，删除 cuBLAS workspace / fused gelu / permute fusion / cuBLASLt log 四个证伪开关。_
