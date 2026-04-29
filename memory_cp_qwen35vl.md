# memory_cp_qwen35vl.md — Qwen3.5-VL Context Parallel (CP) 严格分析

> **范围：** Qwen3.5-VL 系列（dense + MoE，含 122B-A10B）在 Megatron-Bridge
> 上开启 `context_parallel_size > 1`（CP > 1）时的可行性、约束、修复历史。
>
> **结论一句话：** CP 能开，但必须同时满足 4 条硬性前提；当前 12 节点 32K
> 脚本（`scripts/run_sft_qwen35_122b_12node_hade2.sh`）已在 2026-04-29
> 全部满足。
>
> 本文是对 2026-04-29 那轮 fused-CE dynamo broadcast 报错的**根因审计与
> 永久结论**。日常运维参见 `memory.md`，历史决策参见 `memory_legacy.md`。

---

## 1. 总结（TL;DR）

| 项 | 结论 |
|----|------|
| **CP 能不能用** | ✅ **能用**，但有 4 条硬性前提 |
| **典型可用配置** | 96 GPU：TP=2, PP=6, CP=2, EP=4, DP=4, SEQ=32768 |
| **不能用的组合** | ① `--step_func vlm_step` + CP > 1；② PP=1 + CP > 1（double-slice 隐患）；③ `pack_sequences_in_batch=True` + CP > 1（GDN 不支持 THD） |
| **本次是真 bug 还是配置错** | **配置错叠加上游隐患**：步进函数选错（vlm_step）是真根因，fused-CE 在 dynamo 下崩溃只是表面症状 |
| **永久防御** | 已加：(a) `vlm_step.get_batch` CP > 1 时 `AssertionError`；(b) `training/config.py` 在 CP > 1 时自动关闭 native fused CE 并 warn |

---

## 2. 4 条硬性前提（不满足就出错）

| # | 前提 | 不满足时的后果 |
|---|------|----------------|
| 1 | `--step_func qwen3_vl_step`（**绝对**不能用 `vlm_step`） | last PP stage labels 全长 vs logits 切到 SEQ/CP，CE 在 fused 路径报 dynamo broadcast 错；unfused 路径**静默**算出错误 loss |
| 2 | `cross_entropy_loss_fusion=False` 或 `cross_entropy_fusion_impl="te"` | 即使前提 #1 成立，native 融合的 jit_fuser 路径在 CP shape mismatch 下 dynamo 崩溃。**已经被 `training/config.py` 自动 fixup**（CP > 1 时强制关 native 融合并 `warn_rank_0`） |
| 3 | `model.calculate_per_token_loss=True` 且 `ddp.average_in_collective=False` | mcore validator (`config.py:1224-1228`) 直接 assert fail；SFT 下若某 CP rank 全 mask 还会算出 NaN loss |
| 4 | `SEQ % (2 * CP) == 0` 且 `dataset.pack_sequences_in_batch=False` | mcore zigzag 切片要求 `2*CP` 整除（assert）；in-batch packing+CP 在 GDN/线性注意力上**不支持**（GDN 不支持 THD format） |

> 本次 12-node 脚本（`scripts/run_sft_qwen35_122b_12node_hade2.sh`）4 条**全部满足**。

---

## 3. CP 拓扑约束逐项验证（已通过）

按 `world_size = 96, TP=2, PP=6, CP=2, EP=4` 推算：

| 检查项 | 数值 | 结论 |
|--------|------|------|
| `world_size = TP × PP × CP × DP` | 96 = 2×6×2×4 → DP=4 | ✓ |
| `EP ≤ DP` | EP=4 ≤ DP=4 | ✓（Pitfall #17） |
| `SEQ % (2*CP) == 0` | 32768 % 4 = 0 | ✓ |
| `num_kv_heads / TP` | 2 / 2 = 1 | ✓ |
| **`num_kv_heads / CP`** | 2 / 2 = 1 | **不要求**——ring CP 不切 KV head |
| GDN: `linear_num_value_heads / TP / CP` | 64 / 2 / 2 = 16 | ✓ |
| GDN: `linear_num_key_heads / TP / CP` | 16 / 2 / 2 = 4 | ✓ |
| 标准 attention head: `num_attention_heads / TP` | 32 / 2 = 16 | ✓（ring CP 不切 head；Ulysses 模式才需要） |
| `cp_comm_type` | 默认 `"p2p"`（ring） | ✓，无需 `hierarchical_context_parallel_sizes` |
| 模型层数 / PP | 48 / 6 = 8 | ✓（GDN 4 层周期对齐 8） |

### 3.1 关键澄清：CP 与 KV head 的关系

很多人（包括我第一遍）以为 "CP > 1 要求 num_kv_heads 能被 CP 整除"——
**这是错的**。事实如下：

- **Ring CP（`cp_comm_type="p2p"`，默认）**：不切 head，沿序列维 P2P 交换 KV。
  对 KV head 数没有可整除约束。
- **Ulysses CP（`cp_comm_type="a2a"`）**：把 sequence A2A 重排到 head 维，
  要求 `num_attention_heads % CP == 0`（**注意是 Q heads，不是 KV heads**）。
- **Hybrid `a2a+p2p`**：需要 `hierarchical_context_parallel_sizes`，
  否则 CP 通信会被静默关掉（Pitfall：silent 高吞吐 + 错误训练）。

122B `num_kv_heads=2` + CP=2 在 ring 模式下完全 OK，不要被 "2 整除 4" 这种
表面式推理误导。

---

## 4. 真根因审计：`vlm_step` vs `qwen3_vl_step`

### 4.1 报错表象

```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node ...
Attempting to broadcast a dimension of length 65536 at -1!
Mismatching argument at index 1 had torch.Size([65536]);
but expected shape should be broadcastable to [32768]
```

栈顶在 `mcore/fusions/fused_cross_entropy.py` → `cross_entropy.py:59`：
```python
predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
```
- `logits_2d.shape = (32768, vocab)` → 已被 CP 切到 `SEQ/CP = 32768`
  （SEQ=65536, CP=2）
- `masked_target_1d.shape = (65536,)` → 仍是全长 SEQ

### 4.2 为什么会形状不一致

逐层看 PP=6 + CP=2 下 labels 的命运：

| 阶段 | labels 状态 |
|------|-------------|
| `vlm_step.get_batch` 出口 | **全长 [B, SEQ]**——`vlm_step` 内**不调用** `get_batch_on_this_cp_rank` |
| 进入 `model.forward`（PP rank 0） | 因 `is_last_pp_stage=False`，`get_batch_from_iterator` 根本没把 labels 加进 device key（`vlm_step.py:106`），**labels=None** |
| `model.py:557-564` 切片块 | `if labels is not None` → 跳过；只切了 `combined_embeddings` |
| 进入 `model.forward`（PP rank 5，last stage） | labels = **全长 [B, SEQ]** |
| 该 stage 的 model.forward | `pre_process=False`，**整个 line 424 的 if 块都不执行**，CP 切片代码不触发 |
| 进入 LM head + CE | logits `[B, SEQ/CP, V]`（被模型激活流切了）；labels `[B, SEQ]`（全长）→ **形状不一致** |

`vlm_step` 的设计假设是 "model.forward 在 pre_process 分支自己切 labels"，
但 Qwen3VLModel 的实现只在 PP rank 0 切，**而 labels 又只在 last PP stage
被加载**——两个集合不相交，labels 永远不会被切到。

### 4.3 fused CE 是替罪羊，不是真凶

- fused 路径：`@jit_fuser` 把 `logits_2d[arange_1d, masked_target_1d]`
  编译给 dynamo，dynamo 的 fake-tensor `meta_index_Tensor` 对 broadcast
  做严格 check，shape 不一致**立刻 abort**——所以你看到了一个清晰的报错。
- unfused 路径：同一行代码（`cross_entropy.py:59`）在 eager 模式下走标准
  PyTorch fancy indexing。65536 长 target 拿去索引 32768 长 logits 的
  第 0 维，会**触发越界 IndexError 或 silent broadcast**——后者更糟，
  loss 数值错但不报错。

也就是：**就算关掉 fused CE，bug 仍在，只是从 dynamo 崩溃变成静默错误。**

### 4.4 正确路径：`qwen3_vl_step`

`qwen3_vl_step.forward_step:411`：
```python
forward_args = get_batch_on_this_cp_rank(forward_args, cp_group=this_pg_collection.cp)
```

它在 **batch 入口**（不是 model 内部）把 `forward_args["labels"]` /
`forward_args["loss_mask"]` 沿 seq dim 按 CP zigzag 切到 `SEQ/CP`。这一步
对**每个 PP stage**都执行（包括 last stage），所以 last stage 拿到的
labels 是 `[B, SEQ/CP]`，与 logits 形状一致。

数据流自洽性证明：

| PP stage | labels 输入 | step_func 切 | model.forward 切 | 进 CE 时 labels |
|----------|------------|--------------|-----------------|----------------|
| rank 0（first，非 last）| None | noop（None 跳过） | noop（labels=None）| 不进 CE |
| rank 1..4（中间）| None | noop | 无 pre_process，整块跳 | 不进 CE |
| rank 5（last）| 全长 | **切到 SEQ/CP** | 无 pre_process，不再切 | **SEQ/CP** ✓ |

完全一致 ✓。

### 4.5 修复落点（2026-04-29）

1. **`scripts/run_sft_qwen35_122b_12node_hade2.sh`**：
   `--step_func vlm_step` → `--step_func qwen3_vl_step`。
2. **`src/megatron/bridge/training/vlm_step.py:get_batch`**：
   开头加 `if cp_size > 1: raise AssertionError(...)`，把人引到
   `qwen3_vl_step` / `llava_step`。这样别的 122B 脚本（4/6/8/16-node 系列
   仍用 `vlm_step`）只要保持 CP=1 就还安全；一旦想开 CP，会立刻 fail 而
   不是 silent 错误 loss。
3. **`src/megatron/bridge/training/config.py`**：CP > 1 时自动关闭
   `cross_entropy_loss_fusion`（仅当 `impl="native"`），`warn_rank_0`
   提示用户也可以改用 `'te'` impl 保留融合。这是**第二道防线**——即便
   将来又有人用了不切 labels 的 step_func，至少 dynamo 不会再被触发。
4. **`src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py`**：
   在 `cfg.model.cross_entropy_loss_fusion = True` 旁加注释，指向上述
   validator 的自动 fixup，让 CP=1 默认依然享受融合性能。

---

## 5. 已知边界与未触发的隐患

### 5.1 PP=1 + CP > 1 + `qwen3_vl_step` 会 double-slice

`Qwen3VLModel.forward` 在 `if self.pre_process and combined_embeddings is
not None and cp_size > 1 and packed_seq_params is None` 分支里，对
`labels` 又调用了一次 `split_data_cp_rank`（model.py:561-562）。

- PP > 1：PP rank 0 拿到的 labels 是 None（last stage 才装载），切片是 noop ✓
- **PP = 1**：PP rank 0 同时是 last stage，labels 已被 step_func 切到
  `[B, SEQ/CP]`，model 内再切一次变成 `[B, SEQ/(CP×CP)]` ❌ → CE 崩

**当前 12-node 脚本 PP=6，不踩此坑**。但任何 PP=1 + CP > 1 + Qwen3-VL
都会炸。修复方案（未实施）：
- model.py:561-564 改成幂等：先看 labels 当前长度，若已 = `SEQ/CP` 就 noop
- 或者在 model.forward 入口改成 "labels 这一行只在 model 自己负责切的
  step_func 下做"，需要新加一个 flag。

记入 follow-up，本次不修，避免影响其他用例。

### 5.2 MTP + CP

MTP（Multi-Token Prediction）在 last PP stage 多挂一个 transformer block
+ LM head。它对 CP 的兼容性**未在本次审计中验证**。

当前 12-node 脚本因为 last-stage memory 紧张已经设 `mtp_num_layers=0`
（Pitfall #19），顺带规避了这个未知。如果以后要重新打开 MTP + CP，**必
须**单独验证。

### 5.3 `vision_dp_when_cp`

`Qwen3VLModel` 有 `self.config.vision_dp_when_cp` 标志（model.py:442）。
打开时 vision encoder 在 CP rank 之间做 DP 切分，再通过
`AllGatherVisionEmbeddings` 还原；关闭时每个 CP rank 重复跑 vision encoder。

当前 122B 脚本默认 `False`（vision encoder 重复算，多吃一点显存换简单）。
122B 的 vision encoder 相对 LLM 很小，重复算的代价低，**保持默认**。
如果未来 vision encoder 占比变大，可以打开它换显存。

### 5.4 `pack_sequences_in_batch` + CP

GDN/线性注意力**不支持** THD format（Pitfall #7）。in-batch packing 内部
会切成 THD，与 CP 的 BSHD 切片路径冲突。**Qwen3.5-VL 必须保持
`pack_sequences_in_batch=False`**，无论 CP 是否打开。

### 5.5 `cp_comm_type` 选择建议

| 场景 | 推荐 `cp_comm_type` | 原因 |
|------|--------------------|----- |
| 单节点内 CP（CP ≤ 8）| `"a2a"` 或默认 `"p2p"` | 节点内 NVLink 带宽充足，A2A/Ring 都行 |
| 跨节点 CP（CP > 8） | `"a2a+p2p"` + `hierarchical_context_parallel_sizes=[8, CP/8]` | 节点内 A2A，节点间 P2P，避免跨节点 A2A 的带宽瓶颈 |
| 当前 12-node CP=2 | 默认 `"p2p"` | CP 在节点内，无需特殊配置 |

**警告**：`a2a+p2p` 必须搭配 `hierarchical_context_parallel_sizes`，
否则 CP 通信被静默关掉，每个 rank 只看自己的 chunk → 假高吞吐 + 假低 loss
+ 真坏训练。`training/config.py:1338-1345` 已加 assert 拦截。

---

## 6. 12-node 脚本最终安全配置（参考）

```bash
# scripts/run_sft_qwen35_122b_12node_hade2.sh 的关键 OVERRIDES
--step_func qwen3_vl_step             # 前提 #1
--dataset vlm-preloaded
model.tensor_model_parallel_size=2
model.pipeline_model_parallel_size=6
model.context_parallel_size=2
model.expert_model_parallel_size=4    # ≤ DP=4
model.seq_length=32768                # 32768 % (2*CP) = 0
model.calculate_per_token_loss=True   # 前提 #3
ddp.average_in_collective=False       # 前提 #3
model.cross_entropy_loss_fusion=False # 前提 #2（validator 也会自动关）
model.mtp_num_layers=0                # 规避 5.2 + Pitfall #19
dataset.pack_sequences_in_batch=False # 前提 #4 + GDN 限制
```

数据流（再走一遍以确认无歧义）：

```
qwen3_vl_step.forward_step
  → pack_or_pad_batch_sequences   # pad 到 32768，cp_multiple=2*CP=4
  → get_batch_on_this_cp_rank     # labels/loss_mask 切到 SEQ/CP=16384，所有 PP stage 都切
  → model.forward(input_ids 全长, labels=切片后 or None)
     ├ PP rank 0:  pre_process → split combined_embeddings 到 SEQ/CP；labels=None，不再切
     └ PP rank 5:  no pre_process；labels 已是 SEQ/CP；过 LM head + CE 形状一致
  → CE: logits [B, SEQ/CP, V]  vs  labels [B, SEQ/CP]    ✓
```

---

## 7. 与 `memory.md` 的关系

- `memory.md`：日常运维 + 全部 Pitfall #1~#24 + 标准配置表
- 本文（`memory_cp_qwen35vl.md`）：CP 这一专题的**严格分析**与永久结论
- `memory_legacy.md`：历史 bare-metal / 旧 Dockerfile 决策

`memory.md` 的 Pitfall #23 / #24 是事件级记录（什么报错、当时怎么改）；
本文是**原理级**结论，回答 "CP 到底能不能开、为什么、什么时候不能开"。

两者交叉引用：
- `memory.md` → "CP 严格分析见 `memory_cp_qwen35vl.md`"
- 本文 → "事件历史见 `memory.md` Pitfall #23, #24"

---

_Last updated: 2026-04-29 (CP 严格分析首次记录；4 条硬性前提；vlm_step
vs qwen3_vl_step 真根因审计；PP=1 double-slice follow-up；MTP+CP 未知；
12-node 脚本最终安全配置)_
