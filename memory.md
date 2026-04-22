# Qwen3.5-122B-A10B Megatron-Bridge CKPT 转换任务

## 任务目标
将 HuggingFace 格式的 Qwen3.5-122B-A10B 多模态 VLM checkpoint 转换为 Megatron 格式，用于后续 SFT 训练（含 MTP）。

---

## 路径

| 项目 | 路径 |
|---|---|
| HF 模型 | `/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B` |
| 代码库 | `/data/temp/Megatron-Bridge/` |
| 目标 Megatron ckpt | `${WORKSPACE}/models/Qwen/Qwen3.5-122B-A10B`（待定） |

---

## 模型架构（来自 config.json）

### 整体
- **HF 类名**: `Qwen3_5MoeForConditionalGeneration`（MoE VLM）
- **层结构**: 48 层，pattern = 3×GDN + 1×标准Attention，`full_attention_interval=4`
- **权重分片**: 39 个 safetensors，共 1949 个张量
- **顶层前缀**: `model.language_model.*`、`model.visual.*`、`lm_head.*`、`mtp.*`

### LLM text_config
| 参数 | 值 |
|---|---|
| hidden_size | 3072 |
| num_hidden_layers | 48 |
| num_attention_heads | 32（标准 Attention 层） |
| num_key_value_heads | 2 |
| head_dim | 256 |
| num_experts | 256（routed experts） |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 1024 |
| shared_expert_intermediate_size | 1024 |
| vocab_size | 248320 |
| max_position_embeddings | 262144 |
| rope_theta | 10000000 |
| partial_rotary_factor | 0.25 |
| mrope_section | [11, 11, 10] |

### GDN（linear_attention 层）参数
| 参数 | 值 |
|---|---|
| linear_conv_kernel_dim | 4 |
| linear_key_head_dim | 128 |
| linear_num_key_heads | 16 |
| linear_value_head_dim | 128 |
| linear_num_value_heads | 64 |

### MTP (Multi-Token Prediction)
- `mtp_num_hidden_layers = 1`（1 个额外 MTP head）
- `mtp_use_dedicated_embeddings = false`
- MTP 层使用**标准 Attention**（非 GDN）+ 完整 MoE（256 experts + shared expert）
- HF 权重前缀为顶层 `mtp.*`（**不在** `model.language_model.` 下）

### Vision Encoder vision_config
| 参数 | 值 |
|---|---|
| depth | 27 |
| hidden_size | 1152 |
| num_heads | 16 |
| intermediate_size | 4304 |
| patch_size | 16 |
| spatial_merge_size | 2 |
| temporal_patch_size | 2 |
| out_hidden_size | 3072 |
| num_position_embeddings | 2304 |

---

## HF 权重命名规则

### LLM 层
- **GDN 层**（层序 0,1,2,4,5,6,... 即非 full_attention 层）:
  - `model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight`
  - `model.language_model.layers.{i}.linear_attn.in_proj_z.weight`
  - `model.language_model.layers.{i}.linear_attn.in_proj_b.weight`
  - `model.language_model.layers.{i}.linear_attn.in_proj_a.weight`
  - `model.language_model.layers.{i}.linear_attn.conv1d.weight`
  - `model.language_model.layers.{i}.linear_attn.out_proj.weight`
  - `model.language_model.layers.{i}.linear_attn.norm.weight`（需 -1 转 zero-centered）
  - `model.language_model.layers.{i}.linear_attn.A_log`
  - `model.language_model.layers.{i}.linear_attn.dt_bias`
- **标准 Attention 层**（层序 3,7,11,... 每 4 层一个）:
  - `model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
  - `model.language_model.layers.{i}.self_attn.{q,k}_norm.weight`
- **MoE（所有层共用）**:
  - `model.language_model.layers.{i}.mlp.experts.gate_up_proj`（已 fused）
  - `model.language_model.layers.{i}.mlp.experts.down_proj`
  - `model.language_model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight`
  - `model.language_model.layers.{i}.mlp.shared_expert_gate.weight`（router gate）
  - `model.language_model.layers.{i}.mlp.gate.weight`（MoE router）

### MTP 权重（顶层 mtp.*，与 model.* 平级）
- `mtp.fc.weight`、`mtp.pre_fc_norm_embedding.weight`、`mtp.pre_fc_norm_hidden.weight`、`mtp.norm.weight`
- `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight`、`{q,k}_norm.weight`
- `mtp.layers.0.input_layernorm.weight`、`mtp.layers.0.post_attention_layernorm.weight`
- `mtp.layers.0.mlp.gate.weight`（router）
- `mtp.layers.0.mlp.experts.{i}.{gate,up,down}_proj.weight`（256 个，**分离格式，非 fused**）
- `mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight`
- `mtp.layers.0.mlp.shared_expert_gate.weight`

### Vision Encoder
- `model.visual.patch_embed.proj.{weight,bias}`
- `model.visual.pos_embed.weight`
- `model.visual.blocks.{i}.attn.qkv.{weight,bias}`（concatenated QKV）
- `model.visual.blocks.{i}.attn.proj.{weight,bias}`
- `model.visual.blocks.{i}.mlp.linear_fc{1,2}.{weight,bias}`
- `model.visual.blocks.{i}.norm{1,2}.{weight,bias}`
- `model.visual.merger.linear_fc{1,2}.{weight,bias}`
- `model.visual.merger.norm.*`

---

## Megatron-Bridge 代码结构

### 关键文件
| 文件 | 作用 |
|---|---|
| `src/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py` | **核心**：HF↔Megatron 权重映射（含 MTP） |
| `src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py` | Megatron 模型 Provider（配置映射） |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/` | Megatron 侧 Qwen3VLModel 实现 |
| `src/megatron/bridge/models/conversion/auto_bridge.py` | AutoBridge：自动选择 bridge |
| `examples/conversion/convert_checkpoints.py` | 转换入口脚本 |

### Bridge 类（已实现，直接可用）
- `Qwen35VLMoEBridge`：注册 `Qwen3_5MoeForConditionalGeneration` → 处理 MoE 变体
- `Qwen35VLBridge`：注册 `Qwen3_5ForConditionalGeneration` → 处理 Dense 变体

---

## 权重映射特殊处理（转换关键难点）

| Mapping 类 | 用途 |
|---|---|
| `AutoMapping` | 简单 1:1 名称映射 |
| `QKVMapping` | 合并分离 Q/K/V → Megatron QKV |
| `ConcatenatedQKVMapping` | Vision concatenated QKV 处理 |
| `GatedMLPMapping` | gate_proj + up_proj → linear_fc1 |
| `FusedGatedExpertMapping` | HF fused gate_up_proj → Megatron linear_fc1（主 decoder MoE 用） |
| `FusedExpertMapping` | HF down_proj → Megatron linear_fc2（主 decoder MoE 用） |
| `GDNConv1dMapping` | GDN conv1d 特殊处理 |
| `GDNLinearMappingSeparate` | 4 个分离 HF GDN proj 合并为 Megatron in_proj |
| `RMSNorm2ZeroCenteredRMSNormMapping` | RMSNorm weight - 1（zero-centered 转换） |
| `ReplicatedMapping` | TP 下需复制的权重（patch_embed、pos_embed 等） |

**关键差异**：主 decoder MoE expert 用 `FusedGatedExpertMapping`（HF 已 fused）；MTP expert 用 `GatedMLPMapping`（HF 是分离的 gate_proj/up_proj）。

### MTP Megatron ↔ HF 名称映射
```
language_model.mtp.layers.0.eh_proj.weight                          ← mtp.fc.weight
language_model.mtp.layers.0.enorm.weight                            ← mtp.pre_fc_norm_embedding.weight
language_model.mtp.layers.0.hnorm.weight                            ← mtp.pre_fc_norm_hidden.weight
language_model.mtp.layers.0.final_layernorm.weight                  ← mtp.norm.weight
language_model.mtp.layers.0.mtp_model_layer.mlp.router.weight       ← mtp.layers.0.mlp.gate.weight
language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight  ← QKVMapping(q/k/v_proj)
language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1  ← GatedMLPMapping(experts.*.gate/up_proj)
language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2  ← experts.*.down_proj
language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.*    ← mtp.layers.0.mlp.shared_expert.*
```

---

## 转换命令

```bash
cd /data/temp/Megatron-Bridge

# 依赖检查
python -c "import transformers; print(transformers.__version__)"  # 需 >= 5.2.0

# HF → Megatron 转换（单进程，不需要 torchrun）
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B \
    --megatron-path ${WORKSPACE}/models/Qwen/Qwen3.5-122B-A10B \
    --torch-dtype bfloat16
```

AutoBridge 会自动识别 `Qwen3_5MoeForConditionalGeneration` → 选择 `Qwen35VLMoEBridge`。

### conversion.sh 中对 MoE 的并行参数参考
```bash
EP=8, PP=1, TP=1  # 转换时
```

---

## SFT 训练并行策略（slurm_sft.sh 推荐）

| 规模 | 配置 |
|---|---|
| 122B-A10B，4 节点 | `TP=2, PP=6, EP=8` |
| 单节点 8 卡 H800（受限） | `TP=2, PP=1, EP=8` 或 `TP=1, PP=1, EP=8` |

MTP 训练配置：
```python
cfg.model.mtp_num_layers = 1          # 启用 MTP（默认）
cfg.model.mtp_loss_scaling_factor = 0.1
# cfg.model.mtp_num_layers = None     # 如需关闭 MTP
```

---

## 环境调研结论（已确认）

### 硬件资源
| 项目 | 情况 | 结论 |
|---|---|---|
| GPU | 8 × NVIDIA L20Y 80GB，全空闲（0 MB used） | 显存充足（转换为 CPU 操作，不需 GPU） |
| 系统内存 | 2TB 总量，~1.9TB 可用 | 远超 HF 模型 234GB，足够 |
| 磁盘（/ overlay） | 2TB 总量，451GB 剩余 | Megatron ckpt ~234GB，足够 |
| HF 模型大小 | 234GB（39 个 safetensors） | — |

### Python 环境
| 项目 | 状态 |
|---|---|
| 系统 Python | 3.11.13（`/root/miniconda3/bin/python`） |
| torch | 2.9.1 ✅ |
| megatron-core | 0.13.0 ✅ |
| transformers | 5.5.4 ✅（>= 5.2.0 要求满足） |
| flash-attn | 2.7.3 ✅ |
| transformer-engine | 2.10.0 ✅ |
| safetensors | 0.5.3 ✅ |
| megatron-bridge | **未安装**（需 `pip install -e . --no-deps`） |

### uv 问题（已绕过）
- `pyproject.toml` 要求 Python 3.12，uv venv 是空的，需从网络下载全量包（torch 等），网络慢
- **解决方案**：直接用系统 Python 3.11，`pip install -e . --no-deps` 安装 megatron-bridge 本体即可

---

## 执行转换步骤

```bash
cd /data/temp/Megatron-Bridge

# Step 1: 安装 megatron-bridge（仅注册包路径，不拉依赖，秒完成）
pip install -e . --no-deps

# Step 2: 设置输出路径
export WORKSPACE=/mnt/tidal-alsh01/dataset/redone/checkpoints/megatron

# Step 3: 执行转换（单进程，AutoBridge 自动识别 Qwen3_5MoeForConditionalGeneration）
python examples/conversion/convert_checkpoints.py import \
    --hf-model /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B \
    --megatron-path ${WORKSPACE}/Qwen3.5-122B-A10B \
    --torch-dtype bfloat16
```

### 转换状态
- [ ] `pip install -e . --no-deps` 完成
- [ ] `convert_checkpoints.py import` 执行中 / 完成
- [ ] 验证 Megatron ckpt 结构正常

---

## 前置检查清单（已验证）

- [x] `transformers >= 5.2.0`：系统已有 5.5.4
- [x] 显存 / 内存 / 磁盘足够
- [x] HF 模型路径可访问
- [ ] `pip install -e . --no-deps` 安装 megatron-bridge
- [ ] 确认 `${WORKSPACE}` 可写
