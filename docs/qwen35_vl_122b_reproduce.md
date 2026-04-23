# Qwen3.5-122B-A10B 多模态 + MTP SFT 复现指南

本文档记录了在 8×H20-141G(单节点)+ NVIDIA driver 570.133.20 + CUDA 12.9 的环境下,
使用 NVIDIA NeMo 官方 **megatron-bridge** 跑通 Qwen3.5-122B-A10B 的完整流程:
HF 权重转 mcore → 8 卡推理冒烟 → SFT 训练入口。

> 阶段 D(SFT)真正 122B 全量 SFT 需多节点(官方推荐 4×8=32 GPU TP=2 PP=6 EP=8);
> 单节点 SFT 仅作为 shape-correctness smoke,会 OOM。

## 0.0 关于 "bridge" 的命名澄清(很重要)

业界有**两个**同时被叫 "bridge" 的项目,功能不同,**不要混淆**:

| | **本项目使用** | 不是本项目使用的 |
|---|---|---|
| Python 包 | `megatron.bridge` | `mbridge` |
| 安装方式 | 从 `/data/temp/Megatron-Bridge` 源码 + uv venv | `pip install mbridge`(0.15.1 在 PyPI) |
| 维护方 | NVIDIA NeMo Framework 团队 | 个人开源(ISEEKYAN) |
| Qwen3.5 支持 | ✅ `Qwen35VLMoEBridge` 一等公民 | ❌ 0.15.1 完全不支持 |
| 主要场景 | 大规模 SFT/Pretrain 训练管道 | RL Actor 路径(verl 等) |
| 本项目入口 | `from megatron.bridge import AutoBridge` | (未使用) |

verl demo (`run_qwen3_5-122b-a10b-megatron.sh`) 用的是 `mbridge`(因为 verl 适配 RL Actor),
但官方 SFT 路径走的是 megatron-bridge 的 recipe + `examples/conversion/`。我们走 SFT,所以选后者。

所有项目脚本与产物均位于 `/data/temp/Megatron-Bridge/`,在它的 `.venv` 内执行。
`/data/temp/Megatron-LM/` 是一份独立 clone,本项目**未使用**(仅留软链 memory.md 指回主位置)。

---

## 0. 环境前置

| 项 | 要求 | 备注 |
|---|---|---|
| GPU | ≥ 8 张 H20-141G(或同等) | 8 卡 EP=8 推理刚好放下 |
| Driver | ≥ 555 (支持 CUDA 12.8) | 我们的 570.133.20 ✓ |
| 系统 Python | 3.12 | `/usr/bin/python3.12` |
| uv | ≥ 0.9 | `/root/.local/bin/uv` |
| 磁盘 | `/data/temp` ≥ 500GB 空闲 | mcore ckpt ~250GB |
| 主存 | ≥ 400GB | HF 加载 peak ~360GB |
| 网络 | github.com 直连 | 拉 submodule + git deps |

---

## 1. 一次性环境构建

```bash
# 仓库与 submodule
cd /data/temp/Megatron-Bridge
git submodule update --init --recursive  # ~ 1 分钟,拉 NVIDIA/Megatron-LM (commit 997896883)

# uv venv
uv venv .venv --python 3.12 --system-site-packages

# build group
uv sync --locked --only-group build

# 主体 sync (~30-60 min;会装 333+ 包,但跳过 cu13 + torch + TE 后我们自己装)
UV_LINK_MODE=copy uv sync --link-mode copy --locked --all-extras --all-groups

# 卸载 sync 误装的 cu13 链路
uv pip uninstall --python /data/temp/Megatron-Bridge/.venv/bin/python \
  torch triton cuda-toolkit \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx

# torch 2.8 cu128 (与 driver 570 兼容)
UV_LINK_MODE=copy uv pip install --link-mode copy \
  --python /data/temp/Megatron-Bridge/.venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.*" torchvision

# 把 cuDNN 钉到 9.10.2.21 与系统一致(避免 sublib loading 失败)
UV_LINK_MODE=copy uv pip install --link-mode copy --no-deps \
  --python /data/temp/Megatron-Bridge/.venv/bin/python \
  "nvidia-cudnn-cu12==9.10.2.21"

# TransformerEngine 2.7 (TE 2.10 与 torch 2.8 ABI 不匹配)
UV_LINK_MODE=copy uv pip install --link-mode copy --no-build-isolation \
  --python /data/temp/Megatron-Bridge/.venv/bin/python \
  --extra-index-url https://pypi.nvidia.com \
  "transformer-engine[pytorch]==2.7.*"

# 原生 ext (与 torch 2.8 重编)
UV_LINK_MODE=copy uv pip install --link-mode copy --no-build-isolation --no-cache --no-deps \
  --python /data/temp/Megatron-Bridge/.venv/bin/python \
  "flash-attn==2.8.1" "causal-conv1d==1.6.1" "mamba-ssm==2.3.1"

# 自检
.venv/bin/python -c "
import torch, transformer_engine.pytorch, flash_attn, fla, mamba_ssm, causal_conv1d
import megatron.core, megatron.bridge
from megatron.bridge import AutoBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
print('All OK')"
```

---

## 2. HF → Megatron 权重转换

```bash
# 单卡, CPU init, ~23 分钟, peak RAM ~360GB
bash scripts/run_convert_qwen35_122b.sh
```

**产出**:
- `/data/temp/workspace/models/Qwen3.5-122B-A10B-mcore/`
  - `latest_checkpointed_iteration.txt`  → "0"
  - `iter_0000000/`
    - `__0_0.distcp` (~233GB)
    - `.metadata`, `metadata.json`, `run_config.yaml`, `common.pt`, `tokenizer/`

**关键 workaround**: `scripts/convert_qwen35_122b.py` 显式设置 `provider.gradient_accumulation_fusion = False`
(megatron-bridge 检测到 TE 已装就默认开启 fusion,但 `ColumnParallelLinear` 硬要 APEX cuda 扩展)。

---

## 3. 验证 mcore ckpt 内容(无 GPU)

```bash
.venv/bin/python scripts/verify_mcore_ckpt.py /data/temp/workspace/models/Qwen3.5-122B-A10B-mcore
```

**期望输出**:
```
[1/3] File presence + sizes ... ✅
[2/3] run_config.yaml sanity ... ✅ (Qwen35VLMoEModelProvider, mtp=1, experts=256, topk=8)
[3/3] dist_checkpointing metadata
   ✅ vision/ViT             count=  132
   ✅ embedding              count=    4
   ✅ output_layer           count=    1
   ✅ expert_mlp             count=25382
   ✅ gdn/linear_attn        count=  288   <-- A_log/dt_bias/conv1d/in_proj.weight.{alpha,beta,z}
   ✅ full_attention         count=  125
   ✅ mtp                    count=  539
   ✅ shared_expert          count=  245
✅ All hard structural checks passed.
```

---

## 4. 推理冒烟(8 卡 EP=8,真生成 token)

```bash
bash scripts/run_smoke_qwen35_122b.sh
```

**期望**(第 5-15 分钟内会输出):
```
Step 0: ... Top 5: [('The', ...), ('1', ...), ('Based', ...), ('Thinking', ...), ...]
Step 1: ... Selected: ' user'
Step 2: ... Selected: ' wants'
...
======== GENERATED TEXT OUTPUT ========
Generated: <|im_start|>user\n... assistant\n<think>\nThe user wants a one-sentence description of the provided image.\n\n1. Identify the main subject: ...
```

如果输出可读、与 prompt/image 相关,即证明:ViT + GDN + MoE + MRoPE-Interleaved + EP 全部正确。

**关键 workaround** (在 `scripts/smoke_qwen35_122b.py` 中):
1. **`nvidia.__file__` patch** —— TE 2.7 假设它非 None,但 namespace package 没有
2. **`torch.backends.cudnn.enabled = False`** —— 系统 cuDNN 9.10 装得有问题(`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`),禁用后 ViT 的 Conv3d 走 native PyTorch
3. **`can_enable_gradient_accumulation_fusion = lambda: False`** —— 同上,无 APEX

---

## 5. SFT 训练

### 5.1 单节点 8 卡 smoke (会 OOM,仅作 shape 验证)

```bash
bash scripts/run_sft_qwen35_122b.sh smoke
```

会用 TP=1 PP=8 EP=1 + recompute=full + GBS=8 MBS=1。**此场景 122B + Adam 显存约 1.5TB,8 卡 1.1TB 不够**;
如果想看 SFT 入口脚本能否启动,可以用 dense 小模型做替身验证。

### 5.2 多节点(推荐)

官方 recipe `qwen35_vl_122b_a10b_sft_config` 默认 **4 节点 32 GPU**,TP=2 PP=6 EP=8,GBS=36, seq=4096, LR=2e-5。

每个节点上执行(需先 `pdsh`/`pssh` 同步分发):

```bash
# 在 master 节点
NNODES=4 NODE_RANK=0 MASTER_ADDR=10.x.x.x MASTER_PORT=29500 \
  bash scripts/run_sft_qwen35_122b.sh full

# 在其它节点 (rank 1..3)
NNODES=4 NODE_RANK=1 MASTER_ADDR=10.x.x.x MASTER_PORT=29500 \
  bash scripts/run_sft_qwen35_122b.sh full
```

可调环境变量:
| 变量 | 默认 | 说明 |
|---|---|---|
| `GBS` | 32 | global batch size |
| `MBS` | 1 | micro batch size |
| `ITERS` | 1000 | train iters |
| `SAVE_INTERVAL` | 500 | ckpt 保存间隔 |
| `DATASET` | `make_cord_v2_dataset` | 数据集 maker |
| `OUTPUT_DIR` | `/data/temp/workspace/runs/qwen35_122b_sft` | 输出 |

### 5.3 接入私有数据

Recipe 默认走 HF datasets (`make_cord_v2_dataset` = CORD v2 OCR)。私有数据接入两种姿势:

**A. JSON 预加载** (最简):
```bash
# 用 --dataset vlm-preloaded 替代 vlm-hf
... --dataset vlm-preloaded \
    --step_func vlm_step \
    dataset.train_data_path=/data/my_vlm_train.json \
    dataset.image_folder=/data/my_vlm_images \
    dataset.hf_processor_path=$HF_MODEL
```
JSON schema (LLaVA 风格):
```json
{
  "id": "...",
  "image": "rel/path.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image."},
    {"from": "gpt", "value": "..."}
  ]
}
```

**B. Energon webdataset** (大规模):
```bash
... --dataset vlm-energon \
    --step_func vlm_step \
    dataset.path=/data/my_energon_dir
```

---

## 6. 已知坑总结

| # | 坑 | 修法 |
|---|---|---|
| 1 | `--all-extras` 默认装 cu13 | 卸载,改装 cu128 |
| 2 | `pyproject.toml` 把 torch 标 `sys_platform=='never'` | 用 `uv pip install --index-url ...` 绕过 |
| 3 | TE 2.10 与 torch 2.8 ABI 错 | 降到 TE 2.7 |
| 4 | flash-attn 2.8.3 超出 TE 2.7 上限 | 用 2.8.1 |
| 5 | `nohup &` 让 Python NCCL 进程变 zombie | 用 `tmux` + 前台 `tee` |
| 6 | `gradient_accumulation_fusion` 默认 True 但 APEX 没装 | Python patch + CLI override `model.gradient_accumulation_fusion=false` |
| 7 | TE 2.7 假设 `nvidia.__file__` 非 None | 入口脚本注入 `nvidia.__file__` |
| 8 | 系统 cuDNN 9.10 sublib dlopen 失败 | `torch.backends.cudnn.enabled = False`(ViT Conv3d 走 native) |
| 9 | GDN 不支持 packed seq (THD) | 训练保持 `pack_sequences_in_batch=False` |

---

## 7. 文件清单

| 路径 | 用途 |
|---|---|
| `scripts/convert_qwen35_122b.py` | HF → mcore Python 入口(单卡) |
| `scripts/run_convert_qwen35_122b.sh` | 转换 shell 启动器(unbuffered + 单卡 + tmux 友好) |
| `scripts/verify_mcore_ckpt.py` | mcore ckpt 离线结构验证(无 GPU) |
| `scripts/smoke_qwen35_122b.py` | mcore 加载 + VLM 生成(带 patch 注入) |
| `scripts/run_smoke_qwen35_122b.sh` | smoke 启动器(8 卡 EP=8 + tee) |
| `scripts/sft_qwen35_122b.py` | SFT Python 入口(带 patch 注入) |
| `scripts/run_sft_qwen35_122b.sh` | SFT 启动器(`smoke` / `full` 两模式,支持多节点) |
| `scripts/watch_convert.sh` | 转换实时监控仪表盘 |
| `Dockerfile.qwen35` | 完整环境 Dockerfile,以上所有坑都已 hardcode |
| `/data/temp/Megatron-LM/memory.md` | 项目长期记忆 |

---

## 8. 性能数字

| 阶段 | 资源 | 时间 |
|---|---|---|
| uv sync 全套 | — | ~30-60 min |
| 装 cu128 链路 + 重编 ext | — | ~15 min |
| 转换 (单卡 CPU init) | RAM peak 360G | 23 min |
| Smoke 加载+生成 32 token | 8×H20 | ~5-10 min(首次含 NCCL bootstrap) |
| 全量 SFT 一步 (4×8 GPU) | TP=2 PP=6 EP=8 | TBD(待真跑) |
