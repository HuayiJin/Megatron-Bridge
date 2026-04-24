# Qwen3.5-122B-A10B 多模态 + MTP SFT 项目记忆

> 本文件是项目的"长期记忆",用于跨 session 保留关键决策与上下文。
> 任何重大决策变更后请同步更新此文件。
>
> **本项目主依赖**: NVIDIA NeMo 官方 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
> (Python 包名 `megatron.bridge`),**不是** ISEEKYAN/mbridge (PyPI `mbridge`)。
> 所有 wrapper 脚本、ckpt、SFT 入口均位于 `/data/temp/Megatron-Bridge/`,
> 在 `.venv` 内运行。`/data/temp/Megatron-LM/` 是一份独立 clone,**未使用**。

---

## 1. 项目目标
在 Megatron 框架上,对 **Qwen3.5-122B-A10B** 做 **多模态 (image+text) + MTP** 的 SFT。
- HF 权重路径: `/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-122B-A10B`
- 资源池: 8×H20-141G(可扩展到 128+ 卡)
- 临时产物路径: `/data/temp/`(磁盘 14T,剩 4.2T 可用)

---

## 2. 关键决策(已与用户对齐)

| # | 决策 | 备注 |
|---|---|---|
| 1 | 桥接方案: NVIDIA NeMo 官方 **megatron-bridge** (`from megatron.bridge import ...`) | 放弃 ISEEKYAN/mbridge,因 PyPI 0.15.1 不支持 Qwen3.5 |
| 2 | 临时产物落 `/data/temp/`,跑通流程优先 | |
| 3 | 第一里程碑: 单节点 8 卡 EP=8 跑通 HF→Megatron 权重转换 | |
| 4 | 多模态范围: image + text(暂不含 video) | |
| 5 | MTP: 保留并联合训练,loss × 0.1 | recipe 默认行为 |
| 6 | 数据: 先不准备,跑通转换 + dry-run 后再接入 | |

---

## 3. 模型架构关键事实(来自 HF `config.json`)

- `architectures`: `Qwen3_5MoeForConditionalGeneration`,`model_type`: `qwen3_5_moe`
- **MoE**: 256 experts / top-8 / shared expert (intermediate=1024)
- **混合注意力**: 48 层 = 36 `linear_attention` (GDN) + 12 `full_attention` (3:1)
- **MTP**: `mtp_num_hidden_layers=1`,MTP 层用 standard attention(非 GDN)
- **多模态 ViT**: depth=27, hidden=1152, image_token_id=248056, video_token_id=248057
- **MRoPE Interleaved**: `mrope_interleaved=true`, `mrope_section=[11,11,10]`, `partial_rotary_factor=0.25`
- **特殊**: `attn_output_gate=true`
- **限制**: GDN 不支持 packed seq (THD),SFT 必须 BSHD,**不能开 sequence packing**

---

## 4. 仓库与依赖现状(阶段 A 核验结果)

| 组件 | 路径/状态 |
|---|---|
| Megatron-LM (独立 clone,**不用**) | `/data/temp/Megatron-LM` |
| megatron-bridge 源码 | `/data/temp/Megatron-Bridge` (NVIDIA NeMo 官方) ✓ |
| megatron-bridge 内置 Megatron-LM submodule | `/data/temp/Megatron-Bridge/3rdparty/Megatron-LM` **空,未初始化**,需 `git submodule update --init --recursive` |
| HF 权重 | `/mnt/.../Qwen3.5-122B-A10B` (39 个 safetensors) ✓ |
| GPU | 8×H20-3e 各 140GB 空闲 ✓ |
| 主 conda 环境 (Py 3.11, transformers 4.57.1, TE 2.10.0, fla 0.4.1, mbridge 0.15.1) | **不能用**,与 megatron-bridge 不兼容(要求 Py ≥3.12, transformers ≥5.0) |
| uv | 0.9.18 ✓ |
| 系统 Python 3.12 | `/usr/bin/python3.12` ✓ |
| megatron-bridge 安装 | **待安装**,走官方 uv 隔离 venv 流程 |

### 阶段 A 关键决策
- **隔离环境**: 在 `/data/temp/Megatron-Bridge/.venv/` 通过 `uv` 建独立 venv(Py 3.12),**绝不污染主 conda 环境**
- **megatron-core 来源**: `3rdparty/Megatron-LM` 的 editable 安装(submodule commit `997896883b04dee1259d92889e7344be528eefa8`)
- **submodule 已初始化** ✓ (commit 997896883b)
- **uv venv 已创建** ✓ `/data/temp/Megatron-Bridge/.venv` (Python 3.12.3)
- **build group 已装** ✓ (cython, ninja, numpy, pybind11, setuptools, nvidia-mathdx)
- **主体 sync 已装 333 个包** ✓ (含 megatron-core, megatron-bridge editable)
- **🚨 CUDA 兼容问题**:
  - 驱动 570.133.20 → 最高 CUDA 12.9
  - 但 megatron-bridge `pyproject.toml` 强制 `transformer-engine[pytorch,core_cu13]`、`nvidia-cudnn-cu13` 等 cu13 链路
  - 我们误装了 `torch==2.11.0+cu130`,实测 `cuda_avail=False`
  - **决策(用户拍板)**: A 方案,在 venv 里改装 cu128 链路;不升级驱动、不切容器
- **TE 来源**: 主环境已有 `transformer_engine 2.10.0` (cu12 二进制),venv 内复用同等版本即可,不强制 git f031cf87
- **运行入口**: `/data/temp/Megatron-Bridge/.venv/bin/python ...` 或 `uv run --project /data/temp/Megatron-Bridge python ...`

### 已装链路 (cu128 + 驱动 570 兼容) ✅ 全通
| 包 | 已装版本 | 备注 |
|---|---|---|
| torch | **2.8.0+cu128** | 必须 ≥ 2.8 才能匹配 TE 2.7 ABI |
| nvidia-* | cu12 系列(8.x) | |
| transformer_engine | **2.7.0** | TE 2.10 与 torch 2.8 ABI 不匹配,降到 2.7 OK |
| transformer_engine_torch / cu12 | 2.7.0 | NVIDIA pypi `https://pypi.nvidia.com` 拉取 |
| flash-attn | **2.8.1** | TE 2.7 仅支持 ≤ 2.8.1,2.8.3 会警告 |
| fla | 0.4.2 | |
| mamba-ssm | 2.3.1 | no-build-isolation,需 torch 已装 |
| causal-conv1d | 1.6.1 | no-build-isolation,需 torch 已装 |
| transformers | 5.3.0 | uv sync 装入 |
| megatron-core | 0.18.0+997896883 | submodule editable |
| megatron-bridge | 0.5.0+0fc0e617 | editable |

### 已踩过的坑(用于更新 Dockerfile)
1. **CUDA 13 vs 驱动 570**: 驱动 570 支持上限 CUDA 12.9,**绝不能装 cu13**。`uv sync --all-extras` 默认会装 cu13,必须 `--no-install-package nvidia-*-cu13 --no-install-package torch ...`
2. **torch 重装路径**: 主 sync 后,用 `uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.* torchvision`
3. **TE ABI 兼容**: TE 2.10 (默认 lock) 在 torch 2.8 上 `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` 未定义。降到 **TE 2.7** 即可。
4. **flash-attn 版本**: TE 2.7 最高支持 flash-attn 2.8.1,装 2.8.3 会有 warning
5. **uv 缓存导致 ABI 错乱**: torch 升级后必须 `uv cache clean <pkg>` 然后 `--no-cache --no-deps` 重装 flash-attn / mamba-ssm / causal-conv1d
6. **`override-dependencies` 让 torch 不被装**: pyproject.toml 里 `torch; sys_platform == 'never'` 强制不装 torch,需要主动用 `uv pip install` 装入 venv。
7. **megatron-bridge.AutoBridge 入口** ✅ 可 import
8. **Qwen35VLMoEBridge / Qwen35VLMoEModelProvider** ✅ 已存在 (`models.qwen_vl.qwen35_vl_bridge`)
9. **Apex 未装的 warning**: 不阻塞,fallback 到 Torch Norm,**性能略损但功能完整**
10. **modelopt 与 transformers 5.3 兼容性 warning**: 不影响转换,如训练阶段报错再处理
11. **gradient_accumulation_fusion 默认开启但 ColumnParallelLinear 必须 APEX**: `src/megatron/bridge/utils/fusions.py` 检查 TE 已装就返回 True,但 `tensor_parallel/layers.py:937` 硬要 `fused_weight_gradient_mlp_cuda` (APEX cuda ext)。**workaround**: 转换脚本里 `provider.gradient_accumulation_fusion = False`;**Recipe 默认仍会强制 True**,SFT 还需 patch `can_enable_*` 函数 + CLI override `model.gradient_accumulation_fusion=false`(都已封装在 `scripts/sft_qwen35_122b.py`)
12. **nohup + disown + Python NCCL 进程会变 zombie**: 父 shell 一退出,Python 子进程被 init 收养,NCCL 后台线程崩溃后整个进程进入 `Z (zombie)` 但仍占 CPU。**正确做法**: 用 `tmux new -s qwen-conv` + 前台 `tee log`,detach 用 Ctrl-B D
13. **Python 输出无法看到**: 后台模式下 stdout 默认 line-buffered + 父 shell 退出 → 看不到 print。**双保险**: `PYTHONUNBUFFERED=1` + `python -u`
14. **NCCL_DEBUG=INFO 默认会输出 9000+ 行**: 转换日志里全是 NCCL channel 信息淹没业务输出。**抑制**: `NCCL_DEBUG=WARN`
15. **单卡转换够用**: HF 122B 加载约 244GB → CPU mem (本机 2TB),不需多卡分布式;`CUDA_VISIBLE_DEVICES=0` 即可,避免 NCCL 起 8 卡浪费
16. **TE 2.7 假设 `nvidia.__file__` 非 None**: namespace package 没 `__file__`,TE 在 `_nvidia_cudart_include_dir()` 崩。**workaround**: 入口脚本注入 `nvidia.__file__ = os.path.join(list(nvidia.__path__)[0], "__init__.py")`
17. **cuDNN sublib loading 失败 (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`)**: 装 ViT 的 Conv3d 时 cuDNN 子库 dlopen 失败。即使把 `nvidia-cudnn-cu12` 钉到 9.10.2.21 与系统一致,`F.conv3d` 仍无法初始化 cuDNN 描述符。**workaround**: 入口脚本设 `torch.backends.cudnn.enabled = False`,Conv3d 走 native PyTorch(ViT 仅占整体计算 ~1%,可忽略)
18. **GDN 不支持 packed seq (THD)**: 训练保持 `pack_sequences_in_batch=False`(recipe 默认正确)。CP 也建议 1。
19. **128B HF 模型 device_map="cuda" 必 OOM**: 官方 `compare.py` 不能直接用于 122B 比对。改用 `hf_to_megatron_generate_vlm.py` 只跑 mcore + 真生成,生成出可读文本即等价于过 logits 比对。
20. **mcore ckpt 验证脚本** `scripts/verify_mcore_ckpt.py` 可在无 GPU 环境快速判断转换是否成功(检查 26451 个 tensor key 的桶分布)

### 标准启动方式
```bash
# === 阶段 B: 转换 (单卡, ~23 min) ===
tmux new -s qwen-conv
cd /data/temp/Megatron-Bridge
bash scripts/run_convert_qwen35_122b.sh
# detach: Ctrl-B D, reattach: tmux attach -t qwen-conv

# === 阶段 C 离线验证: ckpt 结构 (无 GPU, ~1 min) ===
.venv/bin/python scripts/verify_mcore_ckpt.py /data/temp/workspace/models/Qwen3.5-122B-A10B-mcore

# === 阶段 C 在线验证: 8 卡 EP=8 真生成 (~5-10 min) ===
bash scripts/run_smoke_qwen35_122b.sh

# === 阶段 D: SFT ===
bash scripts/run_sft_qwen35_122b.sh smoke   # 单节点 OOM 但能验证入口
NNODES=4 NODE_RANK=0 MASTER_ADDR=... MASTER_PORT=29500 \
  bash scripts/run_sft_qwen35_122b.sh full  # 多节点真训

# === 监控 ===
watch -n 10 'bash /data/temp/Megatron-Bridge/scripts/watch_convert.sh'
```

完整复现: `/data/temp/Megatron-Bridge/docs/qwen35_vl_122b_reproduce.md`

---

## 5. megatron-bridge 已提供的现成资产(关键!)

- **Bridge**: `src/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py`
- **Provider**: `src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py`
- **Modeling**: `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/`
- **SFT recipe**: `qwen35_vl_122b_a10b_sft_config`
  - 路径: `src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py:424`
  - 默认: TP=2 PP=6 EP=8, LR=2e-5, GBS=36, 4 节点 32 GPU, seq=4096
- **PEFT recipe**: `qwen35_vl_122b_a10b_peft_config` (LoRA)
- **示例脚本目录**: `examples/models/vlm/qwen35_vl/`
  - `conversion.sh` `inference.sh` `slurm_sft.sh` `slurm_peft.sh` `slurm_sft_fsdp.sh`
- **转换工具**: `examples/conversion/convert_checkpoints.py` (import/export 双向)
- **VLM 推理**: `examples/conversion/hf_to_megatron_generate_vlm.py`
- **MTP 内置**: 默认 `mtp_num_layers=1`, `mtp_loss_scaling_factor=0.1`,
  转换时 MTP 权重默认包含;设 `mtp_num_layers=None` 可丢弃

---

## 6. 标准命令模板(执行时按需调整)

```bash
# 转换 HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
  --hf-model /mnt/tidal-alsh01/.../Qwen3.5-122B-A10B \
  --megatron-path /data/temp/workspace/models/Qwen3.5-122B-A10B-mcore

# 推理验证(单节点冒烟用)
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path /mnt/tidal-alsh01/.../Qwen3.5-122B-A10B \
  --megatron_model_path /data/temp/workspace/models/.../iter_0000000 \
  --image_path "<url-or-path>" \
  --prompt "Describe this image." \
  --tp 1 --pp 1 --ep 8

# SFT (slurm 多节点)
# 见 examples/models/vlm/qwen35_vl/slurm_sft.sh
```

---

## 7. 风险登记

| 风险 | 缓解 |
|---|---|
| megatron-bridge 与本地 Megatron-LM 版本不匹配 | 阶段 A 检查 `3rdparty/` 与 pyproject 依赖 |
| GDN 算子(fla)在 H20 + CUDA 上不可用 | 阶段 A `python -c "import fla"` |
| 122B torch_dist ckpt ≈ 250GB,`/data/temp/` 容量 | 当前剩 4.2T,够;转换前再 `df -h` |
| MRoPE Interleaved 与 `attn_output_gate` 在 mcore 实现是否完整 | 官方 bridge 已支持,推理验证捕捉 |
| 单节点训练放不下 122B(参数 BF16 = 244GB) | 转换无 grad/optim 完全够;训练阶段需多节点 |

---

## 8. 执行计划进度

- **阶段 A**: 环境核验 ✅ 完成
- **阶段 B**: 权重转换 ✅ 完成 (2026-04-23 19:11→19:34, 23 min)
  - 单卡 (CUDA_VISIBLE_DEVICES=0) + CPU init,**peak RAM ~360GB / 2TB**,GPU 仅 2.4GB
  - 启动器: `bash /data/temp/Megatron-Bridge/scripts/run_convert_qwen35_122b.sh`
  - 输出: `/data/temp/workspace/models/Qwen3.5-122B-A10B-mcore` (250.3 GB `__0_0.distcp`)
  - 总参数: **125,086,497,008** ≈ 125B
- **阶段 C**: 转换正确性验证 ✅ 完成 (2026-04-23)
  - **离线结构验证** `scripts/verify_mcore_ckpt.py`: 26451 个 tensor key, 全部桶 PASS
    - vision/ViT 132, embedding 4, output_layer 1, expert_mlp **25382**
    - **GDN/linear_attn 288** (A_log, dt_bias, conv1d, in_proj.weight.{alpha,beta,z})
    - full_attention 125, **MTP 539**, shared_expert 245
  - **8 卡 EP=8 真生成 smoke** `scripts/run_smoke_qwen35_122b.sh`:
    - 加载 mcore + VL forward + 32 token greedy 生成成功
    - 输出 `<think>\nThe user wants a one-sentence description...` 完全合理
    - 证明: ViT + GDN + MoE + MRoPE + EP + 整套 forward 全部正确
- **阶段 D**: SFT 入口就绪 ✅ (2026-04-23, 待真跑训练)
  - 入口: `scripts/sft_qwen35_122b.py` + `scripts/run_sft_qwen35_122b.sh`
  - 模式: `smoke` (单节点,会OOM,仅作 shape) / `full` (多节点)
  - Recipe: `qwen35_vl_122b_a10b_sft_config` (TP=2 PP=6 EP=8, 4 节点 32 GPU 默认)
- **阶段 E**: 接入内部数据 + 正式训练 ⏳ 待办

---

## 9. 待用户后续确认

1. 节点数(128 卡是否配 16 节点?)
2. SFT 全量 vs LoRA(取决于阶段 D 后的吞吐评估)
3. 内部数据格式与字段约定

---

## 10. 镜像构建决策 (2026-04-24)

### 用户需求
- Base 选 NGC PyTorch **`25.06-py3`** (CUDA 12.9.1 + Torch 2.8.0a0 + cuDNN 9 + TensorRT 10.11)
- CUDA 12.x 取最新(即 12.9.1,**不再装 cu13**)
- 镜像构建后**直接可用**(开箱即用 SFT)

### 重构后 Dockerfile.qwen35 关键改动
| 改动 | 原 | 新 | 原因 |
|---|---|---|---|
| Torch 安装 | `pip install torch==2.8.* cu128` | **不动**,继承 NGC 自带 | NGC 自带优化版,cu128 wheel 会降级 + 库混乱 |
| TE 安装 | `pip install transformer-engine==2.7.*` | **优先继承 NGC**,`uv sync` 跳过 | NGC 自带 TE 通常已优化 |
| cuDNN | `pin nvidia-cudnn-cu12==9.10.2.21` | **不装**,用系统 cuDNN 9 | 之前的 sublib loading 错就是版本错乱;NGC 内自洽 |
| flash-attn / mamba-ssm / causal-conv1d | 强行重编 | **优先 NGC**,`uv sync` 跳过 | NGC 通常已带预编版,可省 30-60 min build 时间 |
| APEX | 不装 | **build-time 检测**,无则源码编译(可 `--build-arg BUILD_APEX_IF_MISSING=0` 关闭) | APEX 装上后可干掉 `gradient_accumulation_fusion=False` patch |
| Source 映射 | `/opt/build/Megatron-Bridge` 与运行时挂载脱钩 | **`/opt/Megatron-Bridge` fallback + entrypoint 检测 `/workspace/Megatron-Bridge` 挂载优先** | 既能开箱即用,也能挂宿主机源码热改 |
| venv 路径 | `/opt/venv`(可能与 NGC 冲突) | **`/opt/venv-mbridge`**(独立) | 避免与 NGC 自带 `/opt/venv` 撞名 |

### 配套文件
- `docker/entrypoint-mbridge.sh` 实现"挂载优先 + 自动 reinstall editable + 自动 init submodule"
- `.dockerignore` 调整:不再屏蔽 `.git`(让 build 期能 `git submodule update --init`)

### 实施进度
1. ✅ `docker pull nvcr.io/nvidia/pytorch:25.06-py3`(用户已完成)
2. ✅ **探镜像完成**(2026-04-24)— 结果如下表
3. ✅ Dockerfile 已精确化
4. ⏳ `docker build -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 .`(用户待执行)
5. ⏳ 镜像内 sanity + 容器内 smoke

### 探镜像结果(NGC PyTorch 25.06-py3)

| 组件 | 版本 | 我们的处理 |
|---|---|---|
| Python | 3.12.3 (`/usr/bin/python`) | 用之 |
| PyTorch | **2.8.0a0+5228986c39.nv25.06** / cuda 12.9 / cuDNN 91002 | 继承,绝不装 |
| TransformerEngine | **2.4.0+3cd6870** | 继承(注意:不是 2.7,与之前裸机不同) |
| cuDNN libs | `/usr/lib/x86_64-linux-gnu/libcudnn*.so.9.10.2` | 系统级,继承 |
| flash-attn | **2.7.4.post1** | 继承(版本略低于 2.8.1,需观察 Qwen3.5-VL 是否兼容) |
| **APEX cuda ext** | **OK** (`fused_weight_gradient_mlp_cuda` + `FusedRMSNorm`) | 继承,**所有 grad-fusion patch 默认 no-op** |
| mamba-ssm | ❌ 没装 | Dockerfile 装 `==2.3.1` |
| causal-conv1d | ❌ 没装 | Dockerfile 装 `==1.6.1` |
| fla | ❌ 没装 | Dockerfile 装 `==0.4.2` (`flash-linear-attention`) |
| megatron-core | ❌ 没装 | submodule editable |
| uv | ❌ 没装 | Dockerfile 装 `0.9.18` |
| nvcc | 12.9 | 系统级 |

### NGC 容器内 monkey-patch 默认状态(全部 no-op)

所有 wrapper 脚本(`scripts/{convert,smoke,sft}_qwen35_122b.py`)的 patch 改为 **环境变量开关**:

| 环境变量 | 默认 | 行为 |
|---|---|---|
| `MBRIDGE_PATCH_NVIDIA_FILE` | `0` (NGC 容器) | NGC TE 2.4 没有 namespace 问题,无需 patch |
| `MBRIDGE_DISABLE_CUDNN` | `0` (NGC 容器) | NGC cuDNN 9.10 健康,**保留 cuDNN** 让 ViT Conv3d 走优化路径 |
| `MBRIDGE_PATCH_GRAD_FUSION` | `0` (NGC 容器) | NGC APEX cuda ext 齐全,recipe 默认 `gradient_accumulation_fusion=True` 直接可用 |

**裸机回退**: 在主 conda 环境(无 APEX,有 cuDNN 子库 bug)运行时,需要 `export MBRIDGE_PATCH_NVIDIA_FILE=1 MBRIDGE_DISABLE_CUDNN=1 MBRIDGE_PATCH_GRAD_FUSION=1`。
Dockerfile 默认 `MBRIDGE_DISABLE_CUDNN=0`(其他两个 Python 默认就是 0)。

---

## 11. 角色分离: build host vs runtime host (2026-04-24)

### 概念
- **build host**(专用开发机): `docker build` 出镜像,**不需要 GPU**
- **runtime host**(训练节点): `docker run --gpus all`,需要 GPU + driver + nvidia-container-toolkit

### Build host 硬性要求
| 项 | 要求 | 备注 |
|---|---|---|
| OS | Linux x86_64 | NGC 镜像不支持 ARM |
| Docker | ≥ 20.10,推荐 24+ | 需要 BuildKit |
| 磁盘 (Docker root) | ≥ 80GB | 镜像 ~35GB + 中间层 |
| 磁盘 (build context) | ≥ 5GB | source tree + submodule ~200MB |
| RAM | ≥ 16GB,推荐 32GB | nvcc 编译 mamba-ssm 较吃内存 |
| 网络 | nvcr.io / pypi.org / pypi.nvidia.com / github.com / astral.sh | 全部能访问 |
| nvcr.io 凭据 | `docker login nvcr.io` | 免费 NGC API key |
| 源码 | `git clone NVIDIA-NeMo/Megatron-Bridge && git submodule update --init --recursive` | submodule 必须 init |

### Build host **不需要**
- ❌ NVIDIA driver / nvidia-container-toolkit (build 不调 GPU)
- ❌ CUDA toolkit / nvcc (NGC 镜像内自带 12.9)
- ❌ Python / pip / conda / uv (全在镜像内)

### 工具与文档
- `scripts/check_build_host.sh` 一键预检 build host(15 项检查,FAIL/WARN 分级)
- `docs/build_host_setup.md` 完整 build host 准备指南(含一键安装、网络白名单、镜像分发、训练机要求)
- `USER_TODO.md` 已重排为 P0a (build host 准备) → P0b (build) → P0c (分发) → P1 (容器内验证) → P2 (多节点 SFT)

### 镜像分发(build host → runtime hosts)
三种方式(详见 `docs/build_host_setup.md` §8):
1. 私有 registry: `docker tag` + `docker push` + `docker pull`(生产标配)
2. `docker save` + scp/rsync(简单,无 registry)
3. 共享 NAS 上 `docker save` 一份(所有节点 `docker load`)

镜像约 35GB,gzip 后约 12-15GB。

---

## 12.5 Runtime-deferred 依赖策略 (2026-04-24)

### 决策
将 `mamba-ssm` / `causal-conv1d` / `fla` **从镜像移出**,改为容器启动时安装。

### 动机
1. **build 期没 GPU**: `import mamba_ssm` 在 docker build 阶段触发 `RuntimeError: 0 active drivers` —— mamba-ssm 在 import 时探测 CUDA driver。这导致 build-time sanity 必须用 `find_spec` 兜底,体验差。
2. **build 时间**: 这两个 nvcc 编译占总 build 时间 70%+(5-15 min),挪到 runtime 后 build 缩到 3-5 min。
3. **镜像体积**: 这三个包 + nvcc 中间产物约 400MB。
4. **灵活性**: 想换 mamba-ssm 版本不用 rebuild 镜像。

### 实现
- `Dockerfile.qwen35` 删除 STEP 19 (mamba/causal-conv1d 安装) 和 STEP 18 (fla 安装)
- 新增 `scripts/install_runtime_deps.sh` —— 幂等安装脚本
- `docker/entrypoint-mbridge.sh` 启动时**自动调用** install_runtime_deps.sh
- Build sanity 改回**全部真 import**(因 mamba_ssm 不在了)
- Runtime sanity 检测缺失则提示用户跑 install 脚本

### 性能优化:venv 持久化
首次启动等 5-15 min,后续避免重装的方法:
```bash
docker run ... -v /shared/qwen35-venv:/opt/venv-mbridge ...
```
把整个 venv site-packages 挂到宿主机,跨容器复用。

### 关闭自动安装(如要镜像内手动)
```bash
docker run -e MBRIDGE_SKIP_RUNTIME_INSTALL=1 ...
# 然后在容器内手动:
bash scripts/install_runtime_deps.sh
```

---

## 12. Dockerfile submodule 检测 fix (2026-04-24)

### 症状
首次 `docker build` 在 `STEP 14: git submodule update` 失败:
```
fatal: not a git repository (or any of the parent directories): .git
```

### 根因
- 我之前的检测条件 `[ ! -f 3rdparty/Megatron-LM/megatron/__init__.py ]` 永远为真,因为 Megatron-LM 的 `megatron/` 是 PEP 420 namespace package,**没有顶层 __init__.py**(只有子包有)
- 进入分支后尝试 `git submodule update`,但 `docker build` 默认不把 `.git/` 拷进 image,导致 "not a git repository"

### Fix
- 检测条件改为 `[ ! -f 3rdparty/Megatron-LM/megatron/core/__init__.py ]`(`core/` 是 init 后必存在的子包)
- 不再尝试在镜像内 git submodule init,改为 fail-fast 提示用户在 host 上 init
- 已加 `scripts/check_build_host.sh` 预检 submodule 状态

---

_最后更新: 2026-04-24 (build host 角色 + submodule fix + 一键预检脚本)_
