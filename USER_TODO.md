# USER_TODO — 你需要执行的事项

> 本文件列出 Codewiz 无法在当前会话内完成、需要你(用户)动手的事项。
> 完成一项后请在前面打 `[x]`,把结果(命令输出)贴回 chat 即可。

---

## ✅ 已完成

- [x] **Task 1**: `docker pull nvcr.io/nvidia/pytorch:25.06-py3`
- [x] **Task 2**: 探镜像(NGC 25.06 内置组件已确认)
  - Python 3.12.3 / Torch 2.8.0a0+nv25.06 / cu12.9 / cuDNN 9.10.2
  - TE 2.4.0 / flash-attn 2.7.4.post1 / **APEX cuda ext OK**
  - 缺: mamba-ssm / causal-conv1d / fla / megatron-core / uv (我们 Dockerfile 装)

---

## 🔴 P0 — 镜像构建(单台 build,~10-15 min)

### Task 3: 构建项目镜像

```bash
cd /data/temp/Megatron-Bridge
docker build -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 .
```

预计耗时:
- COPY 源码 + uv venv:1-2 min
- `uv sync`:2-3 min(只装小 Python 包,跳过所有 cuda 大件)
- 编译 `mamba-ssm` + `causal-conv1d`:**5-15 min** ← 主要时间消耗(nvcc 编译)
- build-time sanity:30 sec

构建过程中如果某个 import 失败,build 会立即终止并打印 traceback,把它贴回来。

- [ ] 完成
- [ ] 输出最后 30 行(尤其是 `Build-time sanity checks` 那段)

---

### Task 4: 镜像内开箱验证(5 sec)

```bash
docker run --rm --gpus all qwen35-mbridge:cu129 \
  python -c "
import torch, transformer_engine, flash_attn, mamba_ssm, causal_conv1d, fla
import fused_weight_gradient_mlp_cuda
from megatron.bridge import AutoBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
from megatron.bridge.recipes.qwen_vl import qwen35_vl_122b_a10b_sft_config
print('IMAGE OK',
      'torch', torch.__version__,
      'TE', transformer_engine.__version__,
      'cuda_avail', torch.cuda.is_available())
"
```

**期望输出**:
```
IMAGE OK torch 2.8.0a0+5228986c39.nv25.06 TE 2.4.0+3cd6870 cuda_avail True
```

- [ ] 完成

---

## 🟡 P1 — 容器内端到端跑通

### Task 5: 进容器,验证转换 + smoke 推理

```bash
docker run --rm -it --gpus all --shm-size=64g --ulimit memlock=-1 \
  --network host \
  -v /data/temp/Megatron-Bridge:/workspace/Megatron-Bridge \
  -v /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource:/models:ro \
  -v /data/temp/workspace:/workspace/runs \
  qwen35-mbridge:cu129 bash
```

进容器后:

```bash
# 已有的 mcore ckpt 在 /workspace/runs/models/Qwen3.5-122B-A10B-mcore (从宿主机挂载)
# 直接跑 smoke 验证容器内 8 卡推理
HF_MODEL=/models/Qwen3.5-122B-A10B \
MCORE_PATH=/workspace/runs/models/Qwen3.5-122B-A10B-mcore/iter_0000000 \
LOG_DIR=/workspace/runs/logs \
bash scripts/run_smoke_qwen35_122b.sh
```

**期望**: 5-15 分钟内输出 `Generated: ... <think>The user wants ...`(同我们裸机已验证的)

如果失败,把日志后 100 行贴回来。常见可能性:
- TE 2.4 与 megatron-core 0.18 的某个 API 不兼容(需要 import-time 错误)
- flash-attn 2.7.4 < Qwen3.5VL 期望的 2.8(运行时错误)

- [ ] 容器内 smoke 通过

---

## 🟢 P2 — 多节点 SFT 真训

### Task 6: 准备多节点

| 项 | 说明 |
|---|---|
| 节点数 | 4 节点(recipe 默认 TP=2 PP=6 EP=8) |
| 镜像分发 | `docker save qwen35-mbridge:cu129 \| ssh node$i docker load` 或 push 到私有 registry |
| MASTER_ADDR | rank-0 节点 IP |
| 共享存储 | mcore ckpt + 训练输出建议挂 NAS |

- [ ] 完成

---

### Task 7: 准备 SFT 数据

选一种(详见 `docs/qwen35_vl_122b_reproduce.md` §5.3):

- [ ] **A. JSONL** (LLaVA 风格): `{id, image, conversations:[{from,value}...]}`
- [ ] **B. HF datasets 名**: 默认 `make_cord_v2_dataset`
- [ ] **C. Energon webdataset**

---

### Task 8: 启动多节点 SFT

每个节点上(`$RANK` 由你的调度器/手动赋值 0..N-1):

```bash
docker run --rm -it --gpus all --shm-size=64g --ulimit memlock=-1 \
  --network host \
  -v /data/temp/Megatron-Bridge:/workspace/Megatron-Bridge \
  -v /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource:/models:ro \
  -v /data/temp/workspace:/workspace/runs \
  -v /data/your_sft_data:/data:ro \
  qwen35-mbridge:cu129 \
  bash -c "
    HF_MODEL=/models/Qwen3.5-122B-A10B \
    MCORE_PATH=/workspace/runs/models/Qwen3.5-122B-A10B-mcore \
    OUTPUT_DIR=/workspace/runs/qwen35_122b_sft \
    LOG_DIR=/workspace/runs/logs \
    NNODES=4 NODE_RANK=$RANK MASTER_ADDR=10.x.x.x MASTER_PORT=29500 \
    bash scripts/run_sft_qwen35_122b.sh full
  "
```

- [ ] 训练启动(各节点 stdout 看到 NCCL 完成 + iteration 1 loss)
- [ ] loss 正常下降
- [ ] checkpoint 保存正常

---

## 📌 反馈格式

回复时按下面格式贴:

```
## Task X 完成
<命令输出>

## 问题/异常
<如有>
```
