# USER_TODO — 你需要执行的事项

> 本文件列出 Codewiz 无法在当前会话内完成、需要你(用户)动手的事项。
> 完成一项后请在前面打 `[x]`,把结果(命令输出)贴回 chat 即可。
>
> 角色区分(在不同机器上做不同事):
> - **build host**(专用开发机, 无 GPU): docker build → 镜像
> - **runtime host**(训练节点, 8+ H20): docker run → 转换 / smoke / SFT

---

## ✅ 已完成

- [x] **Task 1**: `docker pull nvcr.io/nvidia/pytorch:25.06-py3`
- [x] **Task 2**: 探镜像(NGC 25.06 内置组件已确认)
  - Python 3.12.3 / Torch 2.8.0a0+nv25.06 / cu12.9 / cuDNN 9.10.2
  - TE 2.4.0 / flash-attn 2.7.4.post1 / **APEX cuda ext OK**
  - 缺: mamba-ssm / causal-conv1d / fla / megatron-core / uv (我们 Dockerfile 装)

---

## 🔴 P0a — 专用开发机(build host)环境准备

### Task 3a: 在专用开发机上 clone 仓库 + init submodule

```bash
cd /data/temp  # 或任意 5+ GB 空闲分区
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git submodule update --init --recursive
```

(若你已经在 build host 上有这份仓库,跳过本步)

- [ ] 完成

---

### Task 3b: 把项目脚本/Dockerfile 同步到 build host

把当前 `/data/temp/Megatron-Bridge/` 里我们新增/修改的文件同步到 build host 上的对应位置:

```text
Dockerfile.qwen35
.dockerignore
docker/entrypoint-mbridge.sh
scripts/check_build_host.sh
scripts/convert_qwen35_122b.py
scripts/run_convert_qwen35_122b.sh
scripts/smoke_qwen35_122b.py
scripts/run_smoke_qwen35_122b.sh
scripts/sft_qwen35_122b.py
scripts/run_sft_qwen35_122b.sh
scripts/verify_mcore_ckpt.py
scripts/watch_convert.sh
docs/qwen35_vl_122b_reproduce.md
docs/build_host_setup.md
memory.md
USER_TODO.md
```

简单做法 `rsync`:

```bash
rsync -avz --exclude=.venv --exclude=.git/objects/pack \
  /data/temp/Megatron-Bridge/ user@build-host:/data/temp/Megatron-Bridge/
```

或者 build host 也是同一台机器,跳过本步。

- [ ] 完成

---

### Task 3c: 在 build host 上跑预检脚本

```bash
cd /data/temp/Megatron-Bridge
bash scripts/check_build_host.sh
```

期望末尾输出:
```
=== Summary ===
  passes: 18+, warnings: 0-1, failures: 0
[READY] Build host is ready.
```

详细环境要求见 `docs/build_host_setup.md`(包括如何装 Docker、配 NGC 登录、网络白名单等)。

如果 `[NOT READY]`,按提示修复 `[FAIL]` 项再重跑。常见问题:
- docker 未装 → `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER`
- nvcr.io 不通 → 配 HTTPS_PROXY 或在能联网的机器上预 pull 后 `docker save` 传过来
- 没 NGC 登录 → `docker login nvcr.io`(用户名 `$oauthtoken`,密码是 NGC API key)
- submodule 没 init → `git submodule update --init --recursive`

- [ ] 预检通过

---

## 🔴 P0b — 镜像构建(build host,~10-22 min)

### Task 4: 构建项目镜像

```bash
cd /data/temp/Megatron-Bridge
DOCKER_BUILDKIT=1 docker build -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 .
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

### Task 5: 镜像内开箱验证(在 build host 上,需要有 GPU 才能跑;无 GPU 跳到 Task 6 在训练机验证)

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

无 GPU 时只验证 import:把上面 `--gpus all` 去掉,且最后一行 `cuda_avail` 会是 `False`(正常)。

- [ ] 完成

---

## 🔴 P0c — 镜像分发到训练机

### Task 6: docker save → 训练机 docker load

详见 `docs/build_host_setup.md` §8(共三种方式: 私有 registry / docker save+scp / 共享 NAS)。

最常用的方式:
```bash
# 在 build host
docker save qwen35-mbridge:cu129 | gzip > /shared_nas/qwen35-mbridge-cu129.tar.gz

# 在每个训练节点
gunzip -c /shared_nas/qwen35-mbridge-cu129.tar.gz | docker load
docker images qwen35-mbridge:cu129  # 验证有了
```

- [ ] 完成

---

## 🟡 P1 — 容器内端到端跑通(在训练机上)

### Task 7: 进容器,验证转换 + smoke 推理

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

### Task 8: 准备多节点

| 项 | 说明 |
|---|---|
| 节点数 | 4 节点(recipe 默认 TP=2 PP=6 EP=8) |
| 镜像分发 | `docker save qwen35-mbridge:cu129 \| ssh node$i docker load` 或 push 到私有 registry |
| MASTER_ADDR | rank-0 节点 IP |
| 共享存储 | mcore ckpt + 训练输出建议挂 NAS |

- [ ] 完成

---

### Task 9: 准备 SFT 数据

选一种(详见 `docs/qwen35_vl_122b_reproduce.md` §5.3):

- [ ] **A. JSONL** (LLaVA 风格): `{id, image, conversations:[{from,value}...]}`
- [ ] **B. HF datasets 名**: 默认 `make_cord_v2_dataset`
- [ ] **C. Energon webdataset**

---

### Task 10: 启动多节点 SFT

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
