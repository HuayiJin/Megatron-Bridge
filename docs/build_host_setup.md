# 专用开发机(构建机)准备指南

> 本文档面向**用来 `docker build` 出 `qwen35-mbridge:cu129` 镜像的那台机器**。
> 该机器与"运行训练的机器"分离,**不需要 GPU**,但需要 Docker + 网络 + 磁盘。

---

## 0. 角色界定

| 角色 | 职责 | GPU | nvidia-driver | nvidia-container-toolkit | docker |
|---|---|---|---|---|---|
| **build host (本文档)** | `docker build` | ❌ | ❌ | ❌ | ✅ |
| **runtime host** (训练节点) | `docker run --gpus all ...` | ✅ ≥ 8×H20-141G | ✅ ≥ 555 | ✅ | ✅ |

构建一次,镜像通过 `docker save / docker load` 或私有 registry 分发到所有训练节点。

---

## 1. 硬性要求(检查脚本会强制校验)

| 项 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| OS | Linux x86_64 | Ubuntu 22.04 / 24.04 | NGC 镜像不支持 ARM |
| Docker | 20.10 | 24.x+ | 需要 BuildKit (`DOCKER_BUILDKIT=1` 或 buildx) |
| 磁盘 (Docker root) | 80GB | 150GB | `qwen35-mbridge:cu129` 终态 ~35GB,中间层叠加约 60GB |
| 磁盘 (build context) | 5GB | 10GB | source tree + submodule ~200MB,留余量 |
| RAM | 16GB | 32GB+ | nvcc 编译 mamba-ssm/causal-conv1d 单文件可吃 4-6GB |
| 网络 | 能访问下列域名 | | 见 §3 |
| nvcr.io 凭据 | 已 `docker login nvcr.io` | | 免费 NGC API key |

## 2. 不需要的东西(避免误装)

- ❌ NVIDIA driver(build 不调用 GPU)
- ❌ CUDA toolkit / nvcc(NGC 镜像内自带 nvcc 12.9)
- ❌ Python / pip / conda / uv(全部在镜像内)
- ❌ TransformerEngine / torch / 任何 ML 库
- ❌ APEX(NGC 镜像内自带,我们不重编)

> 如果开发机本来就装了上面这些,不影响 docker build,只是多余。

## 3. 网络白名单

构建时需要从下列域名拉东西:

| 域名 | 用途 | 失败时影响 |
|---|---|---|
| `nvcr.io` | 拉 NGC base 镜像 (~30GB) | **致命**,build 不会开始 |
| `pypi.org` | uv sync 主源 | uv sync 失败 |
| `pypi.nvidia.com` | NVIDIA 扩展索引(本 Dockerfile 暂未用,留作后用) | warn |
| `github.com` | mamba-ssm/causal-conv1d/fla 等 git+url 依赖 | uv sync 失败 |
| `astral.sh` | uv 安装器 | uv 装不上 |
| `download.pytorch.org` | (未使用,留作 fallback) | n/a |

**如在企业内网**:
- 配 `HTTPS_PROXY=http://proxy.example.com:8080` 在 Dockerfile build 期生效:
  ```bash
  docker build \
    --build-arg HTTPS_PROXY=$HTTPS_PROXY \
    --build-arg HTTP_PROXY=$HTTP_PROXY \
    --build-arg NO_PROXY=$NO_PROXY \
    -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 .
  ```
- 或事先用 `docker pull nvcr.io/nvidia/pytorch:25.06-py3` 拉到本地缓存

---

## 4. 一键安装(Ubuntu 22.04 / 24.04 全新机器)

```bash
# 1. Docker (官方一键)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"  # 注销重登 / newgrp docker

# 2. 启用 BuildKit (现代 Docker 默认开)
echo '{"features":{"buildkit":true}}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker

# 3. NGC 登录(去 https://ngc.nvidia.com 注册免费账号,生成 API key)
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>

# 4. 一些便利工具
sudo apt-get install -y tmux git curl jq
```

## 5. 准备代码仓库

```bash
# Clone Megatron-Bridge (NVIDIA NeMo 官方)
cd /data/temp  # 或任何有 5+ GB 空闲的位置
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge

# 拉取 Megatron-Core submodule (commit 997896883b)
git submodule update --init --recursive
```

## 6. 一键预检

在 `Megatron-Bridge` 目录下运行:

```bash
bash scripts/check_build_host.sh
```

期望输出:
```
=== Summary ===
  passes: 18+, warnings: 0-1, failures: 0
[READY] Build host is ready.
```

如果 `[NOT READY]`,按提示修复对应项再重跑。

## 7. 构建镜像

```bash
# (一次性) 拉 NGC base 镜像 ~30GB
docker pull nvcr.io/nvidia/pytorch:25.06-py3

# 构建项目镜像
DOCKER_BUILDKIT=1 docker build -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 .
```

预计耗时 10-22 分钟(主消耗在 mamba-ssm/causal-conv1d nvcc 编译)。

构建成功的标志:Dockerfile sanity 步骤打印 `OK: image is ready to use`,且最后 `docker images qwen35-mbridge:cu129` 看到镜像 ~35GB。

---

## 8. 分发到训练节点

镜像构建完后,分发到训练节点的两种方式:

### 方式 A:私有 registry(推荐,生产环境标配)

```bash
# 在 build host
docker tag qwen35-mbridge:cu129 your-registry.example.com/qwen35-mbridge:cu129
docker push your-registry.example.com/qwen35-mbridge:cu129

# 在每个训练节点
docker pull your-registry.example.com/qwen35-mbridge:cu129
docker tag your-registry.example.com/qwen35-mbridge:cu129 qwen35-mbridge:cu129
```

### 方式 B:`docker save` + scp(简单,无 registry)

```bash
# 在 build host (镜像约 35GB)
docker save qwen35-mbridge:cu129 | gzip > qwen35-mbridge-cu129.tar.gz

# 分发(根据网络选 scp/rsync/共享 NAS)
for node in node{1..4}; do
    rsync -avP qwen35-mbridge-cu129.tar.gz $node:/tmp/
done

# 在每个训练节点
gunzip -c /tmp/qwen35-mbridge-cu129.tar.gz | docker load
```

### 方式 C:共享 NAS 上 `docker save` 一份

如果所有训练节点挂载同一个 NAS:
```bash
docker save qwen35-mbridge:cu129 | gzip > /shared_nas/qwen35-mbridge-cu129.tar.gz
# 训练节点
gunzip -c /shared_nas/qwen35-mbridge-cu129.tar.gz | docker load
```

---

## 9. 训练节点(runtime host)的额外要求

> 这台不是 build host,但顺手贴一下,避免你下一步又问。

| 项 | 要求 |
|---|---|
| GPU | ≥ 8×H20-141G(单节点 smoke);多节点 SFT 推荐 4 节点 32 卡 |
| NVIDIA Driver | ≥ 555(对应 CUDA 12.8+,我们的 570 OK) |
| `nvidia-container-toolkit` | 已装且 `docker info` 列出 `nvidia` runtime |
| Docker | ≥ 20.10 |
| 节点间网络 | NCCL 端口互通(典型 29500 + 动态端口) |
| HF 权重共享存储 | `/mnt/.../Qwen3.5-122B-A10B/` 全部节点可读 |
| mcore ckpt 共享存储 | 转换产物 `/data/temp/workspace/models/...` 全部节点可读 |

```bash
# 训练节点装 nvidia-container-toolkit (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all qwen35-mbridge:cu129 nvidia-smi
```

---

## 10. 常见构建失败 & 修复

| 症状 | 原因 | 修复 |
|---|---|---|
| `fatal: not a git repository` at `git submodule update` | submodule 未在 host init | `git submodule update --init --recursive` |
| `Error response from daemon: Get nvcr.io: ...denied` | 没 `docker login nvcr.io` | 重做登录 |
| `no space left on device` | docker root 满 | `docker system prune -af --volumes` 或迁 `/etc/docker/daemon.json` 的 `data-root` |
| `failed to fetch ... pypi.org` | 无外网 | 配 HTTPS_PROXY 或换内网 PyPI 镜像 |
| nvcc 编译 mamba-ssm OOM | RAM 不够 | 降低并行 `MAX_JOBS=2`,在 Dockerfile build-arg 里加 `ENV MAX_JOBS=2` |
| `transformer_engine` import 失败(sanity 阶段) | NGC TE 2.4 与 megatron-core 0.18 某 API 不兼容 | 贴 traceback 给我们排查;最后回退方案 = 在镜像里覆盖装 TE 2.7 |

---

## 11. 镜像清单

构建产物里的关键路径:

```
qwen35-mbridge:cu129
├── /opt/Megatron-Bridge/          # baked-in 源码 (fallback)
│   └── 3rdparty/Megatron-LM/      # editable megatron-core
├── /opt/venv-mbridge/             # uv venv,继承 NGC site-packages
│   └── bin/python                 # PATH 顶
├── /usr/local/bin/uv              # 0.9.18
├── /usr/local/bin/entrypoint-mbridge.sh   # 挂载优先逻辑
├── 系统级 (NGC 自带,继承)
│   ├── torch 2.8.0a0+nv25.06
│   ├── transformer_engine 2.4.0
│   ├── flash_attn 2.7.4
│   ├── APEX cuda ext (fused_weight_gradient_mlp_cuda)
│   ├── cuDNN 9.10.2
│   └── nvcc 12.9
└── ENTRYPOINT: entrypoint-mbridge.sh
    CMD: bash
```
