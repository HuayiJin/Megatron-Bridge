#!/usr/bin/env bash
# Build-host preflight check for Dockerfile.qwen35.
# Run this on the dedicated build machine BEFORE `docker build`.
# Exits non-zero if any hard requirement is missing.
#
# Usage:
#   bash scripts/check_build_host.sh

set -uo pipefail

# Colours
red()    { printf "\033[31m%s\033[0m" "$*"; }
green()  { printf "\033[32m%s\033[0m" "$*"; }
yellow() { printf "\033[33m%s\033[0m" "$*"; }

PASS=0
FAIL=0
WARN=0

ok()    { echo "  $(green '[OK]')   $*"; PASS=$((PASS+1)); }
fail()  { echo "  $(red   '[FAIL]') $*"; FAIL=$((FAIL+1)); }
warn()  { echo "  $(yellow '[WARN]') $*"; WARN=$((WARN+1)); }

section() { echo; echo "=== $* ==="; }

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.06-py3}"
MIN_DOCKER_DISK_GB="${MIN_DOCKER_DISK_GB:-80}"
MIN_CTX_DISK_GB="${MIN_CTX_DISK_GB:-5}"
MIN_RAM_GB="${MIN_RAM_GB:-16}"

# ---------------------------------------------------------------------------
section "1. OS / arch"
# ---------------------------------------------------------------------------
case "$(uname -s)" in
    Linux) ok "OS = Linux" ;;
    *)     fail "OS = $(uname -s) (only Linux is supported)" ;;
esac
case "$(uname -m)" in
    x86_64) ok "arch = x86_64" ;;
    *)      fail "arch = $(uname -m) (NGC images are x86_64 only)" ;;
esac

if [ -f /etc/os-release ]; then
    . /etc/os-release
    ok "distro = ${PRETTY_NAME:-$ID $VERSION_ID}"
fi

# ---------------------------------------------------------------------------
section "2. Docker"
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    fail "docker command not found — install via https://docs.docker.com/engine/install/"
else
    DOCKER_VER="$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo unknown)"
    ok "docker version = ${DOCKER_VER}"

    if docker info >/dev/null 2>&1; then
        ok "docker daemon reachable"
        DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)"
        ok "docker root dir = ${DOCKER_ROOT}"
    else
        fail "docker daemon not reachable (try: sudo systemctl start docker, or add user to docker group)"
        DOCKER_ROOT=""
    fi
fi

if command -v docker >/dev/null 2>&1; then
    if docker buildx version >/dev/null 2>&1; then
        ok "docker buildx available ($(docker buildx version | head -1))"
    else
        warn "docker buildx not found — Dockerfile uses BuildKit syntax (heredoc); export DOCKER_BUILDKIT=1 or install buildx"
    fi
fi

# Check user is in docker group (or root)
if [ "$(id -u)" -eq 0 ]; then
    ok "running as root"
elif id -nG "$USER" 2>/dev/null | grep -qw docker; then
    ok "user '$USER' in docker group"
else
    warn "user '$USER' NOT in docker group; you may need 'sudo docker ...'"
fi

# ---------------------------------------------------------------------------
section "3. Disk space"
# ---------------------------------------------------------------------------
if [ -n "${DOCKER_ROOT:-}" ] && [ -d "${DOCKER_ROOT}" ]; then
    DOCKER_AVAIL_GB="$(df -BG "${DOCKER_ROOT}" | awk 'NR==2 {gsub("G","",$4); print $4}')"
    if [ "${DOCKER_AVAIL_GB:-0}" -ge "${MIN_DOCKER_DISK_GB}" ]; then
        ok "docker root ${DOCKER_ROOT} has ${DOCKER_AVAIL_GB}GB free (>= ${MIN_DOCKER_DISK_GB}GB)"
    else
        fail "docker root ${DOCKER_ROOT} has only ${DOCKER_AVAIL_GB}GB free (need >= ${MIN_DOCKER_DISK_GB}GB)"
    fi
fi

CTX_PARTITION="$(df -BG "${REPO_ROOT}" | awk 'NR==2 {print $1}')"
CTX_AVAIL_GB="$(df -BG "${REPO_ROOT}" | awk 'NR==2 {gsub("G","",$4); print $4}')"
if [ "${CTX_AVAIL_GB:-0}" -ge "${MIN_CTX_DISK_GB}" ]; then
    ok "build context dir ${REPO_ROOT} has ${CTX_AVAIL_GB}GB free (>= ${MIN_CTX_DISK_GB}GB)"
else
    fail "build context dir ${REPO_ROOT} has only ${CTX_AVAIL_GB}GB free (need >= ${MIN_CTX_DISK_GB}GB)"
fi

# ---------------------------------------------------------------------------
section "4. Memory"
# ---------------------------------------------------------------------------
if [ -r /proc/meminfo ]; then
    TOTAL_MEM_KB="$(awk '/MemTotal:/{print $2}' /proc/meminfo)"
    TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
    if [ "${TOTAL_MEM_GB}" -ge "${MIN_RAM_GB}" ]; then
        ok "total RAM = ${TOTAL_MEM_GB}GB (>= ${MIN_RAM_GB}GB)"
    else
        fail "total RAM = ${TOTAL_MEM_GB}GB (need >= ${MIN_RAM_GB}GB; mamba-ssm/causal-conv1d nvcc compile is RAM-hungry)"
    fi
fi

# ---------------------------------------------------------------------------
section "5. Network"
# ---------------------------------------------------------------------------
check_net() {
    local url="$1" name="$2"
    if curl -fsS --connect-timeout 5 -o /dev/null -I "$url" 2>/dev/null; then
        ok "${name} reachable (${url})"
    else
        # HEAD may not be allowed; try a small GET
        if curl -fsS --connect-timeout 5 -o /dev/null -m 10 "$url" 2>/dev/null; then
            ok "${name} reachable (${url})"
        else
            fail "${name} NOT reachable (${url}) — set HTTPS_PROXY if behind a proxy"
        fi
    fi
}
check_net "https://nvcr.io/v2/" "NGC registry (nvcr.io)"
check_net "https://pypi.org/simple/" "PyPI"
check_net "https://pypi.nvidia.com/" "NVIDIA PyPI (extra index)"
check_net "https://github.com" "GitHub (for git submodule fallback)"
check_net "https://astral.sh" "Astral (uv installer)"

# ---------------------------------------------------------------------------
section "6. NGC login (nvcr.io)"
# ---------------------------------------------------------------------------
if [ -f "${HOME}/.docker/config.json" ] && grep -q "nvcr.io" "${HOME}/.docker/config.json" 2>/dev/null; then
    ok "nvcr.io credential present in ~/.docker/config.json"
else
    warn "nvcr.io credential NOT in ~/.docker/config.json — run: docker login nvcr.io"
fi

# Try a HEAD request to manifest to confirm we can pull
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    if docker manifest inspect "${NGC_IMAGE}" >/dev/null 2>&1; then
        ok "${NGC_IMAGE} manifest accessible"
    else
        warn "${NGC_IMAGE} manifest NOT accessible — try: docker pull ${NGC_IMAGE}"
    fi
fi

# ---------------------------------------------------------------------------
section "7. Source tree (Megatron-Bridge + submodule)"
# ---------------------------------------------------------------------------
if [ -f "${REPO_ROOT}/Dockerfile.qwen35" ]; then
    ok "Dockerfile.qwen35 found at ${REPO_ROOT}"
else
    fail "Dockerfile.qwen35 NOT in ${REPO_ROOT} — wrong REPO_ROOT?"
fi

if [ -f "${REPO_ROOT}/3rdparty/Megatron-LM/megatron/core/__init__.py" ]; then
    ok "Megatron-Core submodule populated"
else
    fail "Megatron-Core submodule NOT populated — run: cd ${REPO_ROOT} && git submodule update --init --recursive"
fi

if [ -d "${REPO_ROOT}/.git" ]; then
    ok ".git present (allows host-side submodule operations)"
else
    warn ".git not in ${REPO_ROOT} — submodule updates won't work; ensure you cloned the full repo"
fi

# ---------------------------------------------------------------------------
section "8. Optional but recommended"
# ---------------------------------------------------------------------------
for tool in tmux git curl jq; do
    if command -v "$tool" >/dev/null 2>&1; then
        ok "${tool} installed"
    else
        warn "${tool} not installed (recommended for long-running build / debugging)"
    fi
done

# ---------------------------------------------------------------------------
section "Summary"
# ---------------------------------------------------------------------------
echo "  passes: ${PASS}, warnings: ${WARN}, failures: ${FAIL}"
if [ "${FAIL}" -eq 0 ]; then
    echo
    echo "$(green '[READY]') Build host is ready."
    echo "  Next step:"
    echo "    docker pull ${NGC_IMAGE}"
    echo "    docker build -f Dockerfile.qwen35 -t qwen35-mbridge:cu129 ${REPO_ROOT}"
    exit 0
else
    echo
    echo "$(red '[NOT READY]') Fix the [FAIL] items above and re-run."
    exit 1
fi
