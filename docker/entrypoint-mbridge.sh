#!/usr/bin/env bash
# Entrypoint for the qwen35-mbridge environment image.
#
# Responsibilities (environment only — NO megatron-bridge business logic):
#   1. Ensure /opt/venv-mbridge/bin is first on PATH.
#   2. Exec the user command.
#
# All megatron-bridge / megatron-core setup (uv sync, editable install, etc.)
# is done by the operator AFTER the container starts, from the /mnt source tree:
#
#   cd /mnt/tidal-alsh01/dataset/redone/hade/dd/Megatron-Bridge
#   bash scripts/install_runtime_deps.sh
#   RANK=<N> MASTER_PORT=23456 bash /mnt/tidal-alsh01/dataset/redone/hade/dd/start.sh

set -e

VENV_DIR="${MBRIDGE_VENV_DIR:-/opt/venv-mbridge}"

# Ensure venv is first on PATH (Dockerfile ENV already sets this; safety net).
export PATH="${VENV_DIR}/bin:${PATH}"

# Ensure timezone is Asia/Shanghai (UTC+8).
# The Dockerfile sets TZ=Asia/Shanghai and symlinks /etc/localtime; this export
# makes TZ visible to Python's datetime / logging even if the host overrides it
# via a bind-mounted /etc/localtime.
export TZ="${TZ:-Asia/Shanghai}"

exec "$@"
