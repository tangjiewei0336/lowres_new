#!/usr/bin/env bash
# 由其他 shell 脚本 source：强制使用仓库内 .venv
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
  export VIRTUAL_ENV="$ROOT/.venv"
elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
  export PATH="${VIRTUAL_ENV}/bin:$PATH"
else
  echo "错误: 未找到虚拟环境。请在仓库根目录执行:" >&2
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
