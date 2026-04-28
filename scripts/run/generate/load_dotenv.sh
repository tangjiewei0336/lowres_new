#!/usr/bin/env bash
# 从仓库根目录加载 .env（如果存在），并导出其中变量。

ROOT_DIR="${1:-}"
if [[ -z "${ROOT_DIR}" ]]; then
  echo "load_dotenv.sh 需要传入仓库根目录路径" >&2
  return 1
fi

DOTENV_PATH="${ROOT_DIR}/.env"
if [[ -f "${DOTENV_PATH}" ]]; then
  # 允许 .env 里引用未定义变量，避免在 set -u 下 source 失败。
  set +u
  set -a
  # shellcheck disable=SC1090
  source "${DOTENV_PATH}"
  set +a
  set -u
fi
