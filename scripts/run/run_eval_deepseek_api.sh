#!/usr/bin/env bash
# DeepSeek API FLORES 评测。使用 conda 环境 lowres。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-deepseek_chat_flores}"

if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  echo "请设置 DEEPSEEK_API_KEY" >&2
  exit 1
fi

exec python scripts/run/run_eval_deepseek_api.py \
  --api-key "${DEEPSEEK_API_KEY}" \
  --base-url "${DEEPSEEK_API_BASE}" \
  --model "${DEEPSEEK_MODEL}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
