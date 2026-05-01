#!/usr/bin/env bash
# 千问（DashScope）：调用 OpenAI 兼容接口批量生成 hypotheses.jsonl。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
# shellcheck source=./load_dotenv.sh
source "${SCRIPT_DIR}/load_dotenv.sh" "${ROOT}"

export QWEN_API_KEY="${QWEN_API_KEY:-}"
export QWEN_API_BASE="${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen-plus}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-qwen_hyp}"
export EVAL_MODEL_FAMILY="${EVAL_MODEL_FAMILY:-qwen}"

if [[ -z "${QWEN_API_KEY}" ]]; then
  echo "请设置 QWEN_API_KEY" >&2
  exit 1
fi

exec python scripts/run/generate/generate_hypotheses_openai.py \
  --api-key "${QWEN_API_KEY}" \
  --base-url "${QWEN_API_BASE}" \
  --model "${QWEN_MODEL}" \
  --model-family "${EVAL_MODEL_FAMILY}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
