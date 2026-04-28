#!/usr/bin/env bash
# MiniMax：调用 OpenAI 兼容接口批量生成 hypotheses.jsonl。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
# shellcheck source=./load_dotenv.sh
source "${SCRIPT_DIR}/load_dotenv.sh" "${ROOT}"

export MINIMAX_API_KEY="${MINIMAX_API_KEY:-}"
export MINIMAX_API_BASE="${MINIMAX_API_BASE:-https://api.minimax.chat/v1}"
export MINIMAX_MODEL="${MINIMAX_MODEL:-MiniMax-Text-01}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-minimax_hyp}"
export EVAL_MODEL_FAMILY="${EVAL_MODEL_FAMILY:-generic}"

if [[ -z "${MINIMAX_API_KEY}" ]]; then
  echo "请设置 MINIMAX_API_KEY" >&2
  exit 1
fi

exec python scripts/run/generate/generate_hypotheses_openai.py \
  --api-key "${MINIMAX_API_KEY}" \
  --base-url "${MINIMAX_API_BASE}" \
  --model "${MINIMAX_MODEL}" \
  --model-family "${EVAL_MODEL_FAMILY}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
