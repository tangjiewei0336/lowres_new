#!/usr/bin/env bash
# 智谱 GLM-4.7：调用 OpenAI 兼容接口批量生成 hypotheses.jsonl。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
# shellcheck source=./load_dotenv.sh
source "${SCRIPT_DIR}/load_dotenv.sh" "${ROOT}"

export ZHIPU_API_KEY="${ZHIPU_API_KEY:-}"
export ZHIPU_API_BASE="${ZHIPU_API_BASE:-https://open.bigmodel.cn/api/paas/v4}"
export ZHIPU_MODEL="${ZHIPU_MODEL:-glm-4.7}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-zhipu_glm_4_7_hyp}"
export EVAL_MODEL_FAMILY="${EVAL_MODEL_FAMILY:-zhipu}"

if [[ -z "${ZHIPU_API_KEY}" ]]; then
  echo "请设置 ZHIPU_API_KEY" >&2
  exit 1
fi

exec python scripts/run/generate/generate_hypotheses_openai.py \
  --api-key "${ZHIPU_API_KEY}" \
  --base-url "${ZHIPU_API_BASE}" \
  --model "${ZHIPU_MODEL}" \
  --model-family "${EVAL_MODEL_FAMILY}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
