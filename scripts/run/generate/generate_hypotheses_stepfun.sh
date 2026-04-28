#!/usr/bin/env bash
# 阶跃星辰 StepFun：调用 OpenAI 兼容接口批量生成 hypotheses.jsonl。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
# shellcheck source=./load_dotenv.sh
source "${SCRIPT_DIR}/load_dotenv.sh" "${ROOT}"

export STEPFUN_API_KEY="${STEPFUN_API_KEY:-}"
export STEPFUN_API_BASE="${STEPFUN_API_BASE:-https://api.stepfun.com/v1}"
export STEPFUN_MODEL="${STEPFUN_MODEL:-step-1-8k}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-stepfun_hyp}"
export EVAL_MODEL_FAMILY="${EVAL_MODEL_FAMILY:-generic}"

if [[ -z "${STEPFUN_API_KEY}" ]]; then
  echo "请设置 STEPFUN_API_KEY" >&2
  exit 1
fi

exec python scripts/run/generate/generate_hypotheses_openai.py \
  --api-key "${STEPFUN_API_KEY}" \
  --base-url "${STEPFUN_API_BASE}" \
  --model "${STEPFUN_MODEL}" \
  --model-family "${EVAL_MODEL_FAMILY}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
