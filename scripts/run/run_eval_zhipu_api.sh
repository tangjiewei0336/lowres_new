#!/usr/bin/env bash
# Zhipu API FLORES 评测。使用 conda 环境 lowres。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

export ZHIPU_API_KEY="${ZHIPU_API_KEY:-}"
export ZHIPU_MODEL="${ZHIPU_MODEL:-glm-4.7}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-zhipu_glm_4_7_flores}"

if [[ -z "${ZHIPU_API_KEY}" ]]; then
  echo "请设置 ZHIPU_API_KEY" >&2
  exit 1
fi

exec python scripts/run/run_eval_zhipu_api.py \
  --api-key "${ZHIPU_API_KEY}" \
  --model "${ZHIPU_MODEL}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
