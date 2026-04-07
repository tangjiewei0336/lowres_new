#!/usr/bin/env bash
# ccMatrix 微调后 checkpoint 评估：将合并/导出后的 HuggingFace 格式目录传给 MODEL_PATH 并用 vLLM 加载。
# 本脚本仅调用 OpenAI 兼容评估客户端；vLLM 需单独启动并指向微调产物。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=activate_venv.sh
source "${SCRIPT_DIR}/activate_venv.sh"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

FINETUNED_DIR="${FINETUNED_DIR:?请设置 FINETUNED_DIR，例如 models/Qwen3-4B_20250407_153012/merged 或导出目录}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-finetuned-mt}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-after_ccmatrix}"
export EVAL_MODEL_FAMILY="${EVAL_MODEL_FAMILY:-generic}"

echo "请确保已用 vLLM 加载: ${FINETUNED_DIR}，且 --served-model-name 与 SERVED_MODEL_NAME=${SERVED_MODEL_NAME} 一致" >&2

exec python scripts/run_eval.py \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --base-url "${OPENAI_API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --model-family "${EVAL_MODEL_FAMILY}" \
  --model-tag "${EVAL_MODEL_TAG}"
