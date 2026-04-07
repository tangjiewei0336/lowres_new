#!/usr/bin/env bash
# 基座模型评估：请先手动启动对应 vLLM 服务，再执行本脚本。
# 使用仓库 .venv 中的 Python。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=activate_venv.sh
source "${SCRIPT_DIR}/activate_venv.sh"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# vLLM OpenAI 兼容地址（与 serve 脚本端口一致）
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# 选择要评的基座（取消注释其中一组）:
# --- SmolLM3-3B ---
# export SERVED_MODEL_NAME=smollm3-3b
# export EVAL_MODEL_TAG=baseline_smollm3_3b
# export EVAL_MODEL_FAMILY=generic

# --- Qwen3-4B（关闭思考模式由 run_eval.py 处理）---
# export SERVED_MODEL_NAME=qwen3-4b
# export EVAL_MODEL_TAG=baseline_qwen3_4b
# export EVAL_MODEL_FAMILY=qwen3

# --- Hunyuan HY-MT1.5-1.8B ---
# export SERVED_MODEL_NAME=hunyuan-mt-1.8b
# export EVAL_MODEL_TAG=baseline_hunyuan_mt18b
# export EVAL_MODEL_FAMILY=generic

if [[ -z "${SERVED_MODEL_NAME:-}" || -z "${EVAL_MODEL_TAG:-}" ]]; then
  echo "请编辑本脚本，设置 SERVED_MODEL_NAME 与 EVAL_MODEL_TAG（与 vLLM --served-model-name 一致）" >&2
  exit 1
fi

exec python scripts/run_eval.py \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --base-url "${OPENAI_API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --model-family "${EVAL_MODEL_FAMILY:-generic}" \
  --model-tag "${EVAL_MODEL_TAG}"
