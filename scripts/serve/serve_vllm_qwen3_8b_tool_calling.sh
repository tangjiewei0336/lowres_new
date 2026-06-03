#!/usr/bin/env bash
# Qwen3-8B with vLLM auto tool calling enabled.
# Uses Qwen3-Coder tool-call parser and disables thinking by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_BASE_NAME="Qwen3-8B"
MODELS_DIR="${ROOT}/models"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${MODELS_DIR}/Qwen3-8B_latest" ]]; then
    MODEL_PATH="${MODELS_DIR}/Qwen3-8B_latest"
  elif [[ -d "${MODELS_DIR}" ]]; then
    mapfile -t MODEL_MATCHES < <(find "${MODELS_DIR}" -maxdepth 1 -type d -name "${MODEL_BASE_NAME}_*" -print | sort -r)
    if (( ${#MODEL_MATCHES[@]} > 0 )); then
      MODEL_PATH="${MODEL_MATCHES[0]}"
      if (( ${#MODEL_MATCHES[@]} > 1 )); then
        {
          echo "WARNING: 模糊匹配到多个 ${MODEL_BASE_NAME}_* 模型目录，将使用排序后的第一个："
          printf '  %s\n' "${MODEL_MATCHES[@]}"
        } >&2
      fi
    fi
  fi
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
未设置 MODEL_PATH，且未在 ${MODELS_DIR}/ 下找到 ${MODEL_BASE_NAME}_latest 或 ${MODEL_BASE_NAME}_<时间戳> 目录。
请先下载：
  python scripts/download_models_to_models_dir.py --only qwen3_8b
或手动指定：
  MODEL_PATH=/path/to/model bash scripts/serve/serve_vllm_qwen3_8b_tool_calling.sh
EOF
  exit 1
fi

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"

echo "Serving model with tool calling: ${MODEL_PATH}" >&2
echo "Tool call parser: qwen3_coder" >&2

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --dtype auto \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
