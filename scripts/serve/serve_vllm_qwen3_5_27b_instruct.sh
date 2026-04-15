#!/usr/bin/env bash
# Qwen3.5-27B-Instruct alias（ModelScope: Qwen/Qwen3.5-27B）
# 默认关闭 thinking，作为翻译/指令模型部署。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lowres-serve}"
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_BASE_NAME="Qwen3.5-27B-Instruct"
FALLBACK_MODEL_BASE_NAME="Qwen3.5-27B"
MODELS_DIR="${ROOT}/models"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${MODELS_DIR}" ]]; then
    MODEL_PATH="$(ls -1dt "${MODELS_DIR}/${MODEL_BASE_NAME}_"* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${MODEL_PATH}" ]]; then
      MODEL_PATH="$(ls -1dt "${MODELS_DIR}/${FALLBACK_MODEL_BASE_NAME}_"* 2>/dev/null | head -n 1 || true)"
    fi
  fi
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
未设置 MODEL_PATH，且未在 ${MODELS_DIR}/ 下找到 ${MODEL_BASE_NAME}_<时间戳> 或 ${FALLBACK_MODEL_BASE_NAME}_<时间戳> 目录。
请先下载：
  python scripts/download_models_to_models_dir.py --only qwen3_5_27b_instruct
或手动指定：
  MODEL_PATH=/path/to/model bash scripts/serve/serve_vllm_qwen3_5_27b_instruct.sh
EOF
  exit 1
fi

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-4}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"

args=(
  vllm serve "${MODEL_PATH}"
  --served-model-name qwen3.5-27b-instruct \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --dtype auto \
  --default-chat-template-kwargs '{"enable_thinking": false}'
)

if [[ "${LANGUAGE_MODEL_ONLY}" == "1" || "${LANGUAGE_MODEL_ONLY}" == "true" ]]; then
  args+=(--language-model-only)
fi

exec "${args[@]}"
