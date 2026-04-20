#!/usr/bin/env bash
# Serve Qwen3-8B with the single all-directions SFT LoRA adapter enabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lowres-serve}"
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_BASE_NAME="Qwen3-8B"
MODELS_DIR="${ROOT}/models"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${MODELS_DIR}/Qwen3-8B_latest" ]]; then
    MODEL_PATH="${MODELS_DIR}/Qwen3-8B_latest"
  elif [[ -d "${MODELS_DIR}" ]]; then
    MODEL_PATH="$(ls -1dt "${MODELS_DIR}/${MODEL_BASE_NAME}_"* 2>/dev/null | head -n 1 || true)"
  fi
fi

ADAPTER_NAME="${ADAPTER_NAME:-qwen3_8b_all_directions_sft}"
ADAPTER_PATH="${ADAPTER_PATH:-${ROOT}/models/Qwen3-8B_all_directions_sft_lora}"

if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH. Set MODEL_PATH=/path/to/Qwen3-8B." >&2
  exit 1
fi
if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "Missing ADAPTER_PATH: ${ADAPTER_PATH}" >&2
  echo "Set ADAPTER_PATH to the trained LoRA adapter directory." >&2
  exit 1
fi

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b-base}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
  --default-chat-template-kwargs '{"enable_thinking": false}'
