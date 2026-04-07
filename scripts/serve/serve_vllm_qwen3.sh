#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_venv.sh
source "${SCRIPT_DIR}/../activate_venv.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_PATH="${MODEL_PATH:?请设置 MODEL_PATH 为 Qwen3-4B 本地目录（仅通过 ModelScope 下载）}"

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --dtype auto
