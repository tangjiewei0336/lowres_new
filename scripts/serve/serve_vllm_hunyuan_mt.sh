#!/usr/bin/env bash
# HY-MT1.5 可能非标准 decoder-only，若 vLLM 报错请参考 README 中 Transformers 推理备选方案。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_venv.sh
source "${SCRIPT_DIR}/../activate_venv.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_PATH="${MODEL_PATH:?请设置 MODEL_PATH 为 HY-MT1.5-1.8B 本地目录（仅通过 ModelScope 下载）}"

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-4096}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name hunyuan-mt-1.8b \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --dtype auto
