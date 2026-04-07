#!/usr/bin/env bash
# HY-MT1.5 可能非标准 decoder-only，若 vLLM 报错请参考 README 中 Transformers 推理备选方案。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lowres-serve}"
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_BASE_NAME="HY-MT1.5-1.8B"
MODELS_DIR="${ROOT}/models"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${MODELS_DIR}" ]]; then
    MODEL_PATH="$(ls -1dt "${MODELS_DIR}/${MODEL_BASE_NAME}_"* 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
未设置 MODEL_PATH，且未在 ${MODELS_DIR}/ 下找到 ${MODEL_BASE_NAME}_<时间戳> 目录。
请先通过 ModelScope 下载到本仓库 models/，或手动指定：
  MODEL_PATH=/path/to/local/model bash scripts/serve/serve_vllm_hunyuan_mt.sh
提示：可用 python scripts/download_models_to_models_dir.py 下载到 models/。
EOF
  exit 1
fi

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
