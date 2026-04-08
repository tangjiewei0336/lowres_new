#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507（Hub: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507）
# 仅非思考模式；可与基座评测脚本共用 run_eval.py 的 qwen3 族逻辑。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lowres-serve}"
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_BASE_NAME="Qwen3-4B-Instruct-2507"
MODELS_DIR="${ROOT}/models"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${MODELS_DIR}" ]]; then
    MODEL_PATH="$(ls -1dt "${MODELS_DIR}/${MODEL_BASE_NAME}_"* 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
未设置 MODEL_PATH，且未在 ${MODELS_DIR}/ 下找到 ${MODEL_BASE_NAME}_<时间戳> 目录。
请先下载：
  python scripts/download_models_to_models_dir.py --only qwen3_4b_instruct_2507
或手动指定：
  MODEL_PATH=/path/to/model bash scripts/serve/serve_vllm_qwen3_instruct_2507.sh
EOF
  exit 1
fi

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name qwen3-4b-instruct-2507 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --dtype auto
