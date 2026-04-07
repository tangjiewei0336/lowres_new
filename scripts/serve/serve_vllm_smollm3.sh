#!/usr/bin/env bash
# 使用仓库 .venv；vLLM 需单独安装: pip install vllm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_venv.sh
source "${SCRIPT_DIR}/../activate_venv.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

# 本地模型目录：请先用 ModelScope 下载后指向该路径，例如:
#   python -c "from modelscope import snapshot_download; print(snapshot_download('HuggingFaceTB/SmolLM3-3B'))"
MODEL_PATH="${MODEL_PATH:?请设置 MODEL_PATH 为 SmolLM3-3B 本地目录（仅通过 ModelScope 下载）}"

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name smollm3-3b \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --dtype auto
