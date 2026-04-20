#!/usr/bin/env bash
# 基座模型评估：请先手动启动对应 vLLM 服务，再执行本脚本。
# 使用 conda 环境 lowres。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=activate_conda_lowres.sh
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

# --- Qwen3-8B（先 download_models_to_models_dir.py --only qwen3_8b，再 scripts/serve/run.sh qwen3-8b）---
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-baseline_qwen3_8b}"
export EVAL_MODEL_FAMILY="${EVAL_MODEL_FAMILY:-qwen3}"

# --- Qwen3.5-27B-Instruct alias（先 download_models_to_models_dir.py --only qwen3_5_27b_instruct，再 scripts/serve/run.sh qwen3.5-27b-instruct）---
# export SERVED_MODEL_NAME=qwen3.5-27b-instruct
# export EVAL_MODEL_TAG=baseline_qwen3_5_27b_instruct
# export EVAL_MODEL_FAMILY=qwen3.5

# --- Qwen3-4B-Instruct-2507（先 download_models_to_models_dir.py --only qwen3_4b_instruct_2507，再 serve_vllm_qwen3_instruct_2507.sh）---
# export SERVED_MODEL_NAME=qwen3-4b-instruct-2507
# export EVAL_MODEL_TAG=baseline_qwen3_4b_instruct_2507
# export EVAL_MODEL_FAMILY=qwen3

# --- Hunyuan HY-MT1.5-1.8B ---
# export SERVED_MODEL_NAME=hunyuan-mt-1.8b
# export EVAL_MODEL_TAG=baseline_hunyuan_mt18b
# export EVAL_MODEL_FAMILY=generic

if [[ -z "${SERVED_MODEL_NAME:-}" || -z "${EVAL_MODEL_TAG:-}" ]]; then
  echo "请编辑本脚本，设置 SERVED_MODEL_NAME 与 EVAL_MODEL_TAG（与 vLLM --served-model-name 一致）" >&2
  exit 1
fi

exec python scripts/run/run_eval.py \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --base-url "${OPENAI_API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --model-family "${EVAL_MODEL_FAMILY:-generic}" \
  --model-tag "${EVAL_MODEL_TAG}"
