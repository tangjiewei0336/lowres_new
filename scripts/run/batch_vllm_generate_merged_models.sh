#!/usr/bin/env bash
# 依次对每个合并模型目录启动 vLLM，并仅生成 hypotheses（不跑 BLEU/COMET）。
# 请在已安装 vllm 与 run_eval 依赖的环境中执行（例如 uv run 或激活 .venv）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

python "${SCRIPT_DIR}/batch_vllm_generate_merged_models.py" \
  --models-dir "${ROOT}/models" \
  --name-glob "${NAME_GLOB:-Qwen3-8B_aug_merged_*}" \
  --output-parent-dir "${OUTPUT_PARENT_DIR:?设置 OUTPUT_PARENT_DIR，例如 eval_multilingual/qwen3_8b_aug_sweep}" \
  --model-family "${EVAL_MODEL_FAMILY:-qwen3}" \
  --served-model-name "${SERVED_MODEL_NAME:-merged}" \
  --port "${PORT:-8000}" \
  "$@"
