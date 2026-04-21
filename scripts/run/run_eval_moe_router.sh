#!/usr/bin/env bash
# Pair-level LoRA MoE 整体评测：请先手动启动对应 vLLM 服务，再执行本脚本。
# 使用 conda 环境 lowres。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

# vLLM OpenAI 兼容地址（与 serve 脚本端口一致）
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# MoE router manifest；如已按 COMET 选好 checkpoint，可改成 best_by_comet 版本
export MOE_ROUTER_MANIFEST="${MOE_ROUTER_MANIFEST:-training/moe_router_manifest.json}"

# 评测标签
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-moe_router_qwen3_8b}"
export EVAL_MODEL_FAMILY="${EVAL_MODEL_FAMILY:-qwen3}"

if [[ ! -f "${MOE_ROUTER_MANIFEST}" ]]; then
  echo "未找到 MOE_ROUTER_MANIFEST: ${MOE_ROUTER_MANIFEST}" >&2
  echo "可设置为 training/moe_router_manifest.json 或 training/moe_router_manifest.best_by_comet.json" >&2
  exit 1
fi

exec python scripts/run/run_eval_moe_router.py \
  --base-url "${OPENAI_API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --router-manifest "${MOE_ROUTER_MANIFEST}" \
  --model-family "${EVAL_MODEL_FAMILY}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
