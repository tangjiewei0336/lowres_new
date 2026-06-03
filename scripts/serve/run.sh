#!/usr/bin/env bash
# 统一入口：按「配置名」调用同目录下 serve_vllm_*.sh（vLLM）。
# 用法: bash scripts/serve/run.sh <配置名>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
用法: bash scripts/serve/run.sh <配置名>

在仓库根目录执行；需已安装 vLLM 与项目依赖（uv / .venv）。

环境变量（与各 serve_vllm_*.sh 一致）:
  MODEL_PATH              本地权重目录（未设则自动选 models/<展示名>_ 下最新时间戳目录）
  PORT                    默认 8000
  TENSOR_PARALLEL_SIZE    默认 1
  MAX_MODEL_LEN           各模型默认值不同，见对应 .sh

配置名:
  qwen3, qwen3-4b           Qwen3-4B 基座（关闭 thinking）
  qwen3-8b                  Qwen3-8B 基座（关闭 thinking）
  qwen3-8b-tool             Qwen3-8B 基座（关闭 thinking，启用 tool calling）
  qwen3-8b-moe              Qwen3-8B + pair-level LoRA MoE adapters
  qwen3-8b-single-sft       Qwen3-8B + all-directions single SFT LoRA
  qwen3-instruct-2507       Qwen3-4B-Instruct-2507
  qwen3.5-27b-instruct      Qwen3.5-27B（关闭 thinking，Instruct alias）
  smollm3                   SmolLM3-3B
  hunyuan-mt, hunyuan       HY-MT1.5-1.8B

示例:
  bash scripts/serve/run.sh qwen3-4b
  bash scripts/serve/run.sh qwen3-8b
  bash scripts/serve/run.sh qwen3-8b-tool
  bash scripts/serve/run.sh qwen3-8b-moe
  bash scripts/serve/run.sh qwen3-8b-single-sft
  TENSOR_PARALLEL_SIZE=4 bash scripts/serve/run.sh qwen3.5-27b-instruct
  MODEL_PATH=/path/to/merged bash scripts/serve/run.sh qwen3-instruct-2507
EOF
}

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  usage
  [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && exit 0
  exit 1
fi

case "${1}" in
  qwen3|qwen3-4b)
    exec bash "${SCRIPT_DIR}/serve_vllm_qwen3.sh"
    ;;
  qwen3-8b|qwen38b)
    exec bash "${SCRIPT_DIR}/serve_vllm_qwen3_8b.sh"
    ;;
  qwen3-8b-tool|qwen3-8b-tool-calling|qwen38b-tool|qwen38b-tool-calling)
    exec bash "${SCRIPT_DIR}/serve_vllm_qwen3_8b_tool_calling.sh"
    ;;
  qwen3-8b-moe|qwen38b-moe|moe-qwen3-8b)
    exec bash "${SCRIPT_DIR}/serve_vllm_qwen3_8b_moe_lora.sh"
    ;;
  qwen3-8b-single-sft|qwen38b-single-sft|single-sft-qwen3-8b)
    exec bash "${SCRIPT_DIR}/serve_vllm_qwen3_8b_single_sft_lora.sh"
    ;;
  qwen3-instruct-2507|qwen3_4b_instruct_2507)
    exec bash "${SCRIPT_DIR}/serve_vllm_qwen3_instruct_2507.sh"
    ;;
  qwen3.5-27b-instruct|qwen3_5_27b_instruct|qwen35-27b-instruct|qwen35_27b_instruct)
    exec bash "${SCRIPT_DIR}/serve_vllm_qwen3_5_27b_instruct.sh"
    ;;
  smollm3|smollm)
    exec bash "${SCRIPT_DIR}/serve_vllm_smollm3.sh"
    ;;
  hunyuan-mt|hunyuan|hy-mt)
    exec bash "${SCRIPT_DIR}/serve_vllm_hunyuan_mt.sh"
    ;;
  *)
    echo "未知配置名: ${1}" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
