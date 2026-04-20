#!/usr/bin/env bash
# Serve Qwen3-8B with all pair-level LoRA MoE adapters enabled.
#
# Usage:
#   bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
#   MODEL_PATH=/path/to/Qwen3-8B MANIFEST=/path/to/moe_router_manifest.json bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lowres-serve}"
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"

MODEL_BASE_NAME="Qwen3-8B"
MODELS_DIR="${ROOT}/models"
MANIFEST="${MANIFEST:-${ROOT}/training/moe_router_manifest.json}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${MODELS_DIR}/Qwen3-8B_latest" ]]; then
    MODEL_PATH="${MODELS_DIR}/Qwen3-8B_latest"
  elif [[ -d "${MODELS_DIR}" ]]; then
    MODEL_PATH="$(ls -1dt "${MODELS_DIR}/${MODEL_BASE_NAME}_"* 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
未设置 MODEL_PATH，且未在 ${MODELS_DIR}/ 下找到 ${MODEL_BASE_NAME}_latest 或 ${MODEL_BASE_NAME}_<时间戳> 目录。
请先下载：
  python scripts/download_models_to_models_dir.py --only qwen3_8b
或手动指定：
  MODEL_PATH=/path/to/model bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
EOF
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  cat >&2 <<EOF
未找到 MoE router manifest: ${MANIFEST}
请先生成：
  python scripts/moe/generate_pair_expert_assets.py
或指定：
  MANIFEST=/path/to/moe_router_manifest.json bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
EOF
  exit 1
fi

PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LORAS="${MAX_LORAS:-20}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b-moe-base}"

mapfile -t LORA_MODULES < <(
  python - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
data = json.loads(manifest.read_text(encoding="utf-8"))
experts = data.get("experts") or []
if not experts:
    raise SystemExit(f"No experts found in {manifest}")
for item in experts:
    name = str(item.get("adapter_name") or "").strip()
    path = str(item.get("adapter_path") or "").strip()
    if not name or not path:
        raise SystemExit(f"Bad expert item: {item!r}")
    print(f"{name}={path}")
PY
)

missing=0
for mod in "${LORA_MODULES[@]}"; do
  path="${mod#*=}"
  if [[ ! -d "${path}" ]]; then
    echo "警告：LoRA adapter 目录不存在：${path}" >&2
    missing=1
  fi
done
if [[ "${STRICT_LORA_PATHS:-1}" == "1" && "${missing}" == "1" ]]; then
  echo "存在缺失的 LoRA adapter 目录；如需忽略检查，设置 STRICT_LORA_PATHS=0。" >&2
  exit 1
fi

echo "Serving base model: ${MODEL_PATH}" >&2
echo "Manifest: ${MANIFEST}" >&2
echo "LoRA modules: ${#LORA_MODULES[@]}" >&2

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-lora \
  --max-loras "${MAX_LORAS}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --lora-modules "${LORA_MODULES[@]}" \
  --default-chat-template-kwargs '{"enable_thinking": false}'
