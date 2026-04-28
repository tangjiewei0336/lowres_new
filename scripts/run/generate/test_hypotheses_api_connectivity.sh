#!/usr/bin/env bash
# 一键检查各 OpenAI 兼容 API 的连通性（鉴权 + 模型可调用性）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
# shellcheck source=./load_dotenv.sh
source "${SCRIPT_DIR}/load_dotenv.sh" "${ROOT}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"

if ! command -v python >/dev/null 2>&1; then
  echo "未找到 python，请先激活可用环境" >&2
  exit 1
fi

pass_count=0
fail_count=0
skip_count=0

test_provider() {
  local provider="$1"
  local key="${2:-}"
  local base_url="${3:-}"
  local model="${4:-}"

  if [[ -z "${key}" ]]; then
    echo "[SKIP] ${provider}: 缺少 API key"
    skip_count=$((skip_count + 1))
    return 0
  fi

  if [[ -z "${model}" ]]; then
    echo "[FAIL] ${provider}: 缺少 model"
    fail_count=$((fail_count + 1))
    return 0
  fi

  if PROVIDER="${provider}" API_KEY="${key}" BASE_URL="${base_url}" MODEL="${model}" TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
    python - <<'PY'
import os
import sys
from openai import OpenAI

provider = os.environ["PROVIDER"]
api_key = os.environ["API_KEY"]
base_url = os.environ.get("BASE_URL", "")
model = os.environ["MODEL"]
timeout = float(os.environ.get("TIMEOUT_SECONDS", "20"))

kwargs = {"api_key": api_key, "timeout": timeout}
if base_url:
    kwargs["base_url"] = base_url

try:
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=2,
        temperature=0,
    )
    content = ""
    if resp.choices and resp.choices[0].message:
        content = (resp.choices[0].message.content or "").strip()
    print(f"{provider} OK; reply={content[:60]!r}")
except Exception as e:
    print(f"{provider} ERROR: {e}", file=sys.stderr)
    raise
PY
  then
    echo "[PASS] ${provider}"
    pass_count=$((pass_count + 1))
  else
    echo "[FAIL] ${provider}"
    fail_count=$((fail_count + 1))
  fi
}

echo "== 开始连通性检查 =="
echo "timeout: ${TIMEOUT_SECONDS}s"

test_provider "zhipu" "${ZHIPU_API_KEY:-}" "${ZHIPU_API_BASE:-https://open.bigmodel.cn/api/paas/v4}" "${ZHIPU_MODEL:-glm-5.1}"
test_provider "stepfun" "${STEPFUN_API_KEY:-}" "${STEPFUN_API_BASE:-https://api.stepfun.com/v1}" "${STEPFUN_MODEL:-step-1-8k}"
test_provider "minimax" "${MINIMAX_API_KEY:-}" "${MINIMAX_API_BASE:-https://api.minimax.chat/v1}" "${MINIMAX_MODEL:-MiniMax-Text-01}"
test_provider "qwen" "${QWEN_API_KEY:-}" "${QWEN_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}" "${QWEN_MODEL:-qwen-plus}"
test_provider "deepseek" "${DEEPSEEK_API_KEY:-}" "${DEEPSEEK_API_BASE:-https://api.deepseek.com}" "${DEEPSEEK_MODEL:-deepseek-chat}"

echo
echo "== 检查完成 =="
echo "PASS: ${pass_count}"
echo "FAIL: ${fail_count}"
echo "SKIP: ${skip_count}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 2
fi

exit 0
