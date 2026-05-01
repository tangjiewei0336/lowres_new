#!/usr/bin/env bash
# 一键检查各 OpenAI 兼容 API 的连通性（鉴权 + 模型可调用性）。
# 测试会发送一个真实的英文 -> 简体中文翻译请求，尽量与评估阶段的提示格式保持一致。
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
import re
import sys
import json
from openai import OpenAI

provider = os.environ["PROVIDER"]
api_key = os.environ["API_KEY"]
base_url = os.environ.get("BASE_URL", "")
model = os.environ["MODEL"]
timeout = float(os.environ.get("TIMEOUT_SECONDS", "20"))

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)

def flatten_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    out.append(text)
                else:
                    inner = item.get("content")
                    if isinstance(inner, str):
                        out.append(inner)
        return "".join(out)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
    return str(content)

def strip_think_tags(text):
    return _THINK_TAG_RE.sub("", text or "").strip()

def build_extra_body(provider_name):
    p = provider_name.lower()
    if p in ("zhipu", "deepseek"):
        return {"thinking": {"type": "disabled"}}
    if p == "qwen":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if p == "minimax":
        return {"reasoning_split": True}
    return None

def message_summary(message):
    raw_content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning", None)
    reasoning_details = getattr(message, "reasoning_details", None)
    return {
        "content_type": type(raw_content).__name__ if raw_content is not None else None,
        "content_preview": flatten_content(raw_content)[:300],
        "content_without_think_preview": strip_think_tags(flatten_content(raw_content))[:300],
        "reasoning_preview": flatten_content(reasoning)[:300],
        "reasoning_details_preview": flatten_content(reasoning_details)[:300],
        "finish_reason": getattr(getattr(resp, "choices", [None])[0], "finish_reason", None),
    }

kwargs = {"api_key": api_key, "timeout": timeout}
if base_url:
    kwargs["base_url"] = base_url

try:
    client = OpenAI(**kwargs)
    source_text = "The weather is nice today, so we decided to walk to the library after lunch."
    user_content = (
        "Translate the following English text into Simplified Chinese. "
        "Output only the translation.\n\n"
        f"{source_text}"
    )
    extra_body = build_extra_body(provider)
    if extra_body:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional machine translation engine.",
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=128,
            temperature=0,
            extra_body=extra_body,
        )
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional machine translation engine.",
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=128,
            temperature=0,
        )
    content = ""
    summary = {}
    if resp.choices and resp.choices[0].message:
        summary = message_summary(resp.choices[0].message)
        content = strip_think_tags(flatten_content(resp.choices[0].message.content))
    if not content:
        raise RuntimeError(f"empty translation response; debug={json.dumps(summary, ensure_ascii=False)}")
    print(f"{provider} OK; reply={content[:60]!r}; debug={json.dumps(summary, ensure_ascii=False)}")
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

test_provider "zhipu" "${ZHIPU_API_KEY:-}" "${ZHIPU_API_BASE:-https://open.bigmodel.cn/api/paas/v4}" "${ZHIPU_MODEL:-glm-4.7}"
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
