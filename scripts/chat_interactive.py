#!/usr/bin/env python3
"""
对已部署模型做交互式对话测试（默认走 vLLM 的 OpenAI 兼容接口，与 run_eval.py 一致）。

前置：先在本机启动 vLLM，例如：
  bash scripts/serve/serve_vllm_qwen3.sh
  # 或合并后的权重：vllm serve /path/to/merged --served-model-name ...

用法：
  conda activate lowres
  python scripts/chat_interactive.py
  python scripts/chat_interactive.py --model qwen3-4b --model-family qwen3

机器翻译式单轮测试（与评测 prompt 风格一致）：
  python scripts/chat_interactive.py --src-lang eng_Latn --tgt-lang zho_Hans --model-family qwen3

命令行内：
  /help   帮助
  /reset  清空多轮历史
  /exit   退出
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from openai import OpenAI


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    fam = (model_family or "").lower()
    if fam in ("qwen3", "qwen", "qwen3-4b"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def mt_user_content(src_lang: str, tgt_lang: str, text: str) -> str:
    return (
        f"Translate from {src_lang} to {tgt_lang}. Output only the translation, no explanations.\n\n{text}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="交互式测试已部署模型（OpenAI 兼容 API）")
    ap.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
        help="含 /v1 的 API base",
    )
    ap.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    ap.add_argument(
        "--model",
        type=str,
        default=os.environ.get("SERVED_MODEL_NAME", "base"),
        help="与 vLLM --served-model-name 一致",
    )
    ap.add_argument(
        "--model-family",
        type=str,
        default=os.environ.get("EVAL_MODEL_FAMILY", "generic"),
        help="qwen3 时关闭 enable_thinking（与评测一致）",
    )
    ap.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="对话模式下的 system 提示（翻译模式会覆盖为 MT system）",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument(
        "--src-lang",
        type=str,
        default=None,
        help="若与 --tgt-lang 同时指定，则进入翻译测试模式（每条用户输入当作待译文本）",
    )
    ap.add_argument("--tgt-lang", type=str, default=None)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    extra = build_extra_body(str(args.model_family))

    mt_mode = bool(args.src_lang and args.tgt_lang)
    messages: list[dict[str, str]] = []

    if mt_mode:
        messages = [
            {
                "role": "system",
                "content": "You are a professional machine translation engine.",
            }
        ]
        print(
            f"翻译模式: {args.src_lang} -> {args.tgt_lang}（输入原文，回车发送；/exit 退出）\n",
            file=sys.stderr,
        )
    else:
        messages = [{"role": "system", "content": str(args.system)}]
        print("对话模式（/reset 清空历史，/exit 退出）\n", file=sys.stderr)

    while True:
        try:
            line = input("> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        s = line.strip()
        if not s:
            continue
        if s in ("/exit", "/quit", ":q"):
            break
        if s == "/reset":
            if mt_mode:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a professional machine translation engine.",
                    }
                ]
            else:
                messages = [{"role": "system", "content": str(args.system)}]
            print("(已清空上下文)", file=sys.stderr)
            continue
        if s == "/help":
            print(
                "/help /reset /exit；翻译模式需同时提供 --src-lang --tgt-lang",
                file=sys.stderr,
            )
            continue

        if mt_mode:
            user_content = mt_user_content(str(args.src_lang), str(args.tgt_lang), s)
            kwargs: dict[str, Any] = {
                "model": args.model,
                "messages": messages
                + [{"role": "user", "content": user_content}],
                "max_tokens": int(args.max_tokens),
                "temperature": float(args.temperature),
            }
            if extra:
                kwargs["extra_body"] = extra
            resp = client.chat.completions.create(**kwargs)
            out = (resp.choices[0].message.content or "").strip()
            print(out)
            # 翻译模式默认单轮：不把历史拼进下一轮（避免噪声）；仅保留 system
            continue

        messages.append({"role": "user", "content": s})
        kwargs = {
            "model": args.model,
            "messages": messages,
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
        }
        if extra:
            kwargs["extra_body"] = extra
        resp = client.chat.completions.create(**kwargs)
        assistant = (resp.choices[0].message.content or "").strip()
        print(assistant)
        messages.append({"role": "assistant", "content": assistant})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
