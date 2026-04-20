#!/usr/bin/env python3
"""
Translate with a pair-level LoRA MoE deployment served by vLLM.

The script reads training/moe_router_manifest.json, selects the adapter model
for --src-lang/--tgt-lang, and calls the vLLM OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from flores_lang_zh import english_translation_instruction  # noqa: E402


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    experts = data.get("experts")
    if not isinstance(experts, list) or not experts:
        raise SystemExit(f"No experts found in manifest: {path}")
    return data


def build_router(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    router: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest.get("experts", []):
        if not isinstance(item, dict):
            continue
        src = str(item.get("src_lang", "")).strip()
        tgt = str(item.get("tgt_lang", "")).strip()
        name = str(item.get("adapter_name", "")).strip()
        if not src or not tgt or not name:
            continue
        router[(src, tgt)] = item
    if not router:
        raise SystemExit("Manifest has no usable expert routes.")
    return router


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    fam = (model_family or "").lower()
    if fam in ("qwen3", "qwen", "qwen3-4b", "qwen3-8b", "qwen3.5"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def read_text_arg(text: str | None, text_file: Path | None) -> str:
    if text is not None:
        return text
    if text_file is not None:
        return text_file.read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide --text, --text-file, or pipe text on stdin.")


def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Route a translation request to the matching LoRA MoE expert.")
    ap.add_argument("--manifest", type=Path, default=root / "training/moe_router_manifest.json")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--src-lang", required=False)
    ap.add_argument("--tgt-lang", required=False)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", type=Path, default=None)
    ap.add_argument("--model-family", default=os.environ.get("EVAL_MODEL_FAMILY", "qwen3"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--list-pairs", action="store_true")
    ap.add_argument("--print-model", action="store_true", help="Print selected adapter model name to stderr.")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    router = build_router(manifest)

    if args.list_pairs:
        for src, tgt in sorted(router):
            item = router[(src, tgt)]
            print(f"{src}\t{tgt}\t{item['adapter_name']}")
        return 0

    if not args.src_lang or not args.tgt_lang:
        raise SystemExit("--src-lang and --tgt-lang are required unless --list-pairs is used.")
    route = router.get((args.src_lang, args.tgt_lang))
    if not route:
        available = ", ".join(f"{s}->{t}" for s, t in sorted(router))
        raise SystemExit(f"No LoRA expert for {args.src_lang}->{args.tgt_lang}. Available: {available}")

    model_name = str(route["adapter_name"])
    text = read_text_arg(args.text, args.text_file)
    instruction = english_translation_instruction(args.src_lang, args.tgt_lang)
    messages = [
        {"role": "system", "content": "You are a professional machine translation engine."},
        {"role": "user", "content": f"{instruction}\n\n{text}"},
    ]

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    extra = build_extra_body(args.model_family)
    if extra:
        kwargs["extra_body"] = extra

    if args.print_model:
        print(f"model={model_name}", file=sys.stderr)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    resp = client.chat.completions.create(**kwargs)
    print((resp.choices[0].message.content or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
