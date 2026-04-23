from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from agent_runtime import LangGraphDictionaryAgentRuntime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


async def main_async() -> int:
    ap = argparse.ArgumentParser(description="LangGraph agent demo that automatically calls the local MCP dictionary server.")
    ap.add_argument("--text", required=True, help="Source text to translate.")
    ap.add_argument("--src-lang", required=True)
    ap.add_argument("--tgt-lang", required=True)
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "qwen3-8b"))
    ap.add_argument("--model-family", default=os.environ.get("AGENT_MODEL_FAMILY", "qwen3"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument(
        "--lexicon-dir",
        type=Path,
        default=repo_root() / "training" / "data" / "dictionaries" / "moe_lexicon",
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    async with LangGraphDictionaryAgentRuntime(
        model=args.model,
        model_family=args.model_family,
        base_url=args.base_url,
        api_key=args.api_key,
        lexicon_dir=args.lexicon_dir,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
    ) as runtime:
        print(
            await runtime.translate(
                text=args.text,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
            )
        )
        return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
