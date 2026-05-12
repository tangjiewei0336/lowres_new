from __future__ import annotations

import argparse
import json

from tool_registry import build_local_dispatcher, env_lexicon_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Local dictionary tool dispatcher (no MCP).")
    ap.add_argument(
        "--tool",
        required=True,
        choices=[
            "list_dictionary_pairs",
            "lookup_dictionary",
        ],
    )
    ap.add_argument("--term", default="nước")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    dispatcher = build_local_dispatcher(
        lexicon_dir=env_lexicon_dir(),
    )

    if args.tool == "list_dictionary_pairs":
        out = dispatcher["list_dictionary_pairs"]()
    elif args.tool == "lookup_dictionary":
        out = dispatcher["lookup_dictionary"](
            term=args.term,
            top_k=args.top_k,
            offset=args.offset,
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
