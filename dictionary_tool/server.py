from __future__ import annotations

import argparse
import json

from tool_registry import build_local_dispatcher, env_lexicon_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Local dictionary tool dispatcher (no MCP).")
    ap.add_argument("--tool", required=True, choices=["list_dictionary_pairs", "lookup_dictionary"])
    ap.add_argument("--src-lang", default="eng_Latn")
    ap.add_argument("--tgt-lang", default="zho_Hans")
    ap.add_argument("--term", default="dictionary")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--fallback-top-k", type=int, default=10)
    args = ap.parse_args()

    dispatcher = build_local_dispatcher(
        lexicon_dir=env_lexicon_dir(),
    )

    if args.tool == "list_dictionary_pairs":
        out = dispatcher["list_dictionary_pairs"]()
    elif args.tool == "lookup_dictionary":
        out = dispatcher["lookup_dictionary"](
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            term=args.term,
            top_k=args.top_k,
            offset=args.offset,
            fallback_top_k=args.fallback_top_k,
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
