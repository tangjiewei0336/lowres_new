#!/usr/bin/env python3
"""
Convert normalized dictionary lexicons into LLaMA-Factory Alpaca translation data.

Default input:
  training/data/dictionaries/moe_lexicon/dict_terms_<src>__<tgt>.jsonl

Default output:
  training/data/multilingual/dictionary_moe/dict_mt_<src>__<tgt>.jsonl
  training/dictionary_moe_dataset_info.snippet.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dictionary_common import (  # noqa: E402
    LOW_RESOURCE_LANGS,
    PIVOT_LANGS,
    lang_name,
    load_jsonl,
    repo_root,
    unique_keep_order,
    write_jsonl_with_preview,
)


def default_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for low in LOW_RESOURCE_LANGS:
        for pivot in PIVOT_LANGS:
            pairs.append((low, pivot))
            pairs.append((pivot, low))
    return pairs


def pairs_from_config(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise SystemExit(f"{path} has no top-level pairs list")
    out: list[tuple[str, str]] = []
    for item in pairs:
        src = item.get("src_lang")
        tgt = item.get("tgt_lang")
        if not src or not tgt:
            raise SystemExit(f"Bad pair item in {path}: {item!r}")
        out.append((src, tgt))
    return list(dict.fromkeys(out))


def make_instruction(src_lang: str, tgt_lang: str) -> str:
    return (
        f"Translate the following {lang_name(src_lang)} dictionary term into {lang_name(tgt_lang)}. "
        "Output only the best translation."
    )


def convert_records(
    path: Path,
    *,
    src_lang: str,
    tgt_lang: str,
    include_meta: bool,
    max_candidates_in_input: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    instruction = make_instruction(src_lang, tgt_lang)
    for rec in load_jsonl(path):
        src_text = str(rec.get("source_text", "")).strip()
        candidates = unique_keep_order(
            str(c) for c in (rec.get("target_candidates", []) or [rec.get("target_text", "")])
        )
        if not src_text or not candidates:
            continue
        input_text = src_text
        if max_candidates_in_input > 0 and len(candidates) > 1:
            # Use candidate hints only when explicitly requested; default training remains single-answer MT.
            hints = "; ".join(candidates[:max_candidates_in_input])
            input_text = f"{src_text}\nCandidate translations: {hints}"
        out: dict[str, Any] = {
            "instruction": instruction,
            "input": input_text,
            "output": candidates[0],
        }
        if include_meta:
            out["meta"] = {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source": rec.get("source"),
                "confidence": rec.get("confidence"),
                "all_targets": candidates,
                "pivot_text": rec.get("pivot_text"),
            }
        records.append(out)
        if limit is not None and len(records) >= limit:
            break
    return records


def dataset_info_entry(src: str, tgt: str, out_subdir: str) -> dict[str, Any]:
    return {
        "file_name": f"{out_subdir.rstrip('/')}/dict_mt_{src}__{tgt}.jsonl",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }


def update_dataset_info(path: Path, snippet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{path} must contain a JSON object")
    else:
        data = {}
    data.update(snippet)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Alpaca JSONL from dictionary lexicons.")
    ap.add_argument("--pairs-config", type=Path, default=repo_root() / "training/moe_pair_limits.json")
    ap.add_argument("--ignore-pairs-config", action="store_true")
    ap.add_argument("--input-dir", type=Path, default=repo_root() / "training/data/dictionaries/moe_lexicon")
    ap.add_argument("--out-dir", type=Path, default=repo_root() / "training/data/multilingual/dictionary_moe")
    ap.add_argument("--out-subdir", default="multilingual/dictionary_moe")
    ap.add_argument("--limit-per-direction", type=int, default=None)
    ap.add_argument("--include-meta", action="store_true")
    ap.add_argument("--max-candidates-in-input", type=int, default=0)
    ap.add_argument(
        "--dataset-info-snippet",
        type=Path,
        default=repo_root() / "training/dictionary_moe_dataset_info.snippet.json",
    )
    ap.add_argument(
        "--llamafactory-dataset-info",
        type=Path,
        default=repo_root() / "training/data/dataset_info.json",
    )
    ap.add_argument("--no-update-llamafactory-dataset-info", action="store_true")
    args = ap.parse_args()

    pairs = default_pairs() if args.ignore_pairs_config else pairs_from_config(args.pairs_config)
    snippet: dict[str, Any] = {}
    total = 0
    for src, tgt in pairs:
        in_path = args.input_dir / f"dict_terms_{src}__{tgt}.jsonl"
        if not in_path.exists():
            print(f"missing input: {in_path}", file=sys.stderr)
            continue
        out_path = args.out_dir / f"dict_mt_{src}__{tgt}.jsonl"
        count = write_jsonl_with_preview(
            out_path,
            convert_records(
                in_path,
                src_lang=src,
                tgt_lang=tgt,
                include_meta=args.include_meta,
                max_candidates_in_input=args.max_candidates_in_input,
                limit=args.limit_per_direction,
            ),
        )
        total += count
        name = f"dictionary_moe_{src}_to_{tgt}"
        snippet[name] = dataset_info_entry(src, tgt, args.out_subdir)
        print(f"wrote {count:>7} {out_path}")

    args.dataset_info_snippet.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_info_snippet.write_text(json.dumps(snippet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote dataset_info snippet: {args.dataset_info_snippet}")
    if not args.no_update_llamafactory_dataset_info:
        update_dataset_info(args.llamafactory_dataset_info, snippet)
        print(f"updated LLaMA-Factory dataset_info: {args.llamafactory_dataset_info}")
    print(f"done: {total} Alpaca dictionary MT rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
