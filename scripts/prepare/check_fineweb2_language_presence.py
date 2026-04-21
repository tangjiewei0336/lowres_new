#!/usr/bin/env python3
"""
Check whether target languages exist in HuggingFaceFW/fineweb-2 and sample their content.

This is a lightweight validation step before synthetic MT generation:
1. Verify that each requested language/config exists in FineWeb-2.
2. Stream a small number of rows from each config.
3. Apply simple script heuristics and write a JSON report.

The script defaults to the 7 languages already used in this repo:
  eng_Latn zho_Hans spa_Latn ind_Latn vie_Latn tha_Thai tgl_Latn
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_LANGS = (
    "eng_Latn",
    "zho_Hans",
    "spa_Latn",
    "ind_Latn",
    "vie_Latn",
    "tha_Thai",
    "tgl_Latn",
)

DEFAULT_LANG_ALIAS: dict[str, str] = {
    "zho_Hans": "cmn_Hani",
}

TRADITIONAL_CHARS = set(
    "體臺灣國語門風龍馬鳥魚貝車長東電會學萬與專業網頁點擊"
    "後來開關樂書時過無對發現為這個們還進廣場線愛雲氣區"
    "醫藥術聽說讀寫貓畫戰爭讓從應當實驗數據資料軟體硬體"
)


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def cjk_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def thai_count(text: str) -> int:
    return sum(1 for ch in text if "\u0e00" <= ch <= "\u0e7f")


def latin_count(text: str) -> int:
    return sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))


def has_traditional_chars(text: str) -> bool:
    return any(ch in TRADITIONAL_CHARS for ch in text)


def resolve_text(row: dict[str, Any], min_chars: int) -> str | None:
    for key in ("text", "content", "raw_content"):
        v = row.get(key)
        if isinstance(v, str):
            s = normalize_ws(v)
            if len(s) >= min_chars:
                return s
    fallback = ""
    for v in row.values():
        if isinstance(v, str):
            s = normalize_ws(v)
            if len(s) > len(fallback):
                fallback = s
    return fallback if len(fallback) >= min_chars else None


def looks_like_lang(text: str, lang: str, *, min_ratio: float) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False
    if lang == "zho_Hans":
        return (cjk_count(text) / max(1, len(chars)) >= min_ratio) and not has_traditional_chars(text)
    if lang == "tha_Thai":
        return thai_count(text) / max(1, len(chars)) >= min_ratio
    # Latin-script languages: weak heuristic only.
    return latin_count(text) / max(1, len(chars)) >= min_ratio


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate language presence in FineWeb-2 before synthetic MT generation.")
    ap.add_argument("--repo-id", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--lang", action="append", dest="langs", help="FLORES code; can be repeated.")
    ap.add_argument("--sample-rows", type=int, default=200, help="Rows to inspect per language.")
    ap.add_argument("--preview-n", type=int, default=5, help="How many passing examples to keep in the report.")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--min-ratio", type=float, default=0.35)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--lang-alias", action="append", dest="lang_aliases", metavar="SRC=DST")
    ap.add_argument(
        "--report",
        type=Path,
        default=root() / "training" / "reports" / "fineweb2_language_presence.json",
    )
    args = ap.parse_args()

    langs = [x.strip() for x in (args.langs or list(DEFAULT_LANGS)) if x and x.strip()]
    alias_map = dict(DEFAULT_LANG_ALIAS)
    for raw in args.lang_aliases or []:
        if "=" not in raw:
            raise SystemExit(f"bad --lang-alias: {raw}")
        src, dst = raw.split("=", 1)
        alias_map[src.strip()] = dst.strip()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    from datasets import get_dataset_config_names, load_dataset

    configs = set(get_dataset_config_names(args.repo_id, token=token or None))
    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "split": args.split,
        "sample_rows": int(args.sample_rows),
        "languages": {},
    }

    for lang in langs:
        repo_lang = alias_map.get(lang, lang)
        entry: dict[str, Any] = {
            "repo_lang": repo_lang,
            "config_exists": repo_lang in configs,
            "rows_seen": 0,
            "rows_passing": 0,
            "samples": [],
        }
        report["languages"][lang] = entry
        if repo_lang not in configs:
            continue

        ds = load_dataset(
            args.repo_id,
            repo_lang,
            split=args.split,
            streaming=True,
            token=token or None,
        )
        for row in ds:
            if entry["rows_seen"] >= int(args.sample_rows):
                break
            text = resolve_text(row, min_chars=int(args.min_chars))
            if not text:
                continue
            entry["rows_seen"] += 1
            ok = looks_like_lang(text, lang, min_ratio=float(args.min_ratio))
            if ok:
                entry["rows_passing"] += 1
                if len(entry["samples"]) < int(args.preview_n):
                    entry["samples"].append(text[:400])
        seen = int(entry["rows_seen"])
        passing = int(entry["rows_passing"])
        entry["pass_ratio"] = (passing / seen) if seen else 0.0

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report: {args.report}")
    for lang, entry in report["languages"].items():
        print(
            f"{lang}: config_exists={entry['config_exists']} rows_seen={entry['rows_seen']} "
            f"rows_passing={entry['rows_passing']} pass_ratio={entry.get('pass_ratio', 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
