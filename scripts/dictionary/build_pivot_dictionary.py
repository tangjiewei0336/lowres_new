#!/usr/bin/env python3
"""
Build MoE dictionary lexicons by pivoting through English.

Inputs are normalized files from prepare_muse_dictionary.py and
prepare_cc_cedict_dictionary.py:
  training/data/dictionaries/lexicon/dict_terms_<src>__<tgt>.jsonl

Default outputs keep the same lexicon schema and cover the 20 MoE directions:
  five low-resource languages <-> English
  five low-resource languages <-> Simplified Chinese
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dictionary_common import (  # noqa: E402
    EN_PIVOT_STOPWORDS,
    LOW_RESOURCE_LANGS,
    clean_text,
    dict_record,
    ensure_dir,
    load_jsonl,
    normalize_key,
    repo_root,
    unique_keep_order,
    write_jsonl_with_preview,
)


PIVOT_LICENSE_NOTE = (
    "Pivoted dictionary entry built from normalized bilingual dictionary sources; "
    "keep downstream usage compatible with all source licenses."
)


def usable_english_pivot(text: str) -> bool:
    key = normalize_key(text)
    if not key or key in EN_PIVOT_STOPWORDS:
        return False
    if len(key) < 3:
        return False
    return True


def lexicon_path(root: Path, src: str, tgt: str) -> Path:
    return root / f"dict_terms_{src}__{tgt}.jsonl"


def load_by_source(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return {}
    for rec in load_jsonl(path):
        grouped[normalize_key(str(rec.get("source_text", "")))].append(rec)
    return dict(grouped)


def load_by_target_candidate(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return {}
    for rec in load_jsonl(path):
        for cand in rec.get("target_candidates", []) or [rec.get("target_text", "")]:
            grouped[normalize_key(str(cand))].append(rec)
    return dict(grouped)


def copy_existing(src_path: Path, out_path: Path, *, overwrite: bool) -> int:
    if not src_path.exists():
        print(f"missing input: {src_path}", file=sys.stderr)
        return 0
    if out_path.resolve() == src_path.resolve():
        count = sum(1 for _ in load_jsonl(out_path))
        return count
    if out_path.exists() and not overwrite:
        count = sum(1 for _ in load_jsonl(out_path))
        print(f"kept existing {out_path} ({count})")
        return count
    ensure_dir(out_path.parent)
    shutil.copyfile(src_path, out_path)
    prev_src = src_path.parent / "previews" / f"{src_path.stem}.preview_50.jsonl"
    prev_out = out_path.parent / "previews" / f"{out_path.stem}.preview_50.jsonl"
    if prev_src.exists():
        ensure_dir(prev_out.parent)
        shutil.copyfile(prev_src, prev_out)
    count = sum(1 for _ in load_jsonl(out_path))
    print(f"copied {count:>7} {out_path}")
    return count


def build_low_to_zh(
    *,
    lang: str,
    input_dir: Path,
    max_candidates: int,
    limit: int | None,
) -> list[dict[str, object]]:
    low_to_en = load_jsonl(lexicon_path(input_dir, lang, "eng_Latn"))
    en_to_zh = load_by_source(lexicon_path(input_dir, "eng_Latn", "zho_Hans"))
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for low_rec in low_to_en:
        src_text = clean_text(str(low_rec.get("source_text", "")))
        if not src_text:
            continue
        key = normalize_key(src_text)
        if key in seen:
            continue
        zh_candidates: list[str] = []
        pivots = low_rec.get("target_candidates", []) or [low_rec.get("target_text", "")]
        matched_pivot = ""
        for pivot in pivots:
            if not usable_english_pivot(str(pivot)):
                continue
            pivot_key = normalize_key(str(pivot))
            matches = en_to_zh.get(pivot_key, [])
            if not matches:
                continue
            matched_pivot = str(pivot)
            for zh_rec in matches:
                zh_candidates.extend(str(c) for c in zh_rec.get("target_candidates", []) or [zh_rec.get("target_text", "")])
        zh_candidates = unique_keep_order(zh_candidates, limit=max_candidates)
        if not zh_candidates:
            continue
        seen.add(key)
        records.append(
            dict_record(
                source="english_pivot_dictionary",
                src_lang=lang,
                tgt_lang="zho_Hans",
                source_text=src_text,
                target_candidates=zh_candidates,
                confidence=0.62,
                source_url="pivot:dict_terms_%s__eng_Latn + dict_terms_eng_Latn__zho_Hans" % lang,
                license_note=PIVOT_LICENSE_NOTE,
                pivot_text=matched_pivot,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def build_zh_to_low(
    *,
    lang: str,
    input_dir: Path,
    max_candidates: int,
    limit: int | None,
) -> list[dict[str, object]]:
    zh_to_en = load_jsonl(lexicon_path(input_dir, "zho_Hans", "eng_Latn"))
    en_to_low = load_by_source(lexicon_path(input_dir, "eng_Latn", lang))
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for zh_rec in zh_to_en:
        zh_text = clean_text(str(zh_rec.get("source_text", "")))
        if not zh_text:
            continue
        key = normalize_key(zh_text)
        if key in seen:
            continue
        low_candidates: list[str] = []
        pivots = zh_rec.get("target_candidates", []) or [zh_rec.get("target_text", "")]
        matched_pivot = ""
        for pivot in pivots:
            if not usable_english_pivot(str(pivot)):
                continue
            pivot_key = normalize_key(str(pivot))
            matches = en_to_low.get(pivot_key, [])
            if not matches:
                continue
            matched_pivot = str(pivot)
            for low_rec in matches:
                low_candidates.extend(
                    str(c) for c in low_rec.get("target_candidates", []) or [low_rec.get("target_text", "")]
                )
        low_candidates = unique_keep_order(low_candidates, limit=max_candidates)
        if not low_candidates:
            continue
        seen.add(key)
        records.append(
            dict_record(
                source="english_pivot_dictionary",
                src_lang="zho_Hans",
                tgt_lang=lang,
                source_text=zh_text,
                target_candidates=low_candidates,
                confidence=0.62,
                source_url="pivot:dict_terms_zho_Hans__eng_Latn + dict_terms_eng_Latn__%s" % lang,
                license_note=PIVOT_LICENSE_NOTE,
                pivot_text=matched_pivot,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Build English-pivoted dictionary lexicons for MoE pairs.")
    ap.add_argument("--input-dir", type=Path, default=repo_root() / "training/data/dictionaries/lexicon")
    ap.add_argument("--out-dir", type=Path, default=repo_root() / "training/data/dictionaries/moe_lexicon")
    ap.add_argument("--langs", nargs="+", default=list(LOW_RESOURCE_LANGS))
    ap.add_argument("--limit-per-direction", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    total = 0
    for lang in args.langs:
        total += copy_existing(
            lexicon_path(args.input_dir, lang, "eng_Latn"),
            lexicon_path(args.out_dir, lang, "eng_Latn"),
            overwrite=args.overwrite,
        )
        total += copy_existing(
            lexicon_path(args.input_dir, "eng_Latn", lang),
            lexicon_path(args.out_dir, "eng_Latn", lang),
            overwrite=args.overwrite,
        )

        low_zh_out = lexicon_path(args.out_dir, lang, "zho_Hans")
        low_zh_count = write_jsonl_with_preview(
            low_zh_out,
            build_low_to_zh(
                lang=lang,
                input_dir=args.input_dir,
                max_candidates=args.max_candidates,
                limit=args.limit_per_direction,
            ),
        )
        total += low_zh_count
        print(f"wrote {low_zh_count:>7} {low_zh_out}")

        zh_low_out = lexicon_path(args.out_dir, "zho_Hans", lang)
        zh_low_count = write_jsonl_with_preview(
            zh_low_out,
            build_zh_to_low(
                lang=lang,
                input_dir=args.input_dir,
                max_candidates=args.max_candidates,
                limit=args.limit_per_direction,
            ),
        )
        total += zh_low_count
        print(f"wrote {zh_low_count:>7} {zh_low_out}")

    print(f"done: {total} MoE lexicon entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
