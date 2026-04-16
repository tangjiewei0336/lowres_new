#!/usr/bin/env python3
"""
Download and normalize MUSE bilingual dictionaries for the MT MoE languages.

The MUSE text files are word-pair dictionaries from:
  https://github.com/facebookresearch/MUSE

Default outputs:
  training/data/dictionaries/raw/muse/<muse_src>-<muse_tgt>.txt
  training/data/dictionaries/lexicon/dict_terms_<src>__<tgt>.jsonl
  training/data/dictionaries/lexicon/previews/*.preview_50.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dictionary_common import (  # noqa: E402
    LOW_RESOURCE_LANGS,
    MUSE_CODES,
    clean_text,
    dict_record,
    ensure_dir,
    looks_simplified_chinese,
    repo_root,
    to_simplified_light,
    unique_keep_order,
    write_jsonl_with_preview,
)


MUSE_BASE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries"
MUSE_LICENSE_NOTE = (
    "MUSE dictionaries are distributed by facebookresearch/MUSE; use as research seed data "
    "unless your project has separately cleared the source licenses."
)


def parse_pairs_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            raise SystemExit(f"Bad pair line in {path}: {raw!r}")
        pairs.append((parts[0], parts[1]))
    return pairs


def default_pairs(include_zh: bool) -> list[tuple[str, str]]:
    pairs = [(lang, "eng_Latn") for lang in LOW_RESOURCE_LANGS]
    if include_zh:
        pairs.append(("zho_Hans", "eng_Latn"))
    return pairs


def muse_url(src_lang: str, tgt_lang: str, base_url: str) -> tuple[str, str, str]:
    src = MUSE_CODES[src_lang]
    tgt = MUSE_CODES[tgt_lang]
    return src, tgt, f"{base_url.rstrip('/')}/{src}-{tgt}.txt"


def download(url: str, out_path: Path, *, force: bool) -> None:
    if out_path.exists() and not force:
        return
    ensure_dir(out_path.parent)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; lowres-muse-dict/1.0)"})
    try:
        with urlopen(req, timeout=180) as resp:
            data = resp.read()
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Failed to download {url}: {exc}") from exc
    out_path.write_bytes(data)


def read_muse_rows(path: Path, *, reverse_columns: bool = False) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if reverse_columns:
                a, b = b, a
            a, b = clean_text(a), clean_text(b)
            if a and b:
                rows.append((a, b))
    return rows


def build_records(
    rows: list[tuple[str, str]],
    *,
    src_lang: str,
    tgt_lang: str,
    url: str,
    max_candidates: int,
    limit: int | None,
) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = {}
    source_surface: dict[str, str] = {}
    for src, tgt in rows:
        if src_lang == "zho_Hans":
            src = to_simplified_light(src)
        if tgt_lang == "zho_Hans":
            tgt = to_simplified_light(tgt)
        if src_lang == "zho_Hans" and not looks_simplified_chinese(src):
            continue
        if tgt_lang == "zho_Hans" and not looks_simplified_chinese(tgt):
            continue
        key = src.casefold()
        source_surface.setdefault(key, src)
        grouped.setdefault(key, []).append(tgt)

    records: list[dict[str, object]] = []
    for key, candidates in grouped.items():
        cleaned = unique_keep_order(candidates, limit=max_candidates)
        if not cleaned:
            continue
        records.append(
            dict_record(
                source="muse",
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                source_text=source_surface[key],
                target_candidates=cleaned,
                confidence=0.82,
                source_url=url,
                license_note=MUSE_LICENSE_NOTE,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare normalized MUSE dictionaries.")
    ap.add_argument("--pairs-file", type=Path, default=None, help="Optional file with '<src> <tgt>' lines.")
    ap.add_argument("--include-zh", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--base-url", default=MUSE_BASE_URL)
    ap.add_argument("--raw-dir", type=Path, default=repo_root() / "training/data/dictionaries/raw/muse")
    ap.add_argument("--out-dir", type=Path, default=repo_root() / "training/data/dictionaries/lexicon")
    ap.add_argument("--limit-per-direction", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    pairs = parse_pairs_file(args.pairs_file) if args.pairs_file else default_pairs(args.include_zh)
    total = 0
    for src_lang, tgt_lang in pairs:
        if src_lang not in MUSE_CODES or tgt_lang not in MUSE_CODES:
            raise SystemExit(f"MUSE code is not configured for {src_lang}->{tgt_lang}")
        muse_src, muse_tgt, url = muse_url(src_lang, tgt_lang, args.base_url)
        raw_path = args.raw_dir / f"{muse_src}-{muse_tgt}.txt"
        download(url, raw_path, force=args.force_download)
        print(f"ready raw={raw_path} url={url}")
        if args.download_only:
            continue

        rows = read_muse_rows(raw_path)
        out_path = args.out_dir / f"dict_terms_{src_lang}__{tgt_lang}.jsonl"
        count = write_jsonl_with_preview(
            out_path,
            build_records(
                rows,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                url=url,
                max_candidates=args.max_candidates,
                limit=args.limit_per_direction,
            ),
        )
        total += count
        print(f"wrote {count:>7} {out_path}")

        if args.bidirectional:
            rev_out_path = args.out_dir / f"dict_terms_{tgt_lang}__{src_lang}.jsonl"
            rev_count = write_jsonl_with_preview(
                rev_out_path,
                build_records(
                    read_muse_rows(raw_path, reverse_columns=True),
                    src_lang=tgt_lang,
                    tgt_lang=src_lang,
                    url=url,
                    max_candidates=args.max_candidates,
                    limit=args.limit_per_direction,
                ),
            )
            total += rev_count
            print(f"wrote {rev_count:>7} {rev_out_path}")

    print(f"done: {total} normalized entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
