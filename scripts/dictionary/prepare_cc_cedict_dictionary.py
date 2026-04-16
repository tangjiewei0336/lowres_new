#!/usr/bin/env python3
"""
Download and normalize CC-CEDICT Chinese-English entries.

Default outputs:
  training/data/dictionaries/raw/cc_cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz
  training/data/dictionaries/lexicon/dict_terms_zho_Hans__eng_Latn.jsonl
  training/data/dictionaries/lexicon/dict_terms_eng_Latn__zho_Hans.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dictionary_common import (  # noqa: E402
    clean_text,
    dict_record,
    ensure_dir,
    looks_simplified_chinese,
    normalize_key,
    repo_root,
    unique_keep_order,
    write_jsonl_with_preview,
)


CEDICT_URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
CEDICT_LICENSE_NOTE = "CC-CEDICT is licensed under CC BY-SA 4.0."
LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[[^\]]*]\s+/(.*)/$")
PARENS_RE = re.compile(r"\s*\([^)]*\)")


def download(url: str, out_path: Path, *, force: bool) -> None:
    if out_path.exists() and not force:
        return
    ensure_dir(out_path.parent)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; lowres-cc-cedict/1.0)"})
    try:
        with urlopen(req, timeout=180) as resp:
            data = resp.read()
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Failed to download {url}: {exc}") from exc
    out_path.write_bytes(data)


def clean_definition(defn: str) -> str:
    defn = clean_text(defn)
    defn = re.sub(r"^CL:.*$", "", defn)
    defn = re.sub(r"^variant of .*$", "", defn, flags=re.IGNORECASE)
    defn = re.sub(r"^old variant of .*$", "", defn, flags=re.IGNORECASE)
    defn = re.sub(r"^abbr\. for .*$", "", defn, flags=re.IGNORECASE)
    defn = re.sub(r"\b(Tw|Taiwan|PRC|Mainland China) pr\..*$", "", defn)
    defn = PARENS_RE.sub("", defn)
    defn = defn.strip(" ;,")
    if not defn or len(defn) > 80:
        return ""
    if any(mark in defn for mark in ("[", "]", "surname ", "see also ")):
        return ""
    return defn


def iter_cedict(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            _traditional, simplified, defs_blob = m.groups()
            if not looks_simplified_chinese(simplified):
                continue
            defs = unique_keep_order(
                clean_definition(part)
                for part in defs_blob.split("/")
                if part and not part.startswith("CL:")
            )
            if defs:
                rows.append((simplified, defs))
    return rows


def build_zh_to_en(rows: list[tuple[str, list[str]]], *, url: str, limit: int | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for zh, defs in rows:
        key = normalize_key(zh)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            dict_record(
                source="cc_cedict",
                src_lang="zho_Hans",
                tgt_lang="eng_Latn",
                source_text=zh,
                target_candidates=defs,
                confidence=0.9,
                source_url=url,
                license_note=CEDICT_LICENSE_NOTE,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def build_en_to_zh(rows: list[tuple[str, list[str]]], *, url: str, limit: int | None) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = {}
    surface: dict[str, str] = {}
    for zh, defs in rows:
        for defn in defs:
            key = normalize_key(defn)
            surface.setdefault(key, defn)
            grouped.setdefault(key, []).append(zh)
    records: list[dict[str, object]] = []
    for key, zh_terms in grouped.items():
        records.append(
            dict_record(
                source="cc_cedict",
                src_lang="eng_Latn",
                tgt_lang="zho_Hans",
                source_text=surface[key],
                target_candidates=zh_terms,
                confidence=0.88,
                source_url=url,
                license_note=CEDICT_LICENSE_NOTE,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare normalized CC-CEDICT dictionaries.")
    ap.add_argument("--url", default=CEDICT_URL)
    ap.add_argument(
        "--raw-path",
        type=Path,
        default=repo_root() / "training/data/dictionaries/raw/cc_cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz",
    )
    ap.add_argument("--out-dir", type=Path, default=repo_root() / "training/data/dictionaries/lexicon")
    ap.add_argument("--limit-per-direction", type=int, default=None)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    download(args.url, args.raw_path, force=args.force_download)
    print(f"ready raw={args.raw_path} url={args.url}")
    if args.download_only:
        return 0

    rows = iter_cedict(args.raw_path)
    zh_out = args.out_dir / "dict_terms_zho_Hans__eng_Latn.jsonl"
    en_out = args.out_dir / "dict_terms_eng_Latn__zho_Hans.jsonl"
    zh_count = write_jsonl_with_preview(zh_out, build_zh_to_en(rows, url=args.url, limit=args.limit_per_direction))
    en_count = write_jsonl_with_preview(en_out, build_en_to_zh(rows, url=args.url, limit=args.limit_per_direction))
    print(f"wrote {zh_count:>7} {zh_out}")
    print(f"wrote {en_count:>7} {en_out}")
    print(f"done: {zh_count + en_count} normalized entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
