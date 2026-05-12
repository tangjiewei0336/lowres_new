#!/usr/bin/env python3
"""Stream Kaikki (Wiktextract) Vietnamese JSONL into dict_terms_vie_Latn__eng_Latn.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dictionary_common import dict_record, ensure_dir, preview_path_for, repo_root  # noqa: E402

KAIKKI_LICENSE = (
    "Structured data from Wiktionary via Kaikki/Wiktextract; Wiktionary text is typically CC BY-SA 3.0 "
    "(confirm for your product)."
)


def sense_glosses(sense: dict) -> list[str]:
    out: list[str] = []
    for key in ("glosses", "raw_glosses"):
        g = sense.get(key)
        if isinstance(g, list):
            for x in g:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
    return out


def sense_examples(sense: dict, *, limit: int = 8) -> list[dict]:
    raw = sense.get("examples")
    if not isinstance(raw, list):
        return []
    rows: list[dict] = []
    for ex in raw[:limit]:
        if not isinstance(ex, dict):
            continue
        row = {k: ex[k] for k in ("text", "translation", "english", "type", "ref") if ex.get(k)}
        if row:
            rows.append(row)
    return rows


def wiktionary_url(word: str) -> str:
    return f"https://en.wiktionary.org/wiki/{quote(word, safe='')}#Vietnamese"


def iter_rows(obj: dict) -> list[dict]:
    lc = obj.get("lang_code")
    if lc != "vi" and str(obj.get("lang") or "") != "Vietnamese":
        return []
    word = str(obj.get("word") or "").strip()
    if not word:
        return []
    pos = str(obj.get("pos") or "unknown")
    senses = obj.get("senses")
    if not isinstance(senses, list):
        return []
    url = wiktionary_url(word)
    out: list[dict] = []
    for sense in senses:
        if not isinstance(sense, dict):
            continue
        glosses = sense_glosses(sense)
        if not glosses:
            continue
        examples = sense_examples(sense)
        try:
            rec = dict_record(
                source="kaikki_wiktextract",
                src_lang="vie_Latn",
                tgt_lang="eng_Latn",
                source_text=word,
                target_candidates=glosses,
                confidence=1.0 if examples else 0.85,
                source_url=url,
                license_note=KAIKKI_LICENSE,
                examples=examples or None,
            )
        except ValueError:
            continue
        rec["sense_pos"] = pos
        sense_id = sense.get("id")
        if isinstance(sense_id, str):
            rec["sense_id"] = sense_id
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build vie_Latn→eng_Latn lexicon from Kaikki Vietnamese JSONL.")
    ap.add_argument("--input", type=Path, required=True, help="Path to kaikki.org-dictionary-Vietnamese.jsonl")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "training" / "data" / "dictionaries" / "moe_lexicon",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max input lines to read (0 = all).")
    args = ap.parse_args()
    inp = args.input
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")
    out_path = args.output_dir / "dict_terms_vie_Latn__eng_Latn.jsonl"
    ensure_dir(out_path.parent)
    preview_lines: list[str] = []
    count = 0
    n_in = 0
    with inp.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.limit and n_in >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rec in iter_rows(obj):
                js = json.dumps(rec, ensure_ascii=False)
                fout.write(js + "\n")
                if len(preview_lines) < 50:
                    preview_lines.append(js)
                count += 1
    prev = preview_path_for(out_path, 50)
    ensure_dir(prev.parent)
    prev.write_text("\n".join(preview_lines) + ("\n" if preview_lines else ""), encoding="utf-8")
    print(f"wrote {count} rows from {n_in} input lines -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
