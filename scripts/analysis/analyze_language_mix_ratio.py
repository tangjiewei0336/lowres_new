#!/usr/bin/env python3
"""
Estimate source/target/other script mixing in each hypothesis.

This is script-based, so Latin->Latin directions are marked as shared-script
and cannot be separated into source vs target by script alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[\u0e00-\u0e7f]+|[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?|\d+(?:[.,]\d+)*",
    re.UNICODE,
)

LANG_SCRIPTS = {
    "zho": {"Han"},
    "tha": {"Thai"},
    "eng": {"Latin"},
    "spa": {"Latin"},
    "vie": {"Latin"},
    "ind": {"Latin"},
    "tgl": {"Latin"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lang_scripts(lang: str) -> set[str]:
    return set(LANG_SCRIPTS.get((lang or "").split("_", 1)[0], set()))


def token_script(token: str) -> str:
    scripts = Counter()
    for ch in token:
        if ch.isdigit():
            scripts["Number"] += 1
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for script in ("LATIN", "CJK UNIFIED", "CJK COMPATIBILITY", "THAI"):
            if script in name:
                if script.startswith("CJK"):
                    scripts["Han"] += 1
                elif script == "THAI":
                    scripts["Thai"] += 1
                else:
                    scripts["Latin"] += 1
                break
    if not scripts:
        return "Other"
    return scripts.most_common(1)[0][0]


def classify(script: str, src_scripts: set[str], tgt_scripts: set[str]) -> str:
    if script == "Number":
        return "neutral"
    in_src = script in src_scripts
    in_tgt = script in tgt_scripts
    if in_src and in_tgt:
        return "shared"
    if in_src:
        return "source"
    if in_tgt:
        return "target"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze source/target/other token mixing in hypotheses.")
    ap.add_argument("--hypotheses-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.hypotheses_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "language_mix_ratio.csv"
    aggregate: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "sample_id",
                "src_lang",
                "tgt_lang",
                "token_total",
                "source_tokens",
                "target_tokens",
                "shared_tokens",
                "other_tokens",
                "neutral_tokens",
                "source_ratio",
                "target_ratio",
                "shared_ratio",
                "other_ratio",
                "neutral_ratio",
                "ambiguous_same_script",
            ],
        )
        writer.writeheader()
        for row in rows:
            corpus = row.get("eval_corpus") or row.get("dataset") or ""
            pair = row.get("eval_pair") or f"{row.get('src_lang', '')}->{row.get('tgt_lang', '')}"
            src_lang = str(row.get("src_lang", ""))
            tgt_lang = str(row.get("tgt_lang", ""))
            src_scripts = lang_scripts(src_lang)
            tgt_scripts = lang_scripts(tgt_lang)
            counts = Counter()
            for tok in TOKEN_RE.findall(str(row.get("hypothesis", ""))):
                counts[classify(token_script(tok), src_scripts, tgt_scripts)] += 1
            total = sum(counts.values())
            aggregate[(str(corpus), str(pair))].update(counts)
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "sample_id": row.get("sample_id", ""),
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "token_total": total,
                    "source_tokens": counts["source"],
                    "target_tokens": counts["target"],
                    "shared_tokens": counts["shared"],
                    "other_tokens": counts["other"],
                    "neutral_tokens": counts["neutral"],
                    "source_ratio": counts["source"] / total if total else 0.0,
                    "target_ratio": counts["target"] / total if total else 0.0,
                    "shared_ratio": counts["shared"] / total if total else 0.0,
                    "other_ratio": counts["other"] / total if total else 0.0,
                    "neutral_ratio": counts["neutral"] / total if total else 0.0,
                    "ambiguous_same_script": bool(src_scripts & tgt_scripts),
                }
            )

    out_summary = args.out_dir / "language_mix_ratio.by_pair.csv"
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["corpus", "pair", "token_total", "source_ratio", "target_ratio", "shared_ratio", "other_ratio", "neutral_ratio"],
        )
        writer.writeheader()
        for (corpus, pair), counts in sorted(aggregate.items()):
            total = sum(counts.values())
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "token_total": total,
                    "source_ratio": counts["source"] / total if total else 0.0,
                    "target_ratio": counts["target"] / total if total else 0.0,
                    "shared_ratio": counts["shared"] / total if total else 0.0,
                    "other_ratio": counts["other"] / total if total else 0.0,
                    "neutral_ratio": counts["neutral"] / total if total else 0.0,
                }
            )

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
