#!/usr/bin/env python3
"""Compare hypothesis/source length ratio against reference/source ratio."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    from pythainlp.tokenize import word_tokenize as thai_word_tokenize
except Exception:  # pragma: no cover
    thai_word_tokenize = None

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WORD_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?|\d+(?:[.,]\d+)*",
    re.UNICODE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str, lang: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if lang.startswith("tha") and thai_word_tokenize is not None:
        return [x.strip() for x in thai_word_tokenize(text, engine="newmm") if x.strip()]
    if lang.startswith("zho") or HAN_RE.search(text):
        return WORD_RE.findall(text)
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def safe_ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze hypothesis/reference length ratios.")
    ap.add_argument("--hypotheses-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--unit", choices=["token", "char"], default="token")
    args = ap.parse_args()

    rows = read_jsonl(args.hypotheses_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"length_ratios.{args.unit}.csv"
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "sample_id",
                "src_len",
                "hyp_len",
                "ref_len",
                "hyp_src_ratio",
                "ref_src_ratio",
                "ratio_delta",
                "hyp_ref_len_ratio",
            ],
        )
        writer.writeheader()
        for row in rows:
            corpus = row.get("eval_corpus") or row.get("dataset") or ""
            pair = row.get("eval_pair") or f"{row.get('src_lang', '')}->{row.get('tgt_lang', '')}"
            if args.unit == "char":
                src_len = len(str(row.get("source_text", "")).strip())
                hyp_len = len(str(row.get("hypothesis", "")).strip())
                ref_len = len(str(row.get("reference_text", "")).strip())
            else:
                src_len = len(tokenize(str(row.get("source_text", "")), str(row.get("src_lang", ""))))
                hyp_len = len(tokenize(str(row.get("hypothesis", "")), str(row.get("tgt_lang", ""))))
                ref_len = len(tokenize(str(row.get("reference_text", "")), str(row.get("tgt_lang", ""))))
            hyp_src = safe_ratio(hyp_len, src_len)
            ref_src = safe_ratio(ref_len, src_len)
            item = {
                "hyp_src_ratio": hyp_src,
                "ref_src_ratio": ref_src,
                "ratio_delta": hyp_src - ref_src,
                "hyp_ref_len_ratio": safe_ratio(hyp_len, ref_len),
            }
            grouped[(str(corpus), str(pair))].append(item)
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "sample_id": row.get("sample_id", ""),
                    "src_len": src_len,
                    "hyp_len": hyp_len,
                    "ref_len": ref_len,
                    **item,
                }
            )

    out_summary = args.out_dir / f"length_ratios.by_pair.{args.unit}.csv"
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "num",
                "mean_hyp_src_ratio",
                "mean_ref_src_ratio",
                "mean_ratio_delta",
                "median_ratio_delta",
                "mean_hyp_ref_len_ratio",
            ],
        )
        writer.writeheader()
        for (corpus, pair), vals in sorted(grouped.items()):
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "num": len(vals),
                    "mean_hyp_src_ratio": mean(v["hyp_src_ratio"] for v in vals),
                    "mean_ref_src_ratio": mean(v["ref_src_ratio"] for v in vals),
                    "mean_ratio_delta": mean(v["ratio_delta"] for v in vals),
                    "median_ratio_delta": median(v["ratio_delta"] for v in vals),
                    "mean_hyp_ref_len_ratio": mean(v["hyp_ref_len_ratio"] for v in vals),
                }
            )

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
