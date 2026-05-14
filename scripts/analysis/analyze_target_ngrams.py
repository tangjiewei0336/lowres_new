#!/usr/bin/env python3
"""
Analyze target-language 1/2-grams in hypotheses and references.

Example:
  conda run -n lowres python scripts/analysis/analyze_target_ngrams.py \
    --hypotheses-jsonl hypotheses_generate_parallel/20260504_205143/qwen/hypotheses.jsonl \
    --monolingual-jsonl zho_Hans=training/data/monolingual/fineweb2_pt_zho_Hans.jsonl \
    --out-dir analysis/qwen
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
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


def iter_jsonl_text(path: Path, text_keys: list[str]) -> Any:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, str):
                yield row
                continue
            if not isinstance(row, dict):
                continue
            for key in text_keys:
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    yield value
                    break


def tokenize(text: str, lang: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if lang.startswith("tha") and thai_word_tokenize is not None:
        return [x.strip() for x in thai_word_tokenize(text, engine="newmm") if x.strip()]
    if lang.startswith("zho") or HAN_RE.search(text):
        return WORD_RE.findall(text)
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def parse_lang_path(items: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--monolingual-jsonl 需要 LANG=PATH 格式: {item}")
        lang, raw_path = item.split("=", 1)
        lang = lang.strip()
        path = Path(raw_path).expanduser()
        if not lang or not path.is_file():
            raise SystemExit(f"单语数据不存在或语言为空: {item}")
        out[lang].append(path)
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Hypothesis/reference target n-gram frequency analysis.")
    ap.add_argument("--hypotheses-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, action="append", default=[1, 2], help="n-gram order; can repeat.")
    ap.add_argument(
        "--monolingual-jsonl",
        action="append",
        default=[],
        metavar="LANG=PATH",
        help="Optional target-language monolingual JSONL. Can repeat.",
    )
    ap.add_argument("--monolingual-text-key", action="append", default=["text", "content", "raw_content"])
    ap.add_argument("--top-k", type=int, default=0, help="0 writes all n-grams; otherwise top K per group.")
    args = ap.parse_args()

    rows = read_jsonl(args.hypotheses_jsonl)
    if not rows:
        raise SystemExit(f"No rows found: {args.hypotheses_jsonl}")

    orders = sorted(set(args.n))
    test_counts: dict[tuple[str, str, str, str, int], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str, str, str, int]] = Counter()

    for row in rows:
        corpus = row.get("eval_corpus") or row.get("dataset") or ""
        pair = row.get("eval_pair") or f"{row.get('src_lang', '')}->{row.get('tgt_lang', '')}"
        tgt_lang = row.get("tgt_lang") or ""
        for side, field in (("hypothesis", "hypothesis"), ("reference", "reference_text")):
            toks = tokenize(str(row.get(field, "")), tgt_lang)
            for n in orders:
                grams = ngrams(toks, n)
                key = (str(corpus), str(pair), str(tgt_lang), side, n)
                test_counts[key].update(grams)
                totals[key] += len(grams)

    mono_paths = parse_lang_path(args.monolingual_jsonl)
    mono_counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    mono_totals: Counter[tuple[str, int]] = Counter()
    for lang, paths in mono_paths.items():
        for path in paths:
            for text in iter_jsonl_text(path, args.monolingual_text_key):
                toks = tokenize(text, lang)
                for n in orders:
                    grams = ngrams(toks, n)
                    mono_counts[(lang, n)].update(grams)
                    mono_totals[(lang, n)] += len(grams)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "target_ngrams.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "tgt_lang",
                "side",
                "n",
                "ngram",
                "test_count",
                "test_total",
                "test_freq",
                "monolingual_count",
                "monolingual_total",
                "monolingual_freq",
            ],
        )
        writer.writeheader()
        for key in sorted(test_counts):
            corpus, pair, tgt_lang, side, n = key
            items = test_counts[key].most_common(args.top_k or None)
            mono_total = mono_totals[(tgt_lang, n)]
            for gram, count in items:
                mono_count = mono_counts[(tgt_lang, n)].get(gram, 0)
                writer.writerow(
                    {
                        "corpus": corpus,
                        "pair": pair,
                        "tgt_lang": tgt_lang,
                        "side": side,
                        "n": n,
                        "ngram": gram,
                        "test_count": count,
                        "test_total": totals[key],
                        "test_freq": count / totals[key] if totals[key] else 0.0,
                        "monolingual_count": mono_count,
                        "monolingual_total": mono_total,
                        "monolingual_freq": mono_count / mono_total if mono_total else "",
                    }
                )

    summary = {
        "hypotheses_jsonl": str(args.hypotheses_jsonl),
        "num_rows": len(rows),
        "orders": orders,
        "monolingual_jsonl": {k: [str(p) for p in v] for k, v in mono_paths.items()},
        "output_csv": str(out_csv),
        "tokenization_note": "zho/Han uses Han characters plus Latin/number tokens; Thai uses PyThaiNLP when installed; other languages use regex word tokens.",
    }
    (args.out_dir / "target_ngrams.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
