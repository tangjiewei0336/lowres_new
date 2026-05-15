#!/usr/bin/env python3
"""Compare hypothesis/source length ratio against reference/source ratio.

Output CSV (`length_ratios.<unit>.csv`) feeds `build_mix_len_ratio_config.py`
to derive per-language-pair length-ratio bands from `ref_src_ratio` statistics.
Use that script's `--hypotheses-jsonl` when you want the same tokenizer as mix
(`mix_hypothesis_candidates` TOKEN_RE) instead of this analyzer's per-lang rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
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


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return out.strip("_") or "unknown"


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def maybe_write_plots(out_dir: Path, grouped: dict[tuple[str, str], list[dict[str, float]]], unit: str, max_plot_groups: int) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skip plots: matplotlib unavailable: {e}")
        return []

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(grouped)
    if max_plot_groups and max_plot_groups > 0:
        groups = groups[:max_plot_groups]
    written: list[Path] = []
    for corpus, pair in groups:
        vals = grouped[(corpus, pair)]
        hyp_src = [v["hyp_src_ratio"] for v in vals]
        ref_src = [v["ref_src_ratio"] for v in vals]
        delta = [v["ratio_delta"] for v in vals]

        fig, ax = plt.subplots(figsize=(6.2, 6.0))
        ax.scatter(ref_src, hyp_src, s=12, alpha=0.55, color="#4477AA")
        max_v = max(hyp_src + ref_src) if hyp_src or ref_src else 1.0
        ax.plot([0, max_v], [0, max_v], linestyle="--", color="#666666", linewidth=1.0)
        ax.set_xlabel(f"Reference/source length ratio ({unit})")
        ax.set_ylabel(f"Hypothesis/source length ratio ({unit})")
        ax.set_title(f"Length ratio scatter: {corpus} {pair}")
        fig.tight_layout()
        path = plot_dir / f"length_ratio__{safe_name(corpus)}__{safe_name(pair)}__scatter_{unit}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(delta, bins=35, color="#CC6677", alpha=0.85)
        ax.axvline(0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xlabel("hyp_src_ratio - ref_src_ratio")
        ax.set_ylabel("Sentences")
        ax.set_title(f"Length ratio delta: {corpus} {pair}")
        fig.tight_layout()
        path = plot_dir / f"length_ratio__{safe_name(corpus)}__{safe_name(pair)}__delta_hist_{unit}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze hypothesis/reference length ratios.")
    ap.add_argument("--hypotheses-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--unit", choices=["token", "char"], default="token")
    ap.add_argument("--short-threshold", type=float, default=0.85, help="Flag hyp/ref length ratio below this value.")
    ap.add_argument("--long-threshold", type=float, default=1.15, help="Flag hyp/ref length ratio above this value.")
    ap.add_argument("--max-plot-groups", type=int, default=0, help="0 plots all corpus/pair groups.")
    ap.add_argument("--no-plots", action="store_true")
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
                "is_short",
                "is_long",
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
                "is_short": safe_ratio(hyp_len, ref_len) < args.short_threshold if ref_len else False,
                "is_long": safe_ratio(hyp_len, ref_len) > args.long_threshold if ref_len else False,
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
                "std_ratio_delta",
                "p05_ratio_delta",
                "p95_ratio_delta",
                "mean_hyp_ref_len_ratio",
                "median_hyp_ref_len_ratio",
                "p05_hyp_ref_len_ratio",
                "p95_hyp_ref_len_ratio",
                "short_count",
                "short_rate",
                "long_count",
                "long_rate",
                "length_ratio_corr",
            ],
        )
        writer.writeheader()
        for (corpus, pair), vals in sorted(grouped.items()):
            deltas = [v["ratio_delta"] for v in vals]
            hyp_ref = [v["hyp_ref_len_ratio"] for v in vals]
            hyp_src = [v["hyp_src_ratio"] for v in vals]
            ref_src = [v["ref_src_ratio"] for v in vals]
            short_count = sum(1 for v in vals if v["is_short"])
            long_count = sum(1 for v in vals if v["is_long"])
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "num": len(vals),
                    "mean_hyp_src_ratio": mean(v["hyp_src_ratio"] for v in vals),
                    "mean_ref_src_ratio": mean(v["ref_src_ratio"] for v in vals),
                    "mean_ratio_delta": mean(deltas),
                    "median_ratio_delta": median(deltas),
                    "std_ratio_delta": pstdev(deltas) if len(deltas) > 1 else 0.0,
                    "p05_ratio_delta": quantile(deltas, 0.05),
                    "p95_ratio_delta": quantile(deltas, 0.95),
                    "mean_hyp_ref_len_ratio": mean(hyp_ref),
                    "median_hyp_ref_len_ratio": median(hyp_ref),
                    "p05_hyp_ref_len_ratio": quantile(hyp_ref, 0.05),
                    "p95_hyp_ref_len_ratio": quantile(hyp_ref, 0.95),
                    "short_count": short_count,
                    "short_rate": short_count / len(vals) if vals else 0.0,
                    "long_count": long_count,
                    "long_rate": long_count / len(vals) if vals else 0.0,
                    "length_ratio_corr": pearson(ref_src, hyp_src),
                }
            )

    out_report = args.out_dir / f"length_ratios.report.{args.unit}.md"
    lines = [
        "# Length Ratio Report",
        "",
        f"Unit: {args.unit}. Compares hypothesis/source ratio to reference/source ratio.",
        f"Short threshold: hyp/ref < {args.short_threshold:.3f}; long threshold: hyp/ref > {args.long_threshold:.3f}.",
        "",
    ]
    for (corpus, pair), vals in sorted(grouped.items()):
        deltas = [v["ratio_delta"] for v in vals]
        hyp_ref = [v["hyp_ref_len_ratio"] for v in vals]
        short_count = sum(1 for v in vals if v["is_short"])
        long_count = sum(1 for v in vals if v["is_long"])
        lines.extend(
            [
                f"## {corpus} {pair}",
                "",
                f"- Mean hyp/ref length ratio: {mean(hyp_ref):.4f}; median: {median(hyp_ref):.4f}; p05/p95: {quantile(hyp_ref, 0.05):.4f}/{quantile(hyp_ref, 0.95):.4f}",
                f"- Mean ratio delta: {mean(deltas):.4f}; median: {median(deltas):.4f}; p05/p95: {quantile(deltas, 0.05):.4f}/{quantile(deltas, 0.95):.4f}",
                f"- Short sentences: {short_count}/{len(vals)} ({short_count / len(vals) if vals else 0.0:.2%}); long sentences: {long_count}/{len(vals)} ({long_count / len(vals) if vals else 0.0:.2%})",
                "",
            ]
        )
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = maybe_write_plots(args.out_dir, grouped, args.unit, args.max_plot_groups)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_report}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plots under {args.out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
