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
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
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


def maybe_write_plots(out_dir: Path, rows: list[dict[str, Any]], max_plot_groups: int) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skip plots: matplotlib unavailable: {e}")
        return []

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(str(row["corpus"]), str(row["pair"]))].append(row)
    groups = sorted(by_pair)
    if max_plot_groups and max_plot_groups > 0:
        groups = groups[:max_plot_groups]

    written: list[Path] = []
    labels = ["source_ratio", "target_ratio", "shared_ratio", "other_ratio", "neutral_ratio"]
    colors = ["#CC6677", "#4477AA", "#AA4499", "#DDCC77", "#999999"]
    for corpus, pair in groups:
        vals = by_pair[(corpus, pair)]
        means = [mean(float(r[x]) for r in vals) for x in labels]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        left = 0.0
        for label, value, color in zip(labels, means, colors, strict=True):
            ax.barh([0], [value], left=[left], label=label.replace("_ratio", ""), color=color)
            left += value
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Mean token share")
        ax.set_title(f"Language/script mix: {corpus} {pair}")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=5)
        fig.tight_layout()
        path = plot_dir / f"language_mix__{safe_name(corpus)}__{safe_name(pair)}__stacked.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist([float(r["source_ratio"]) + float(r["other_ratio"]) for r in vals], bins=30, color="#CC6677", alpha=0.85)
        ax.set_xlabel("source_ratio + other_ratio")
        ax.set_ylabel("Sentences")
        ax.set_title(f"Non-target script mass per sentence\n{corpus} {pair}")
        fig.tight_layout()
        path = plot_dir / f"language_mix__{safe_name(corpus)}__{safe_name(pair)}__nontarget_hist.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze source/target/other token mixing in hypotheses.")
    ap.add_argument("--hypotheses-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bad-ratio-threshold", type=float, default=0.05, help="Flag sentences with source+other ratio above this value.")
    ap.add_argument("--max-plot-groups", type=int, default=0, help="0 plots all corpus/pair groups.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.hypotheses_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "language_mix_ratio.csv"
    aggregate: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    detail_rows: list[dict[str, Any]] = []

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
                "nontarget_ratio",
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
            out_row = {
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
                    "nontarget_ratio": (counts["source"] + counts["other"]) / total if total else 0.0,
                }
            detail_rows.append(out_row)
            writer.writerow(out_row)

    out_summary = args.out_dir / "language_mix_ratio.by_pair.csv"
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "num",
                "token_total",
                "source_ratio",
                "target_ratio",
                "shared_ratio",
                "other_ratio",
                "neutral_ratio",
                "mean_nontarget_ratio",
                "median_nontarget_ratio",
                "p90_nontarget_ratio",
                "p95_nontarget_ratio",
                "bad_sentence_count",
                "bad_sentence_rate",
            ],
        )
        writer.writeheader()
        for (corpus, pair), counts in sorted(aggregate.items()):
            total = sum(counts.values())
            vals = [r for r in detail_rows if r["corpus"] == corpus and r["pair"] == pair]
            nontarget = [float(r["nontarget_ratio"]) for r in vals]
            bad_count = sum(1 for x in nontarget if x > args.bad_ratio_threshold)
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "num": len(vals),
                    "token_total": total,
                    "source_ratio": counts["source"] / total if total else 0.0,
                    "target_ratio": counts["target"] / total if total else 0.0,
                    "shared_ratio": counts["shared"] / total if total else 0.0,
                    "other_ratio": counts["other"] / total if total else 0.0,
                    "neutral_ratio": counts["neutral"] / total if total else 0.0,
                    "mean_nontarget_ratio": mean(nontarget) if nontarget else 0.0,
                    "median_nontarget_ratio": median(nontarget) if nontarget else 0.0,
                    "p90_nontarget_ratio": quantile(nontarget, 0.90),
                    "p95_nontarget_ratio": quantile(nontarget, 0.95),
                    "bad_sentence_count": bad_count,
                    "bad_sentence_rate": bad_count / len(nontarget) if nontarget else 0.0,
                }
            )

    out_report = args.out_dir / "language_mix_ratio.report.md"
    lines = [
        "# Language Mix Ratio Report",
        "",
        "Script-based source/target/other token mixing in hypotheses.",
        f"Bad sentence threshold: source_ratio + other_ratio > {args.bad_ratio_threshold:.3f}.",
        "Latin->Latin directions are ambiguous and appear mostly as shared-script.",
        "",
    ]
    for (corpus, pair), counts in sorted(aggregate.items()):
        total = sum(counts.values())
        vals = [r for r in detail_rows if r["corpus"] == corpus and r["pair"] == pair]
        nontarget = [float(r["nontarget_ratio"]) for r in vals]
        bad_count = sum(1 for x in nontarget if x > args.bad_ratio_threshold)
        lines.extend(
            [
                f"## {corpus} {pair}",
                "",
                f"- Token total: {total}",
                f"- Source/target/shared/other/neutral ratios: "
                f"{counts['source'] / total if total else 0.0:.2%} / "
                f"{counts['target'] / total if total else 0.0:.2%} / "
                f"{counts['shared'] / total if total else 0.0:.2%} / "
                f"{counts['other'] / total if total else 0.0:.2%} / "
                f"{counts['neutral'] / total if total else 0.0:.2%}",
                f"- Mean non-target ratio: {mean(nontarget) if nontarget else 0.0:.2%}; p95: {quantile(nontarget, 0.95):.2%}",
                f"- Bad sentence rate: {bad_count}/{len(nontarget)} ({bad_count / len(nontarget) if nontarget else 0.0:.2%})",
                "",
            ]
        )
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = maybe_write_plots(args.out_dir, detail_rows, args.max_plot_groups)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_report}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plots under {args.out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
