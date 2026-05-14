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
import math
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


def freq(counter: Counter[str], total: int, key: str) -> float:
    return counter.get(key, 0) / total if total else 0.0


def top_mass(counter: Counter[str], total: int, k: int) -> float:
    if not total:
        return 0.0
    return sum(v for _, v in counter.most_common(k)) / total


def entropy(counter: Counter[str], total: int) -> float:
    if not total:
        return 0.0
    h = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def js_divergence(
    left: Counter[str],
    left_total: int,
    right: Counter[str],
    right_total: int,
) -> float:
    vocab = set(left) | set(right)
    if not left_total or not right_total or not vocab:
        return 0.0
    js = 0.0
    for gram in vocab:
        p = left.get(gram, 0) / left_total
        q = right.get(gram, 0) / right_total
        m = 0.5 * (p + q)
        if p > 0:
            js += 0.5 * p * math.log2(p / m)
        if q > 0:
            js += 0.5 * q * math.log2(q / m)
    return js


def total_variation(
    left: Counter[str],
    left_total: int,
    right: Counter[str],
    right_total: int,
) -> float:
    vocab = set(left) | set(right)
    if not left_total or not right_total or not vocab:
        return 0.0
    return 0.5 * sum(abs(left.get(g, 0) / left_total - right.get(g, 0) / right_total) for g in vocab)


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return out.strip("_") or "unknown"


def write_distribution_comparison(
    *,
    out_dir: Path,
    test_counts: dict[tuple[str, str, str, str, int], Counter[str]],
    totals: Counter[tuple[str, str, str, str, int]],
    orders: list[int],
    top_k: int,
) -> tuple[Path, Path]:
    out_csv = out_dir / "target_ngram_distribution_compare.csv"
    out_md = out_dir / "target_ngram_distribution_report.md"
    groups = sorted({(corpus, pair, tgt_lang, n) for corpus, pair, tgt_lang, _side, n in test_counts})

    rows: list[dict[str, Any]] = []
    report_lines = [
        "# Target n-gram Distribution Comparison",
        "",
        "Compares model hypotheses against references for each corpus/language pair/n-gram order.",
        "Higher JS divergence and total variation indicate a larger distribution shift.",
        "",
    ]
    for corpus, pair, tgt_lang, n in groups:
        hyp_key = (corpus, pair, tgt_lang, "hypothesis", n)
        ref_key = (corpus, pair, tgt_lang, "reference", n)
        hyp = test_counts.get(hyp_key, Counter())
        ref = test_counts.get(ref_key, Counter())
        hyp_total = totals[hyp_key]
        ref_total = totals[ref_key]
        hyp_top = [g for g, _ in hyp.most_common(top_k)]
        ref_top = [g for g, _ in ref.most_common(top_k)]
        overlap = len(set(hyp_top) & set(ref_top))
        hyp_only = [g for g in hyp_top if g not in set(ref_top)][:10]
        ref_only = [g for g in ref_top if g not in set(hyp_top)][:10]
        row = {
            "corpus": corpus,
            "pair": pair,
            "tgt_lang": tgt_lang,
            "n": n,
            "hyp_total": hyp_total,
            "ref_total": ref_total,
            "hyp_vocab": len(hyp),
            "ref_vocab": len(ref),
            "vocab_overlap": len(set(hyp) & set(ref)),
            "top_k": top_k,
            "top_k_overlap": overlap,
            "top_k_overlap_ratio": overlap / top_k if top_k else 0.0,
            "hyp_top_k_mass": top_mass(hyp, hyp_total, top_k),
            "ref_top_k_mass": top_mass(ref, ref_total, top_k),
            "hyp_entropy": entropy(hyp, hyp_total),
            "ref_entropy": entropy(ref, ref_total),
            "js_divergence": js_divergence(hyp, hyp_total, ref, ref_total),
            "total_variation": total_variation(hyp, hyp_total, ref, ref_total),
            "hyp_only_top": "; ".join(hyp_only),
            "ref_only_top": "; ".join(ref_only),
        }
        rows.append(row)

        report_lines.extend(
            [
                f"## {corpus} {pair} n={n}",
                "",
                f"- JS divergence: {row['js_divergence']:.4f}",
                f"- Total variation: {row['total_variation']:.4f}",
                f"- Top-{top_k} overlap: {overlap}/{top_k} ({row['top_k_overlap_ratio']:.2%})",
                f"- Hypothesis top-{top_k} mass: {row['hyp_top_k_mass']:.2%}; reference top-{top_k} mass: {row['ref_top_k_mass']:.2%}",
                f"- Hypothesis-only top items: {row['hyp_only_top'] or 'NA'}",
                f"- Reference-only top items: {row['ref_only_top'] or 'NA'}",
                "",
            ]
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "corpus",
            "pair",
            "tgt_lang",
            "n",
            "hyp_total",
            "ref_total",
            "hyp_vocab",
            "ref_vocab",
            "vocab_overlap",
            "top_k",
            "top_k_overlap",
            "top_k_overlap_ratio",
            "hyp_top_k_mass",
            "ref_top_k_mass",
            "hyp_entropy",
            "ref_entropy",
            "js_divergence",
            "total_variation",
            "hyp_only_top",
            "ref_only_top",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return out_csv, out_md


def maybe_plot_distribution_comparison(
    *,
    out_dir: Path,
    test_counts: dict[tuple[str, str, str, str, int], Counter[str]],
    totals: Counter[tuple[str, str, str, str, int]],
    top_k: int,
    max_plots: int,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skip plots: matplotlib unavailable: {e}")
        return []

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({(corpus, pair, tgt_lang, n) for corpus, pair, tgt_lang, _side, n in test_counts})
    if max_plots and max_plots > 0:
        groups = groups[:max_plots]

    written: list[Path] = []
    for corpus, pair, tgt_lang, n in groups:
        hyp_key = (corpus, pair, tgt_lang, "hypothesis", n)
        ref_key = (corpus, pair, tgt_lang, "reference", n)
        hyp = test_counts.get(hyp_key, Counter())
        ref = test_counts.get(ref_key, Counter())
        hyp_total = totals[hyp_key]
        ref_total = totals[ref_key]
        if not hyp or not ref:
            continue

        selected: list[str] = []
        for gram, _ in (hyp + ref).most_common(top_k):
            if gram not in selected:
                selected.append(gram)
        selected = selected[:top_k]

        height = max(5.0, min(16.0, 0.32 * len(selected) + 2.0))
        fig, ax = plt.subplots(figsize=(11, height))
        ypos = list(range(len(selected)))
        hyp_vals = [freq(hyp, hyp_total, gram) for gram in selected]
        ref_vals = [freq(ref, ref_total, gram) for gram in selected]
        ax.barh([y - 0.18 for y in ypos], hyp_vals, height=0.35, label="hypothesis", color="#4477AA")
        ax.barh([y + 0.18 for y in ypos], ref_vals, height=0.35, label="reference", color="#CC6677")
        ax.set_yticks(ypos)
        ax.set_yticklabels(selected, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top target {n}-grams: {corpus} {pair}")
        ax.legend()
        fig.tight_layout()
        bar_path = plot_dir / f"{safe_name(corpus)}__{safe_name(pair)}__n{n}__top{top_k}.png"
        fig.savefig(bar_path, dpi=180)
        plt.close(fig)
        written.append(bar_path)

        vocab = sorted(set(hyp) | set(ref))
        x = [freq(ref, ref_total, gram) for gram in vocab]
        y = [freq(hyp, hyp_total, gram) for gram in vocab]
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.scatter(x, y, s=10, alpha=0.45, color="#228833")
        max_v = max(x + y) if x or y else 0.0
        ax.plot([0, max_v], [0, max_v], color="#666666", linewidth=1.0, linestyle="--")
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_xlabel("Reference frequency")
        ax.set_ylabel("Hypothesis frequency")
        ax.set_title(f"Hypothesis vs reference {n}-gram frequencies\n{corpus} {pair}")
        fig.tight_layout()
        scatter_path = plot_dir / f"{safe_name(corpus)}__{safe_name(pair)}__n{n}__scatter.png"
        fig.savefig(scatter_path, dpi=180)
        plt.close(fig)
        written.append(scatter_path)

    return written


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
    ap.add_argument("--compare-top-k", type=int, default=30, help="Top K used in distribution summary and plots.")
    ap.add_argument("--max-plot-groups", type=int, default=0, help="0 plots all corpus/pair/n groups.")
    ap.add_argument("--no-plots", action="store_true", help="Only write CSV/Markdown summaries.")
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

    compare_csv, compare_md = write_distribution_comparison(
        out_dir=args.out_dir,
        test_counts=test_counts,
        totals=totals,
        orders=orders,
        top_k=args.compare_top_k,
    )
    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = maybe_plot_distribution_comparison(
            out_dir=args.out_dir,
            test_counts=test_counts,
            totals=totals,
            top_k=args.compare_top_k,
            max_plots=args.max_plot_groups,
        )

    summary = {
        "hypotheses_jsonl": str(args.hypotheses_jsonl),
        "num_rows": len(rows),
        "orders": orders,
        "monolingual_jsonl": {k: [str(p) for p in v] for k, v in mono_paths.items()},
        "output_csv": str(out_csv),
        "distribution_compare_csv": str(compare_csv),
        "distribution_report_md": str(compare_md),
        "plot_count": len(plot_paths),
        "plot_dir": str(args.out_dir / "plots") if plot_paths else "",
        "tokenization_note": "zho/Han uses Han characters plus Latin/number tokens; Thai uses PyThaiNLP when installed; other languages use regex word tokens.",
    }
    (args.out_dir / "target_ngrams.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {compare_csv}")
    print(f"Wrote {compare_md}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plots under {args.out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
