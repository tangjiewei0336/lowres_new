#!/usr/bin/env python3
"""Score each hypothesis sentence with COMET-QE and reference-based COMET."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run.run_eval import (  # noqa: E402
    configure_offline_transformers,
    load_comet_model,
    patch_comet_checkpoint_pretrained_model,
    prepare_comet_checkpoint,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return out.strip("_") or "unknown"


def score_stats(values: list[float], low_threshold: float) -> dict[str, Any]:
    if not values:
        return {
            "num": 0,
            "mean": "",
            "median": "",
            "std": "",
            "min": "",
            "p05": "",
            "p25": "",
            "p75": "",
            "p95": "",
            "max": "",
            "low_count": "",
            "low_rate": "",
        }
    low_count = sum(1 for v in values if v < low_threshold)
    return {
        "num": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values),
        "low_count": low_count,
        "low_rate": low_count / len(values),
    }


def maybe_write_plots(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_plot_groups: int,
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
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(str(row["corpus"]), str(row["pair"]))].append(row)
    groups = sorted(by_pair)
    if max_plot_groups and max_plot_groups > 0:
        groups = groups[:max_plot_groups]

    written: list[Path] = []
    for corpus, pair in groups:
        vals = by_pair[(corpus, pair)]
        comet = [float(r["comet"]) for r in vals if r.get("comet") not in ("", None)]
        comet_qe = [float(r["comet_qe"]) for r in vals if r.get("comet_qe") not in ("", None)]
        if comet:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.hist(comet, bins=35, color="#4477AA", alpha=0.85)
            ax.set_xlabel("COMET")
            ax.set_ylabel("Sentences")
            ax.set_title(f"COMET distribution: {corpus} {pair}")
            fig.tight_layout()
            path = plot_dir / f"comet__{safe_name(corpus)}__{safe_name(pair)}__hist.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(path)
        if comet_qe:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.hist(comet_qe, bins=35, color="#CC6677", alpha=0.85)
            ax.set_xlabel("COMET-QE")
            ax.set_ylabel("Sentences")
            ax.set_title(f"COMET-QE distribution: {corpus} {pair}")
            fig.tight_layout()
            path = plot_dir / f"comet_qe__{safe_name(corpus)}__{safe_name(pair)}__hist.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(path)
        paired = [(float(r["comet"]), float(r["comet_qe"])) for r in vals if r.get("comet") not in ("", None) and r.get("comet_qe") not in ("", None)]
        if paired:
            fig, ax = plt.subplots(figsize=(6.2, 6.0))
            ax.scatter([x for x, _ in paired], [y for _, y in paired], s=12, alpha=0.55, color="#228833")
            ax.set_xlabel("COMET")
            ax.set_ylabel("COMET-QE")
            ax.set_title(f"COMET vs COMET-QE: {corpus} {pair}")
            fig.tight_layout()
            path = plot_dir / f"comet_vs_qe__{safe_name(corpus)}__{safe_name(pair)}__scatter.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(path)
    return written


def score_model(
    rows: list[dict[str, Any]],
    *,
    model_arg: str,
    run_dir: Path,
    batch_size: int,
    encoder_model: Path | None,
    needs_ref: bool,
) -> list[float] | None:
    if str(model_arg).lower() in {"none", "off", "disabled", "disable"}:
        return None
    ckpt, torch_mod, load_fn = prepare_comet_checkpoint(model_arg, run_dir, encoder_path=encoder_model)
    if not ckpt or torch_mod is None or load_fn is None:
        return None
    gpus = 1 if torch_mod.cuda.is_available() else 0
    ckpt = patch_comet_checkpoint_pretrained_model(ckpt, encoder_model)
    model = load_comet_model(load_fn, ckpt)
    data = []
    for row in rows:
        item = {"src": row["source_text"], "mt": row["hypothesis"]}
        if needs_ref:
            item["ref"] = row["reference_text"]
        data.append(item)
    pred = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = pred.get("scores", []) if isinstance(pred, dict) else getattr(pred, "scores", [])
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError(f"COMET output length mismatch: got {len(scores)} scores for {len(rows)} rows")
    return [float(x) for x in scores]


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-sentence COMET-QE and COMET scoring for hypotheses.jsonl.")
    ap.add_argument("--hypotheses-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--comet-model", default="models/Unbabel_wmt22-comet-da")
    ap.add_argument("--comet-qe-model", default="models/Unbabel_wmt22-cometkiwi-da")
    ap.add_argument("--comet-encoder-model", type=Path, default=Path("models/xlm-roberta-large"))
    ap.add_argument("--comet-batch-size", type=int, default=8)
    ap.add_argument("--offline-eval-assets", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--low-comet-threshold", type=float, default=0.70)
    ap.add_argument("--low-comet-qe-threshold", type=float, default=0.50)
    ap.add_argument("--max-plot-groups", type=int, default=0, help="0 plots all corpus/pair groups.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.hypotheses_jsonl)
    for idx, row in enumerate(rows, start=1):
        for key in ("source_text", "hypothesis", "reference_text"):
            if key not in row:
                raise SystemExit(f"Row {idx} missing {key}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder = args.comet_encoder_model if args.comet_encoder_model.is_dir() else None
    configure_offline_transformers(encoder, bool(args.offline_eval_assets))

    comet_scores = score_model(
        rows,
        model_arg=args.comet_model,
        run_dir=args.out_dir / "comet_ref_model",
        batch_size=args.comet_batch_size,
        encoder_model=encoder,
        needs_ref=True,
    )
    comet_qe_scores = score_model(
        rows,
        model_arg=args.comet_qe_model,
        run_dir=args.out_dir / "comet_qe_model",
        batch_size=args.comet_batch_size,
        encoder_model=encoder,
        needs_ref=False,
    )

    out_jsonl = args.out_dir / "sentence_comet_scores.jsonl"
    out_csv = args.out_dir / "sentence_comet_scores.csv"
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    detail_rows: list[dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as jf, out_csv.open("w", encoding="utf-8", newline="") as cf:
        fieldnames = ["corpus", "pair", "sample_id", "src_lang", "tgt_lang", "comet", "comet_qe", "is_low_comet", "is_low_comet_qe"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            corpus = row.get("eval_corpus") or row.get("dataset") or ""
            pair = row.get("eval_pair") or f"{row.get('src_lang', '')}->{row.get('tgt_lang', '')}"
            comet = comet_scores[i] if comet_scores is not None else None
            comet_qe = comet_qe_scores[i] if comet_qe_scores is not None else None
            enriched = dict(row)
            enriched["comet"] = comet
            enriched["comet_qe"] = comet_qe
            jf.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            out_row = {
                    "corpus": corpus,
                    "pair": pair,
                    "sample_id": row.get("sample_id", ""),
                    "src_lang": row.get("src_lang", ""),
                    "tgt_lang": row.get("tgt_lang", ""),
                    "comet": "" if comet is None else comet,
                    "comet_qe": "" if comet_qe is None else comet_qe,
                    "is_low_comet": "" if comet is None else comet < args.low_comet_threshold,
                    "is_low_comet_qe": "" if comet_qe is None else comet_qe < args.low_comet_qe_threshold,
                }
            detail_rows.append(out_row)
            writer.writerow(out_row)
            if comet is not None:
                groups[(str(corpus), str(pair))]["comet"].append(comet)
            if comet_qe is not None:
                groups[(str(corpus), str(pair))]["comet_qe"].append(comet_qe)

    out_summary = args.out_dir / "sentence_comet_scores.by_pair.csv"
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "num_comet",
                "mean_comet",
                "median_comet",
                "std_comet",
                "min_comet",
                "p05_comet",
                "p25_comet",
                "p75_comet",
                "p95_comet",
                "max_comet",
                "low_comet_count",
                "low_comet_rate",
                "num_comet_qe",
                "mean_comet_qe",
                "median_comet_qe",
                "std_comet_qe",
                "min_comet_qe",
                "p05_comet_qe",
                "p25_comet_qe",
                "p75_comet_qe",
                "p95_comet_qe",
                "max_comet_qe",
                "low_comet_qe_count",
                "low_comet_qe_rate",
                "comet_qe_corr",
            ],
        )
        writer.writeheader()
        for (corpus, pair), vals in sorted(groups.items()):
            comet = vals.get("comet", [])
            comet_qe = vals.get("comet_qe", [])
            comet_stats = score_stats(comet, args.low_comet_threshold)
            qe_stats = score_stats(comet_qe, args.low_comet_qe_threshold)
            pair_rows = [r for r in detail_rows if r["corpus"] == corpus and r["pair"] == pair]
            paired = [(float(r["comet"]), float(r["comet_qe"])) for r in pair_rows if r.get("comet") not in ("", None) and r.get("comet_qe") not in ("", None)]
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "num_comet": comet_stats["num"],
                    "mean_comet": comet_stats["mean"],
                    "median_comet": comet_stats["median"],
                    "std_comet": comet_stats["std"],
                    "min_comet": comet_stats["min"],
                    "p05_comet": comet_stats["p05"],
                    "p25_comet": comet_stats["p25"],
                    "p75_comet": comet_stats["p75"],
                    "p95_comet": comet_stats["p95"],
                    "max_comet": comet_stats["max"],
                    "low_comet_count": comet_stats["low_count"],
                    "low_comet_rate": comet_stats["low_rate"],
                    "num_comet_qe": qe_stats["num"],
                    "mean_comet_qe": qe_stats["mean"],
                    "median_comet_qe": qe_stats["median"],
                    "std_comet_qe": qe_stats["std"],
                    "min_comet_qe": qe_stats["min"],
                    "p05_comet_qe": qe_stats["p05"],
                    "p25_comet_qe": qe_stats["p25"],
                    "p75_comet_qe": qe_stats["p75"],
                    "p95_comet_qe": qe_stats["p95"],
                    "max_comet_qe": qe_stats["max"],
                    "low_comet_qe_count": qe_stats["low_count"],
                    "low_comet_qe_rate": qe_stats["low_rate"],
                    "comet_qe_corr": pearson([x for x, _ in paired], [y for _, y in paired]) if paired else "",
                }
            )

    out_report = args.out_dir / "sentence_comet_scores.report.md"
    lines = [
        "# Sentence COMET Score Report",
        "",
        f"Low COMET threshold: {args.low_comet_threshold:.3f}; low COMET-QE threshold: {args.low_comet_qe_threshold:.3f}.",
        "",
    ]
    for (corpus, pair), vals in sorted(groups.items()):
        comet = vals.get("comet", [])
        comet_qe = vals.get("comet_qe", [])
        comet_stats = score_stats(comet, args.low_comet_threshold)
        qe_stats = score_stats(comet_qe, args.low_comet_qe_threshold)
        pair_rows = [r for r in detail_rows if r["corpus"] == corpus and r["pair"] == pair]
        paired = [(float(r["comet"]), float(r["comet_qe"])) for r in pair_rows if r.get("comet") not in ("", None) and r.get("comet_qe") not in ("", None)]
        lines.extend(
            [
                f"## {corpus} {pair}",
                "",
                f"- COMET mean/median/p05: {comet_stats['mean'] if comet else 'NA'} / {comet_stats['median'] if comet else 'NA'} / {comet_stats['p05'] if comet else 'NA'}",
                f"- Low COMET rate: {comet_stats['low_count'] if comet else 'NA'}/{comet_stats['num'] if comet else 'NA'} ({comet_stats['low_rate'] if comet else 'NA'})",
                f"- COMET-QE mean/median/p05: {qe_stats['mean'] if comet_qe else 'NA'} / {qe_stats['median'] if comet_qe else 'NA'} / {qe_stats['p05'] if comet_qe else 'NA'}",
                f"- Low COMET-QE rate: {qe_stats['low_count'] if comet_qe else 'NA'}/{qe_stats['num'] if comet_qe else 'NA'} ({qe_stats['low_rate'] if comet_qe else 'NA'})",
                f"- COMET vs COMET-QE Pearson correlation: {pearson([x for x, _ in paired], [y for _, y in paired]) if paired else 'NA'}",
                "",
            ]
        )
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = maybe_write_plots(args.out_dir, detail_rows, max_plot_groups=args.max_plot_groups)

    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_report}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plots under {args.out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
