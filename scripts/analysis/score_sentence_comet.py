#!/usr/bin/env python3
"""Score each hypothesis sentence with COMET-QE and reference-based COMET."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
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
    ap.add_argument("--comet-qe-model", default="Unbabel/wmt22-cometkiwi-da")
    ap.add_argument("--comet-encoder-model", type=Path, default=Path("models/xlm-roberta-large"))
    ap.add_argument("--comet-batch-size", type=int, default=8)
    ap.add_argument("--offline-eval-assets", action=argparse.BooleanOptionalAction, default=True)
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
    with out_jsonl.open("w", encoding="utf-8") as jf, out_csv.open("w", encoding="utf-8", newline="") as cf:
        fieldnames = ["corpus", "pair", "sample_id", "src_lang", "tgt_lang", "comet", "comet_qe"]
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
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "sample_id": row.get("sample_id", ""),
                    "src_lang": row.get("src_lang", ""),
                    "tgt_lang": row.get("tgt_lang", ""),
                    "comet": "" if comet is None else comet,
                    "comet_qe": "" if comet_qe is None else comet_qe,
                }
            )
            if comet is not None:
                groups[(str(corpus), str(pair))]["comet"].append(comet)
            if comet_qe is not None:
                groups[(str(corpus), str(pair))]["comet_qe"].append(comet_qe)

    out_summary = args.out_dir / "sentence_comet_scores.by_pair.csv"
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["corpus", "pair", "num_comet", "mean_comet", "num_comet_qe", "mean_comet_qe"])
        writer.writeheader()
        for (corpus, pair), vals in sorted(groups.items()):
            writer.writerow(
                {
                    "corpus": corpus,
                    "pair": pair,
                    "num_comet": len(vals.get("comet", [])),
                    "mean_comet": mean(vals["comet"]) if vals.get("comet") else "",
                    "num_comet_qe": len(vals.get("comet_qe", [])),
                    "mean_comet_qe": mean(vals["comet_qe"]) if vals.get("comet_qe") else "",
                }
            )

    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
