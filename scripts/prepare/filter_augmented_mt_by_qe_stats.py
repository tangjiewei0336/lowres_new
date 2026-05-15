#!/usr/bin/env python3
"""
Filter augmented MT data with COMET-QE plus statistical quality checks.

Typical use: compress each 100k augmented FineWeb pair file to 50k high-quality
sentence pairs.

Example:
  conda run -n lowres python scripts/prepare/filter_augmented_mt_by_qe_stats.py \
    --input-dir training/data/multilingual/fineweb2_synth \
    --out-dir training/data/multilingual/fineweb2_synth_qe50k \
    --target-size 50000 \
    --qe-model models/Unbabel_wmt22-cometkiwi-da

Dry/statistics-only smoke test:
  conda run -n lowres python scripts/prepare/filter_augmented_mt_by_qe_stats.py \
    --input training/data/multilingual/fineweb2_synth/fineweb_synth_eng_Latn__zho_Hans.jsonl \
    --out-dir /tmp/qe_filter_smoke \
    --target-size 100 \
    --qe-model none
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run.run_eval import (  # noqa: E402
    configure_offline_transformers,
    load_comet_model,
    patch_comet_checkpoint_pretrained_model,
    prepare_comet_checkpoint,
)


PAIR_RE = re.compile(r"(.+?)__([^./]+)$")
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


@dataclass
class ScoredRow:
    idx: int
    row: dict[str, Any]
    src: str
    tgt: str
    src_lang: str
    tgt_lang: str
    src_len: int
    tgt_len: int
    len_ratio: float
    target_script_ratio: float
    source_script_in_target_ratio: float
    qe_score: float | None
    stats_penalty: float
    composite_score: float
    hard_ok: bool
    reject_reasons: list[str]


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_text_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    src = row.get("input") or row.get("source_text") or row.get("src") or row.get("source")
    tgt = row.get("output") or row.get("target_text") or row.get("tgt") or row.get("target")
    if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
        return src.strip(), tgt.strip()
    return None


def infer_pair_from_path(path: Path, prefix: str) -> tuple[str, str]:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    match = PAIR_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot infer src/tgt from filename: {path.name}")
    return match.group(1), match.group(2)


def lang_scripts(lang: str) -> set[str]:
    return set(LANG_SCRIPTS.get((lang or "").split("_", 1)[0], set()))


def char_script(ch: str) -> str | None:
    code = ord(ch)
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "Han"
    if 0x0E00 <= code <= 0x0E7F:
        return "Thai"
    name = ""
    try:
        import unicodedata

        name = unicodedata.name(ch)
    except ValueError:
        return None
    if "LATIN" in name:
        return "Latin"
    return None


def script_ratios(text: str, src_lang: str, tgt_lang: str) -> tuple[float, float]:
    src_scripts = lang_scripts(src_lang)
    tgt_scripts = lang_scripts(tgt_lang)
    script_chars = 0
    target_chars = 0
    source_chars = 0
    for ch in text:
        script = char_script(ch)
        if script is None:
            continue
        script_chars += 1
        if script in tgt_scripts:
            target_chars += 1
        if script in src_scripts and script not in tgt_scripts:
            source_chars += 1
    if script_chars == 0:
        return 0.0, 0.0
    return target_chars / script_chars, source_chars / script_chars


def token_len(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def normalize_scores(values: list[float | None]) -> list[float]:
    present = [v for v in values if v is not None]
    if not present:
        return [0.0 for _ in values]
    lo = min(present)
    hi = max(present)
    if hi == lo:
        return [1.0 if v is not None else 0.0 for v in values]
    return [((v - lo) / (hi - lo)) if v is not None else 0.0 for v in values]


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


def score_qe(
    rows: list[dict[str, str]],
    *,
    qe_model: str,
    out_dir: Path,
    batch_size: int,
    encoder_model: Path | None,
) -> list[float | None]:
    if str(qe_model).lower() in {"none", "off", "disabled", "disable"}:
        return [None for _ in rows]
    ckpt, torch_mod, load_fn = prepare_comet_checkpoint(qe_model, out_dir, encoder_path=encoder_model)
    if not ckpt or torch_mod is None or load_fn is None:
        raise RuntimeError(f"Could not load COMET-QE model: {qe_model}")
    gpus = 1 if torch_mod.cuda.is_available() else 0
    ckpt = patch_comet_checkpoint_pretrained_model(ckpt, encoder_model)
    model = load_comet_model(load_fn, ckpt)
    data = [{"src": r["src"], "mt": r["mt"]} for r in rows]
    pred = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = pred.get("scores", []) if isinstance(pred, dict) else getattr(pred, "scores", [])
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError(f"COMET-QE output length mismatch: got {len(scores)} for {len(rows)}")
    return [float(x) for x in scores]


def build_scored_rows(
    *,
    rows: list[dict[str, Any]],
    src_lang: str,
    tgt_lang: str,
    qe_scores: list[float | None],
    min_src_len: int,
    min_tgt_len: int,
    min_len_ratio: float,
    max_len_ratio: float,
    min_target_script_ratio: float,
    max_source_script_ratio: float,
    qe_weight: float,
    stats_weight: float,
) -> list[ScoredRow]:
    normalized_qe = normalize_scores(qe_scores)
    scored: list[ScoredRow] = []
    for idx, (row, qe, qe_norm) in enumerate(zip(rows, qe_scores, normalized_qe, strict=True)):
        pair = resolve_text_pair(row)
        if pair is None:
            src = ""
            tgt = ""
        else:
            src, tgt = pair
        src_len = token_len(src)
        tgt_len = token_len(tgt)
        len_ratio = tgt_len / src_len if src_len else 0.0
        target_script_ratio, source_script_ratio = script_ratios(tgt, src_lang, tgt_lang)
        reasons: list[str] = []
        penalty = 0.0
        if not src or not tgt:
            reasons.append("empty_text")
            penalty += 1.0
        if src_len < min_src_len:
            reasons.append("short_source")
            penalty += 0.5
        if tgt_len < min_tgt_len:
            reasons.append("short_target")
            penalty += 0.5
        if len_ratio < min_len_ratio:
            reasons.append("low_len_ratio")
            penalty += min(1.0, min_len_ratio - len_ratio)
        if len_ratio > max_len_ratio:
            reasons.append("high_len_ratio")
            penalty += min(1.0, (len_ratio - max_len_ratio) / max_len_ratio)
        if target_script_ratio < min_target_script_ratio:
            reasons.append("low_target_script_ratio")
            penalty += min(1.0, min_target_script_ratio - target_script_ratio)
        if source_script_ratio > max_source_script_ratio:
            reasons.append("source_script_leakage")
            penalty += min(1.0, source_script_ratio - max_source_script_ratio)
        hard_ok = not reasons
        stats_score = max(0.0, 1.0 - penalty)
        composite = qe_weight * qe_norm + stats_weight * stats_score
        scored.append(
            ScoredRow(
                idx=idx,
                row=row,
                src=src,
                tgt=tgt,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_len=src_len,
                tgt_len=tgt_len,
                len_ratio=len_ratio,
                target_script_ratio=target_script_ratio,
                source_script_in_target_ratio=source_script_ratio,
                qe_score=qe,
                stats_penalty=penalty,
                composite_score=composite,
                hard_ok=hard_ok,
                reject_reasons=reasons,
            )
        )
    return scored


def select_rows(
    scored: list[ScoredRow],
    *,
    target_size: int,
    dedupe_pair: bool,
    dedupe_source: bool,
    allow_fill_from_rejected: bool,
    write_order: str,
) -> list[ScoredRow]:
    ranked = sorted(
        scored,
        key=lambda x: (
            x.hard_ok,
            x.composite_score,
            x.qe_score if x.qe_score is not None else float("-inf"),
            -x.stats_penalty,
        ),
        reverse=True,
    )
    selected: list[ScoredRow] = []
    seen_pair: set[str] = set()
    seen_src: set[str] = set()

    def try_add(item: ScoredRow, *, require_hard_ok: bool) -> bool:
        if require_hard_ok and not item.hard_ok:
            return False
        pair_hash = sha1_text(item.src + "\n" + item.tgt)
        src_hash = sha1_text(item.src)
        if dedupe_pair and pair_hash in seen_pair:
            return False
        if dedupe_source and src_hash in seen_src:
            return False
        seen_pair.add(pair_hash)
        seen_src.add(src_hash)
        selected.append(item)
        return True

    for item in ranked:
        if len(selected) >= target_size:
            break
        try_add(item, require_hard_ok=True)

    if allow_fill_from_rejected and len(selected) < target_size:
        for item in ranked:
            if len(selected) >= target_size:
                break
            try_add(item, require_hard_ok=False)

    if write_order == "original":
        selected.sort(key=lambda x: x.idx)
    return selected


def add_filter_meta(item: ScoredRow) -> dict[str, Any]:
    out = dict(item.row)
    meta = dict(out.get("meta") or {})
    meta["qe_stats_filter"] = {
        "selected": True,
        "qe_score": item.qe_score,
        "composite_score": item.composite_score,
        "stats_penalty": item.stats_penalty,
        "src_len": item.src_len,
        "tgt_len": item.tgt_len,
        "len_ratio": item.len_ratio,
        "target_script_ratio": item.target_script_ratio,
        "source_script_in_target_ratio": item.source_script_in_target_ratio,
        "hard_ok": item.hard_ok,
        "reject_reasons": item.reject_reasons,
    }
    out["meta"] = meta
    return out


def write_score_csv(path: Path, scored: list[ScoredRow], selected_ids: set[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "selected",
                "src_lang",
                "tgt_lang",
                "qe_score",
                "composite_score",
                "stats_penalty",
                "hard_ok",
                "reject_reasons",
                "src_len",
                "tgt_len",
                "len_ratio",
                "target_script_ratio",
                "source_script_in_target_ratio",
                "source_preview",
                "target_preview",
            ],
        )
        writer.writeheader()
        for item in scored:
            writer.writerow(
                {
                    "idx": item.idx,
                    "selected": item.idx in selected_ids,
                    "src_lang": item.src_lang,
                    "tgt_lang": item.tgt_lang,
                    "qe_score": "" if item.qe_score is None else item.qe_score,
                    "composite_score": item.composite_score,
                    "stats_penalty": item.stats_penalty,
                    "hard_ok": item.hard_ok,
                    "reject_reasons": ";".join(item.reject_reasons),
                    "src_len": item.src_len,
                    "tgt_len": item.tgt_len,
                    "len_ratio": item.len_ratio,
                    "target_script_ratio": item.target_script_ratio,
                    "source_script_in_target_ratio": item.source_script_in_target_ratio,
                    "source_preview": item.src[:120],
                    "target_preview": item.tgt[:120],
                }
            )


def summarize(path: Path, input_path: Path, scored: list[ScoredRow], selected: list[ScoredRow]) -> dict[str, Any]:
    qe_values = [x.qe_score for x in scored if x.qe_score is not None]
    selected_qe = [x.qe_score for x in selected if x.qe_score is not None]
    reason_counts = Counter(reason for x in scored for reason in x.reject_reasons)
    selected_ids = {x.idx for x in selected}
    rejected = [x for x in scored if x.idx not in selected_ids]
    summary = {
        "input_path": str(input_path),
        "output_path": str(path),
        "input_rows": len(scored),
        "selected_rows": len(selected),
        "hard_ok_rows": sum(1 for x in scored if x.hard_ok),
        "filled_from_rejected": sum(1 for x in selected if not x.hard_ok),
        "mean_qe_all": mean(qe_values) if qe_values else None,
        "mean_qe_selected": mean(selected_qe) if selected_qe else None,
        "median_qe_all": median(qe_values) if qe_values else None,
        "median_qe_selected": median(selected_qe) if selected_qe else None,
        "p05_qe_all": quantile(qe_values, 0.05) if qe_values else None,
        "p05_qe_selected": quantile(selected_qe, 0.05) if selected_qe else None,
        "mean_len_ratio_selected": mean([x.len_ratio for x in selected]) if selected else None,
        "mean_target_script_ratio_selected": mean([x.target_script_ratio for x in selected]) if selected else None,
        "reject_reason_counts": dict(reason_counts),
        "top_rejected_by_score": [
            {
                "idx": x.idx,
                "qe_score": x.qe_score,
                "composite_score": x.composite_score,
                "reject_reasons": x.reject_reasons,
                "source_preview": x.src[:120],
                "target_preview": x.tgt[:120],
            }
            for x in sorted(rejected, key=lambda y: y.composite_score, reverse=True)[:20]
        ],
    }
    return summary


def output_path_for(input_path: Path, input_root: Path, out_dir: Path, suffix: str) -> Path:
    try:
        rel = input_path.relative_to(input_root)
    except ValueError:
        rel = Path(input_path.name)
    return out_dir / rel.with_name(f"{rel.stem}{suffix}.jsonl")


def process_one(
    *,
    input_path: Path,
    input_root: Path,
    out_dir: Path,
    filename_prefix: str,
    target_size: int,
    suffix: str,
    qe_model: str,
    qe_batch_size: int,
    encoder_model: Path | None,
    min_src_len: int,
    min_tgt_len: int,
    min_len_ratio: float,
    max_len_ratio: float,
    min_target_script_ratio: float,
    max_source_script_ratio: float,
    qe_weight: float,
    stats_weight: float,
    dedupe_pair: bool,
    dedupe_source: bool,
    allow_fill_from_rejected: bool,
    write_order: str,
    overwrite: bool,
) -> dict[str, Any]:
    src_lang, tgt_lang = infer_pair_from_path(input_path, filename_prefix)
    out_path = output_path_for(input_path, input_root, out_dir, suffix)
    score_csv = out_path.with_suffix(".scores.csv")
    summary_json = out_path.with_suffix(".summary.json")
    if out_path.exists() and not overwrite:
        print(f"skip existing: {out_path}")
        return json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.is_file() else {"output_path": str(out_path), "skipped": True}

    rows_all = read_jsonl(input_path)
    usable_rows: list[dict[str, Any]] = []
    qe_inputs: list[dict[str, str]] = []
    for row in rows_all:
        pair = resolve_text_pair(row)
        if pair is None:
            usable_rows.append(row)
            qe_inputs.append({"src": "", "mt": ""})
        else:
            usable_rows.append(row)
            qe_inputs.append({"src": pair[0], "mt": pair[1]})

    qe_scores = score_qe(
        qe_inputs,
        qe_model=qe_model,
        out_dir=out_dir / "comet_qe_model",
        batch_size=qe_batch_size,
        encoder_model=encoder_model,
    )
    scored = build_scored_rows(
        rows=usable_rows,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        qe_scores=qe_scores,
        min_src_len=min_src_len,
        min_tgt_len=min_tgt_len,
        min_len_ratio=min_len_ratio,
        max_len_ratio=max_len_ratio,
        min_target_script_ratio=min_target_script_ratio,
        max_source_script_ratio=max_source_script_ratio,
        qe_weight=qe_weight,
        stats_weight=stats_weight,
    )
    n = min(target_size, len(scored)) if target_size and target_size > 0 else len(scored)
    selected = select_rows(
        scored,
        target_size=n,
        dedupe_pair=dedupe_pair,
        dedupe_source=dedupe_source,
        allow_fill_from_rejected=allow_fill_from_rejected,
        write_order=write_order,
    )
    selected_ids = {x.idx for x in selected}
    write_jsonl(out_path, [add_filter_meta(x) for x in selected])
    write_score_csv(score_csv, scored, selected_ids)
    summary = summarize(out_path, input_path, scored, selected)
    summary["src_lang"] = src_lang
    summary["tgt_lang"] = tgt_lang
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{input_path} -> {out_path} selected={len(selected)}/{len(scored)}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter augmented MT JSONL by COMET-QE and statistical checks.")
    ap.add_argument("--input", type=Path, action="append", dest="inputs", default=[])
    ap.add_argument("--input-dir", type=Path, default=root() / "training" / "data" / "multilingual" / "fineweb2_synth")
    ap.add_argument("--glob", default="fineweb_synth_*__*.jsonl")
    ap.add_argument("--filename-prefix", default="fineweb_synth_")
    ap.add_argument("--out-dir", type=Path, default=root() / "training" / "data" / "multilingual" / "fineweb2_synth_qe50k")
    ap.add_argument("--suffix", default="_qe50k")
    ap.add_argument("--target-size", type=int, default=50000)
    ap.add_argument("--qe-model", default="models/Unbabel_wmt22-cometkiwi-da")
    ap.add_argument("--qe-batch-size", type=int, default=16)
    ap.add_argument("--comet-encoder-model", type=Path, default=root() / "models" / "xlm-roberta-large")
    ap.add_argument("--offline-eval-assets", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-src-len", type=int, default=3)
    ap.add_argument("--min-tgt-len", type=int, default=1)
    ap.add_argument("--min-len-ratio", type=float, default=0.30)
    ap.add_argument("--max-len-ratio", type=float, default=3.00)
    ap.add_argument("--min-target-script-ratio", type=float, default=0.50)
    ap.add_argument("--max-source-script-ratio", type=float, default=0.25)
    ap.add_argument("--qe-weight", type=float, default=0.80)
    ap.add_argument("--stats-weight", type=float, default=0.20)
    ap.add_argument("--dedupe-pair", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dedupe-source", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--allow-fill-from-rejected", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--write-order", choices=["score", "original"], default="score")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inputs = args.inputs or sorted(
        p for p in args.input_dir.glob(args.glob)
        if p.is_file() and "previews" not in p.parts
    )
    if not inputs:
        raise SystemExit(f"No input files found under {args.input_dir} with glob {args.glob}")

    encoder = args.comet_encoder_model if args.comet_encoder_model.is_dir() else None
    configure_offline_transformers(encoder, bool(args.offline_eval_assets))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_root = args.input_dir.resolve()

    summaries: list[dict[str, Any]] = []
    for input_path in tqdm(inputs, desc="files"):
        summaries.append(
            process_one(
                input_path=input_path,
                input_root=input_root,
                out_dir=args.out_dir,
                filename_prefix=args.filename_prefix,
                target_size=args.target_size,
                suffix=args.suffix,
                qe_model=args.qe_model,
                qe_batch_size=args.qe_batch_size,
                encoder_model=encoder,
                min_src_len=args.min_src_len,
                min_tgt_len=args.min_tgt_len,
                min_len_ratio=args.min_len_ratio,
                max_len_ratio=args.max_len_ratio,
                min_target_script_ratio=args.min_target_script_ratio,
                max_source_script_ratio=args.max_source_script_ratio,
                qe_weight=args.qe_weight,
                stats_weight=args.stats_weight,
                dedupe_pair=bool(args.dedupe_pair),
                dedupe_source=bool(args.dedupe_source),
                allow_fill_from_rejected=bool(args.allow_fill_from_rejected),
                write_order=args.write_order,
                overwrite=bool(args.overwrite),
            )
        )

    summary_path = args.out_dir / "filter_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.out_dir / "filter_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "src_lang",
            "tgt_lang",
            "input_rows",
            "selected_rows",
            "hard_ok_rows",
            "filled_from_rejected",
            "mean_qe_all",
            "mean_qe_selected",
            "p05_qe_all",
            "p05_qe_selected",
            "mean_len_ratio_selected",
            "mean_target_script_ratio_selected",
            "input_path",
            "output_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"summary: {summary_path}")
    print(f"summary_csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
