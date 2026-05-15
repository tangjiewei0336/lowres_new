#!/usr/bin/env python3
"""
根据 analyze_length_ratios.py 输出的逐句 CSV（或 hypotheses.jsonl）生成 mix_hypothesis_candidates 用的配置：
对每个语向（或 corpus + 语向），用 ref_src_ratio 的 mean ± sigma * pstdev 界定 min_len_ratio / max_len_ratio，
裁剪到 [--min-floor, --max-cap]，pstdev≈0 时用 --zero-std-half-span 保证带宽。

analyze_length_ratios 示例：
  conda run -n lowres python scripts/analysis/analyze_length_ratios.py \\
    --hypotheses-jsonl eval_multilingual/foo/hypotheses.jsonl \\
    --out-dir /tmp/ratio_stats --unit token --no-plots

用统计 CSV 生成带分语向 entries 的配置：
  conda run -n lowres python scripts/analysis/build_mix_len_ratio_config.py \\
    --from-ratios-csv /tmp/ratio_stats/length_ratios.token.csv \\
    --sigma 2 --output configs/mix_from_stats.json

仅 eval_pair（不区分 corpus）：--group-by pair

只用与 mix_hypothesis_candidates 相同 TOKEN（非 analyze 语种分词）：--hypotheses-jsonl …

合并进已有配置文件：--merge-into-existing configs/mix_hypothesis_candidates.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pstdev(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def fmean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0


def clamp_span(
    lo: float,
    hi: float,
    mu: float,
    *,
    pstdev_obs: float,
    min_floor: float,
    max_cap: float,
    zero_half_span: float,
    min_bandwidth: float,
) -> tuple[float, float]:
    if pstdev_obs <= 0:
        lo, hi = mu - zero_half_span, mu + zero_half_span
    if hi - lo < min_bandwidth:
        pad = (min_bandwidth - (hi - lo)) / 2.0
        lo -= pad
        hi += pad
    lo = max(lo, min_floor)
    hi = min(hi, max_cap)
    if hi <= lo:
        hi = lo + max(1e-4, min_bandwidth / 10.0)
    return lo, hi


def _mix_token_len() -> Any:
    import sys

    inf_dir = Path(__file__).resolve().parents[1] / "inference"
    if str(inf_dir) not in sys.path:
        sys.path.insert(0, str(inf_dir))
    import mix_hypothesis_candidates as mmc  # noqa: PLC0415

    return mmc.token_len


def collect_ratio_rows_csv(path: Path) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ref_src_ratio" not in reader.fieldnames:
            raise ValueError(f"CSV 缺少 ref_src_ratio 列: {path}")
        for row in reader:
            try:
                ratio = float(row["ref_src_ratio"])
            except (TypeError, ValueError):
                continue
            corpus = (row.get("corpus") or "").strip()
            pair = (row.get("pair") or "").strip()
            if not pair:
                continue
            out.append((corpus, pair, ratio))
    return out


def collect_ratio_rows_hypo(path: Path) -> list[tuple[str, str, float]]:
    token_len = _mix_token_len()
    out: list[tuple[str, str, float]] = []
    for row in read_jsonl(path):
        corpus = str(row.get("eval_corpus") or row.get("dataset") or "").strip()
        pair = str(row.get("eval_pair") or f"{row.get('src_lang', '')}->{row.get('tgt_lang', '')}").strip()
        if not pair:
            continue
        src = str(row.get("source_text", ""))
        ref = str(row.get("reference_text", ""))
        sl = token_len(src)
        if sl <= 0:
            continue
        out.append((corpus, pair, token_len(ref) / sl))
    return out


def aggregate_bounds(
    rows: list[tuple[str, str, float]],
    *,
    group_by: str,
    sigma_mult: float,
    min_floor: float,
    max_cap: float,
    zero_half_span: float,
    min_bandwidth: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """返回 (pairs, corpus_pairs)。

    corpus_pairs 仅在 group_by corpus_pair 且 corpus 非空时填入；否则会写入 pairs。"""
    if group_by == "pair":
        buckets: defaultdict[str, list[float]] = defaultdict(list)
        for corpus, pair, ratio in rows:
            _ = corpus
            buckets[pair].append(ratio)
    else:
        cp_buckets: defaultdict[str, list[float]] = defaultdict(list)
        p_buckets: defaultdict[str, list[float]] = defaultdict(list)
        for corpus, pair, ratio in rows:
            if corpus:
                cp_buckets[f"{corpus}|{pair}"].append(ratio)
            else:
                p_buckets[pair].append(ratio)
        buckets_pair = dict(p_buckets)
        buckets_cp = dict(cp_buckets)
        pairs_out_p: dict[str, dict[str, float]] = {}
        corpus_pairs_out_p: dict[str, dict[str, float]] = {}
        all_keys_pairs = sorted(buckets_pair.keys())
        for pk in all_keys_pairs:
            bounds = _one_bucket_bounds(
                buckets_pair[pk],
                sigma_mult=sigma_mult,
                min_floor=min_floor,
                max_cap=max_cap,
                zero_half_span=zero_half_span,
                min_bandwidth=min_bandwidth,
            )
            if bounds:
                pairs_out_p[pk] = bounds
        for ck in sorted(buckets_cp):
            bounds = _one_bucket_bounds(
                buckets_cp[ck],
                sigma_mult=sigma_mult,
                min_floor=min_floor,
                max_cap=max_cap,
                zero_half_span=zero_half_span,
                min_bandwidth=min_bandwidth,
            )
            if bounds:
                corpus_pairs_out_p[ck] = bounds
        return pairs_out_p, corpus_pairs_out_p

    pairs_only: dict[str, dict[str, float]] = {}
    for pk in sorted(buckets):
        bounds = _one_bucket_bounds(
            buckets[pk],
            sigma_mult=sigma_mult,
            min_floor=min_floor,
            max_cap=max_cap,
            zero_half_span=zero_half_span,
            min_bandwidth=min_bandwidth,
        )
        if bounds:
            pairs_only[pk] = bounds
    return pairs_only, {}


def _one_bucket_bounds(
    ratios: list[float],
    *,
    sigma_mult: float,
    min_floor: float,
    max_cap: float,
    zero_half_span: float,
    min_bandwidth: float,
) -> dict[str, float] | None:
    if not ratios:
        return None
    mu = fmean(ratios)
    sig = pstdev(ratios)
    raw_lo = mu - sigma_mult * sig
    raw_hi = mu + sigma_mult * sig
    lo, hi = clamp_span(
        raw_lo,
        raw_hi,
        mu,
        pstdev_obs=sig,
        min_floor=min_floor,
        max_cap=max_cap,
        zero_half_span=zero_half_span,
        min_bandwidth=min_bandwidth,
    )
    return {"min_len_ratio": round(lo, 6), "max_len_ratio": round(hi, 6)}


def default_from_bounds(pairs: dict[str, Any], corpus_pairs: dict[str, Any], min_floor: float, max_cap: float) -> dict[str, float]:
    all_bounds = list(pairs.values()) + list(corpus_pairs.values())
    if not all_bounds:
        return {"min_len_ratio": 0.25, "max_len_ratio": 4.0}
    lo = max(min_floor, min(b["min_len_ratio"] for b in all_bounds) - 0.05)
    hi = min(max_cap, max(b["max_len_ratio"] for b in all_bounds) + 0.05)
    if hi <= lo:
        hi = lo + 0.1
    return {"min_len_ratio": round(lo, 6), "max_len_ratio": round(hi, 6)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build per-pair mix len_ratio JSON.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-ratios-csv", type=Path)
    src.add_argument("--hypotheses-jsonl", type=Path)
    ap.add_argument("--group-by", choices=["corpus_pair", "pair"], default="corpus_pair")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--min-floor", type=float, default=0.05, dest="min_floor")
    ap.add_argument("--max-cap", type=float, default=12.0, dest="max_cap")
    ap.add_argument("--zero-std-half-span", type=float, default=0.35, dest="zero_half_span")
    ap.add_argument("--min-bandwidth", type=float, default=0.08, dest="min_bandwidth")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-candidates", type=int, default=5)
    ap.add_argument("--qe-score-field", default="comet_qe")
    ap.add_argument("--merge-into-existing", type=Path, default=None)
    args = ap.parse_args()

    existing: dict[str, Any] = {}
    if args.merge_into_existing:
        existing = json.loads(Path(args.merge_into_existing).read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("--merge-into-existing 须为 JSON object")

    if args.from_ratios_csv:
        ratio_rows = collect_ratio_rows_csv(args.from_ratios_csv)
        source_note = str(args.from_ratios_csv.resolve())
    else:
        ratio_rows = collect_ratio_rows_hypo(args.hypotheses_jsonl)  # type: ignore[arg-type]
        source_note = str(Path(args.hypotheses_jsonl).resolve())

    pairs_new, corpus_new = aggregate_bounds(
        ratio_rows,
        group_by=args.group_by,
        sigma_mult=args.sigma,
        min_floor=args.min_floor,
        max_cap=args.max_cap,
        zero_half_span=args.zero_half_span,
        min_bandwidth=args.min_bandwidth,
    )

    pairs_merged = dict(existing.get("pairs") or {})
    corpus_merged = dict(existing.get("corpus_pairs") or {})
    pairs_merged.update(pairs_new)
    corpus_merged.update(corpus_new)

    default_lr = existing.get("default_len_ratio") or default_from_bounds(
        pairs_merged, corpus_merged, args.min_floor, args.max_cap
    )

    out: dict[str, Any] = {
        "default_len_ratio": default_lr,
        "pairs": pairs_merged,
        "corpus_pairs": corpus_merged,
        "max_candidates": int(existing.get("max_candidates", args.max_candidates)),
        "qe_score_field": str(existing.get("qe_score_field", args.qe_score_field)),
        "hypothesis_normalize_for_dedupe": bool(existing.get("hypothesis_normalize_for_dedupe", True)),
        "_generated": {
            "source": source_note,
            "group_by": args.group_by,
            "sigma": args.sigma,
            "min_floor": args.min_floor,
            "max_cap": args.max_cap,
            "zero_std_half_span": args.zero_half_span,
            "min_bandwidth": args.min_bandwidth,
            "pairs_written_or_updated": len(pairs_new),
            "corpus_pairs_written_or_updated": len(corpus_new),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
