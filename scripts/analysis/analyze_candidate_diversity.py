#!/usr/bin/env python3
"""
对 flat candidates.jsonl 中「同一句源文的多个译文候选」做多样性统计：
Distinct-n 与 n-gram 重复率（MT / 对话生成里常见指标）。

定义（每个评测样本一组 K 条 hypothesis）：
  - distinct_n_pooled：将该样本 K 条候选的 n-gram 合并计数，
      |unique n-grams| / |total n-gram positions|（跨候选池化，越高越多样）。
  - repetition_rate_pooled：1 - distinct_n_pooled（池化重复率）。
  - distinct_n_mean_per_cand：每条候选单独算 distinct_n 再对 K 条取平均。
  - repetition_rate_mean_per_cand：每条候选 1 - distinct_n 再平均（句内重复）。
  - pairwise_jaccard_distinct：候选两两 n-gram Jaccard 距离（1 - |∩|/|∪|）的平均值。

分词与 analyze_target_ngrams.py 一致（泰语 PyThai、中文按字/词块等）。

示例（路径相对 lowres_new）：
  python3 scripts/analysis/analyze_candidate_diversity.py \\
    --candidates-jsonl eval_multilingual/foo/candidates.jsonl \\
    --out-dir eval_multilingual/foo/candidate_diversity
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent / "run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import run_eval as eval_common  # noqa: E402

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


def pair_label(row: dict[str, Any]) -> str:
    return str(row.get("eval_pair") or f"{row['src_lang']}->{row['tgt_lang']}")


def _candidate_sort_key(row: dict[str, Any]) -> int:
    raw = row.get("candidate_id", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def tokenize(text: str, tgt_lang: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if tgt_lang.startswith("tha") and thai_word_tokenize is not None:
        return [x.strip() for x in thai_word_tokenize(text, engine="newmm") if x.strip()]
    if tgt_lang.startswith("zho") or tgt_lang.startswith("cmn") or HAN_RE.search(text):
        return WORD_RE.findall(text)
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def ngram_list(tokens: list[str], n: int) -> list[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_and_repetition(grams: list[str]) -> tuple[float, float, int, int]:
    """返回 (distinct_ratio, repetition_rate, n_unique, n_total)。"""
    total = len(grams)
    if total == 0:
        return 0.0, 0.0, 0, 0
    uniq = len(set(grams))
    distinct = uniq / total
    repetition = 1.0 - distinct
    return distinct, repetition, uniq, total


def jaccard_distinct(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


@dataclass
class SampleMetrics:
    corpus: str
    pair: str
    sample_id: str
    tgt_lang: str
    n_candidates: int
    n_unique_texts: int
    by_n: dict[int, dict[str, float]]


def analyze_group(
    cands: list[dict[str, Any]],
    orders: list[int],
    *,
    normalize_ws: bool,
) -> SampleMetrics:
    base = cands[0]
    corpus = str(base.get("eval_corpus") or base.get("dataset") or "")
    pair = pair_label(base)
    tgt_lang = str(base.get("tgt_lang", "") or "")
    texts: list[str] = []
    for c in cands:
        t = str(c.get("hypothesis", "") or "")
        if normalize_ws:
            t = re.sub(r"\s+", " ", t.strip())
        if t:
            texts.append(t)
    n_unique_texts = len(set(texts))

    by_n: dict[int, dict[str, float]] = {}
    tokenized = [tokenize(t, tgt_lang) for t in texts]

    for n in orders:
        per_cand_grams = [ngram_list(tok, n) for tok in tokenized]
        pooled: list[str] = []
        for g in per_cand_grams:
            pooled.extend(g)

        d_pool, r_pool, u_pool, t_pool = distinct_and_repetition(pooled)

        per_distinct: list[float] = []
        per_repetition: list[float] = []
        sets_per_cand: list[set[str]] = []
        for g in per_cand_grams:
            d, r, _, _ = distinct_and_repetition(g)
            per_distinct.append(d)
            per_repetition.append(r)
            sets_per_cand.append(set(g))

        pairwise: list[float] = []
        if len(sets_per_cand) >= 2:
            for a, b in combinations(sets_per_cand, 2):
                pairwise.append(jaccard_distinct(a, b))

        by_n[n] = {
            "distinct_n_pooled": round(d_pool, 6),
            "repetition_rate_pooled": round(r_pool, 6),
            "distinct_n_mean_per_cand": round(sum(per_distinct) / len(per_distinct), 6) if per_distinct else 0.0,
            "repetition_rate_mean_per_cand": round(sum(per_repetition) / len(per_repetition), 6)
            if per_repetition
            else 0.0,
            "pairwise_jaccard_distinct_mean": round(sum(pairwise) / len(pairwise), 6) if pairwise else 0.0,
            "n_grams_pooled_total": float(t_pool),
            "n_grams_pooled_unique": float(u_pool),
        }

    return SampleMetrics(
        corpus=corpus,
        pair=pair,
        sample_id=str(base.get("sample_id", "")),
        tgt_lang=tgt_lang,
        n_candidates=len(texts),
        n_unique_texts=n_unique_texts,
        by_n=by_n,
    )


def group_candidates(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "hypothesis" not in row:
            raise ValueError("candidates 行需含 hypothesis")
        groups[eval_common.result_key(row)].append(row)
    for key in groups:
        groups[key].sort(key=_candidate_sort_key)
    return groups


def aggregate(samples: list[SampleMetrics], orders: list[int]) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], list[SampleMetrics]] = defaultdict(list)
    for s in samples:
        by_pair[(s.corpus, s.pair)].append(s)

    pair_rows: list[dict[str, Any]] = []
    for (corpus, pair), subset in sorted(by_pair.items()):
        row: dict[str, Any] = {"corpus": corpus, "pair": pair, "n_samples": len(subset)}
        row["mean_n_candidates"] = round(sum(s.n_candidates for s in subset) / len(subset), 4)
        row["mean_n_unique_texts"] = round(sum(s.n_unique_texts for s in subset) / len(subset), 4)
        for n in orders:
            for metric in (
                "distinct_n_pooled",
                "repetition_rate_pooled",
                "distinct_n_mean_per_cand",
                "repetition_rate_mean_per_cand",
                "pairwise_jaccard_distinct_mean",
            ):
                vals = [s.by_n[n][metric] for s in subset if n in s.by_n]
                row[f"{metric}_n{n}"] = round(sum(vals) / len(vals), 6) if vals else 0.0
        pair_rows.append(row)

    global_row: dict[str, Any] = {"n_samples": len(samples)}
    if samples:
        global_row["mean_n_candidates"] = round(sum(s.n_candidates for s in samples) / len(samples), 4)
        for n in orders:
            for metric in (
                "distinct_n_pooled",
                "repetition_rate_pooled",
                "distinct_n_mean_per_cand",
                "repetition_rate_mean_per_cand",
                "pairwise_jaccard_distinct_mean",
            ):
                vals = [s.by_n[n][metric] for s in samples if n in s.by_n]
                global_row[f"mean_{metric}_n{n}"] = round(sum(vals) / len(vals), 6) if vals else 0.0

    return {"global": global_row, "by_pair": pair_rows}


def write_per_sample_csv(path: Path, samples: list[SampleMetrics], orders: list[int]) -> None:
    fieldnames = [
        "corpus",
        "pair",
        "sample_id",
        "tgt_lang",
        "n_candidates",
        "n_unique_texts",
    ]
    for n in orders:
        for suffix in (
            "distinct_n_pooled",
            "repetition_rate_pooled",
            "distinct_n_mean_per_cand",
            "repetition_rate_mean_per_cand",
            "pairwise_jaccard_distinct_mean",
        ):
            fieldnames.append(f"{suffix}_n{n}")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {
                "corpus": s.corpus,
                "pair": s.pair,
                "sample_id": s.sample_id,
                "tgt_lang": s.tgt_lang,
                "n_candidates": s.n_candidates,
                "n_unique_texts": s.n_unique_texts,
            }
            for n in orders:
                m = s.by_n.get(n, {})
                for suffix in (
                    "distinct_n_pooled",
                    "repetition_rate_pooled",
                    "distinct_n_mean_per_cand",
                    "repetition_rate_mean_per_cand",
                    "pairwise_jaccard_distinct_mean",
                ):
                    row[f"{suffix}_n{n}"] = m.get(suffix, "")
            writer.writerow(row)


def write_report(path: Path, summary: dict[str, Any], orders: list[int]) -> None:
    g = summary["global"]
    lines = [
        "# Candidate diversity (Distinct-n & repetition)",
        "",
        f"- Samples: **{g.get('n_samples', 0)}**",
        f"- Mean candidates per sample: **{g.get('mean_n_candidates', 0)}**",
        "",
        "## Metrics (macro mean over samples)",
        "",
        "| n | Distinct (pooled) | Repetition (pooled) | Distinct (per cand) | Repetition (per cand) | Pairwise Jaccard dist |",
        "|--:|--:|--:|--:|--:|--:|",
    ]
    for n in orders:
        lines.append(
            f"| {n} | {g.get(f'mean_distinct_n_pooled_n{n}', 0):.4f} | "
            f"{g.get(f'mean_repetition_rate_pooled_n{n}', 0):.4f} | "
            f"{g.get(f'mean_distinct_n_mean_per_cand_n{n}', 0):.4f} | "
            f"{g.get(f'mean_repetition_rate_mean_per_cand_n{n}', 0):.4f} | "
            f"{g.get(f'mean_pairwise_jaccard_distinct_mean_n{n}', 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "**Pooled**：同一句的 K 条候选 n-gram 合并后算 distinct / repetition。",
            "**Per cand**：每条候选内部 distinct，再对 K 条取平均。",
            "**Pairwise Jaccard distinct**：候选间 n-gram 集合差异（1 − Jaccard similarity）。",
            "",
            "详见 `candidate_diversity_per_sample.csv` 与 `candidate_diversity_by_pair.csv`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Distinct-n and n-gram repetition among candidates per source sentence.")
    ap.add_argument("--candidates-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, action="append", default=[1, 2], help="n-gram 阶数，可重复传入，默认 1 与 2。")
    ap.add_argument(
        "--normalize-whitespace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="去重空白后再统计（默认开启）。",
    )
    args = ap.parse_args()

    if not args.candidates_jsonl.is_file():
        raise SystemExit(f"未找到: {args.candidates_jsonl}")

    orders = sorted(set(args.n))
    if not orders or any(x < 1 for x in orders):
        raise SystemExit("--n 须为正整数")

    rows = read_jsonl(args.candidates_jsonl)
    if not rows:
        raise SystemExit("candidates.jsonl 为空")

    groups = group_candidates(rows)
    samples: list[SampleMetrics] = []
    for _key, cands in sorted(groups.items()):
        if len(cands) < 1:
            continue
        samples.append(
            analyze_group(cands, orders, normalize_ws=bool(args.normalize_whitespace))
        )

    summary = aggregate(samples, orders)
    summary["candidates_jsonl"] = str(args.candidates_jsonl.resolve())
    summary["n_orders"] = orders
    summary["definitions"] = {
        "distinct_n_pooled": "|unique n-grams| / |total n-gram positions| over all K candidates",
        "repetition_rate_pooled": "1 - distinct_n_pooled",
        "distinct_n_mean_per_cand": "mean over candidates of (unique/total) within each hypothesis",
        "repetition_rate_mean_per_cand": "mean over candidates of (1 - distinct) within each hypothesis",
        "pairwise_jaccard_distinct_mean": "mean over candidate pairs of (1 - |∩|/|∪|) on n-gram sets",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidate_diversity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_per_sample_csv(args.out_dir / "candidate_diversity_per_sample.csv", samples, orders)
    if summary["by_pair"]:
        with (args.out_dir / "candidate_diversity_by_pair.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary["by_pair"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["by_pair"])
    write_report(args.out_dir / "candidate_diversity_report.md", summary, orders)

    g = summary["global"]
    print(f"完成。输出: {args.out_dir.resolve()}")
    for n in orders:
        print(
            f"  n={n} — distinct_pooled={g.get(f'mean_distinct_n_pooled_n{n}', 0):.4f} "
            f"repetition_pooled={g.get(f'mean_repetition_rate_pooled_n{n}', 0):.4f} "
            f"pairwise_dist={g.get(f'mean_pairwise_jaccard_distinct_mean_n{n}', 0):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
