#!/usr/bin/env python3
"""
Oracle BLEU vs system BLEU on flat candidates.jsonl（MT 论文常见上界分析）。

- System BLEU：每个样本取固定候选（默认 candidate_id=0），再算 corpus BLEU。
- Oracle BLEU：每个样本在全部候选里选 sentence-level BLEU 最高者，再算 corpus BLEU。

与 rerank 后实际输出对比时，可传 --hypotheses-jsonl，用其中的 hypothesis 作为 system 侧。

示例（路径相对 lowres_new 根目录）：
  python3 scripts/analysis/compare_oracle_bleu.py \\
    --candidates-jsonl eval_multilingual/foo/candidates.jsonl \\
    --out-dir eval_multilingual/foo/oracle_bleu_analysis

  python3 scripts/analysis/compare_oracle_bleu.py \\
    --candidates-jsonl eval_multilingual/foo/candidates.jsonl \\
    --hypotheses-jsonl eval_multilingual/foo/hypotheses.jsonl \\
    --out-dir eval_multilingual/foo/oracle_bleu_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent / "run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import run_eval as eval_common  # noqa: E402


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


_SPM_CACHE: dict[str, Any] = {}


def _get_spm_processor(spm_model: Path) -> Any:
    key = str(spm_model.resolve())
    proc = _SPM_CACHE.get(key)
    if proc is not None:
        return proc
    import sentencepiece as spm

    proc = spm.SentencePieceProcessor()
    proc.Load(str(spm_model))
    _SPM_CACHE[key] = proc
    return proc


def _spm_encode(text: str, spm_model: Path) -> str:
    proc = _get_spm_processor(spm_model)
    return " ".join(proc.EncodeAsPieces((text or "").strip()))


def sentence_bleu(
    hyp: str,
    ref: str,
    *,
    eval_corpus: str,
    eval_pair: str,
    policy: str,
    flores200_spm: Path | None,
) -> float:
    import sacrebleu

    tok = eval_common.sacrebleu_tokenize_for_group(eval_corpus, eval_pair, policy)
    if tok == eval_common._THAI_SACREbleu_TOK:
        h = eval_common._segment_thai_pythai_words([hyp or ""])[0]
        r = eval_common._segment_thai_pythai_words([ref or ""])[0]
        return float(sacrebleu.sentence_bleu(h, [r], tokenize="none").score)
    if tok == "flores200" and flores200_spm and flores200_spm.is_file():
        h = _spm_encode(hyp or "", flores200_spm)
        r = _spm_encode(ref or "", flores200_spm)
        return float(sacrebleu.sentence_bleu(h, [r], tokenize="none").score)
    return float(sacrebleu.sentence_bleu(hyp or "", [ref or ""], tokenize=tok).score)


@dataclass
class SampleOracleRow:
    key: tuple[Any, ...]
    corpus: str
    pair: str
    sample_id: str
    reference: str
    system_hyp: str
    oracle_hyp: str
    system_cand_id: int | None
    oracle_cand_id: int
    n_candidates: int
    sent_bleu_system: float
    sent_bleu_oracle: float
    oracle_gain: float


def group_candidates(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "hypothesis" not in row or "reference_text" not in row:
            raise ValueError("candidates 行需含 hypothesis 与 reference_text")
        groups[eval_common.result_key(row)].append(row)
    for key in groups:
        groups[key].sort(key=_candidate_sort_key)
    return groups


def load_system_hypotheses(path: Path) -> dict[tuple[Any, ...], str]:
    out: dict[tuple[Any, ...], str] = {}
    for row in read_jsonl(path):
        out[eval_common.result_key(row)] = str(row.get("hypothesis", "") or "")
    return out


def corpus_bleu_for_lists(
    hyps: list[str],
    refs: list[str],
    *,
    eval_corpus: str,
    eval_pair: str,
    policy: str,
    flores200_spm: Path | None,
) -> tuple[float, str]:
    return eval_common.corpus_bleu_with_fallbacks(
        hyps,
        refs,
        eval_corpus,
        eval_pair,
        policy,
        flores200_spm_model=flores200_spm,
    )


def analyze(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    *,
    system_by_key: dict[tuple[Any, ...], str] | None,
    system_candidate_id: int,
    bleu_policy: str,
    flores200_spm: Path | None,
) -> tuple[list[SampleOracleRow], dict[str, Any]]:
    per_sample: list[SampleOracleRow] = []
    # (corpus, pair) -> lists
    bucket_hyps_sys: dict[tuple[str, str], list[str]] = defaultdict(list)
    bucket_hyps_orc: dict[tuple[str, str], list[str]] = defaultdict(list)
    bucket_refs: dict[tuple[str, str], list[str]] = defaultdict(list)

    for key, cands in tqdm(sorted(groups.items()), desc="sentence-bleu", unit="sample"):
        if not cands:
            continue
        base = cands[0]
        corpus = str(base.get("eval_corpus") or base.get("dataset") or "")
        pair = pair_label(base)
        ref = str(base.get("reference_text", "") or "")
        eval_pair = pair

        if system_by_key is not None:
            if key not in system_by_key:
                raise KeyError(f"hypotheses 缺少样本 key={key}")
            system_hyp = system_by_key[key]
            system_cid: int | None = None
        else:
            pick = None
            for c in cands:
                if _candidate_sort_key(c) == system_candidate_id:
                    pick = c
                    break
            if pick is None:
                pick = cands[0]
            system_hyp = str(pick.get("hypothesis", "") or "")
            system_cid = _candidate_sort_key(pick)

        best_score = float("-inf")
        oracle_hyp = system_hyp
        oracle_cid = _candidate_sort_key(cands[0])
        for c in cands:
            hyp = str(c.get("hypothesis", "") or "")
            sc = sentence_bleu(
                hyp,
                ref,
                eval_corpus=corpus,
                eval_pair=eval_pair,
                policy=bleu_policy,
                flores200_spm=flores200_spm,
            )
            if sc > best_score:
                best_score = sc
                oracle_hyp = hyp
                oracle_cid = _candidate_sort_key(c)

        bleu_sys = sentence_bleu(
            system_hyp,
            ref,
            eval_corpus=corpus,
            eval_pair=eval_pair,
            policy=bleu_policy,
            flores200_spm=flores200_spm,
        )
        bleu_orc = best_score

        per_sample.append(
            SampleOracleRow(
                key=key,
                corpus=corpus,
                pair=pair,
                sample_id=str(base.get("sample_id", "")),
                reference=ref,
                system_hyp=system_hyp,
                oracle_hyp=oracle_hyp,
                system_cand_id=system_cid,
                oracle_cand_id=oracle_cid,
                n_candidates=len(cands),
                sent_bleu_system=bleu_sys,
                sent_bleu_oracle=bleu_orc,
                oracle_gain=bleu_orc - bleu_sys,
            )
        )
        bk = (corpus, pair)
        bucket_hyps_sys[bk].append(system_hyp)
        bucket_hyps_orc[bk].append(oracle_hyp)
        bucket_refs[bk].append(ref)

    by_pair_rows: list[dict[str, Any]] = []

    for (corpus, pair), refs in sorted(bucket_refs.items()):
        hyps_sys = bucket_hyps_sys[(corpus, pair)]
        hyps_orc = bucket_hyps_orc[(corpus, pair)]
        bleu_sys, tok_sys = corpus_bleu_for_lists(
            hyps_sys, refs, eval_corpus=corpus, eval_pair=pair, policy=bleu_policy, flores200_spm=flores200_spm
        )
        bleu_orc, tok_orc = corpus_bleu_for_lists(
            hyps_orc, refs, eval_corpus=corpus, eval_pair=pair, policy=bleu_policy, flores200_spm=flores200_spm
        )
        n = len(refs)
        subset = [s for s in per_sample if s.corpus == corpus and s.pair == pair]
        mean_sent_gain = sum(s.oracle_gain for s in subset) / n if n else 0.0
        oracle_wins = sum(1 for s in subset if s.oracle_gain > 1e-9)
        by_pair_rows.append(
            {
                "corpus": corpus,
                "pair": pair,
                "n": n,
                "mean_n_candidates": sum(s.n_candidates for s in subset) / n if n else 0.0,
                "corpus_bleu_system": round(bleu_sys, 4),
                "corpus_bleu_oracle": round(bleu_orc, 4),
                "oracle_gap": round(bleu_orc - bleu_sys, 4),
                "mean_sent_bleu_gain": round(mean_sent_gain, 4),
                "oracle_win_rate": round(oracle_wins / n, 4) if n else 0.0,
                "bleu_tokenize_system": tok_sys,
                "bleu_tokenize_oracle": tok_orc,
            }
        )
    total_n = sum(r["n"] for r in by_pair_rows)
    if total_n:
        w_sys = sum(r["corpus_bleu_system"] * r["n"] for r in by_pair_rows) / total_n
        w_orc = sum(r["corpus_bleu_oracle"] * r["n"] for r in by_pair_rows) / total_n
    else:
        w_sys = w_orc = 0.0

    macro_sys = sum(r["corpus_bleu_system"] for r in by_pair_rows) / len(by_pair_rows) if by_pair_rows else 0.0
    macro_orc = sum(r["corpus_bleu_oracle"] for r in by_pair_rows) / len(by_pair_rows) if by_pair_rows else 0.0

    # 仅单语向时给出真· pooled corpus BLEU（与 run_eval 单组一致）
    pooled_sys = pooled_orc = None
    if len(by_pair_rows) == 1 and per_sample:
        pooled_sys = by_pair_rows[0]["corpus_bleu_system"]
        pooled_orc = by_pair_rows[0]["corpus_bleu_oracle"]

    summary = {
        "n_samples": len(per_sample),
        "n_pairs": len(by_pair_rows),
        "system_source": "hypotheses_jsonl" if system_by_key else f"candidate_id={system_candidate_id}",
        "bleu_tokenize_policy": bleu_policy,
        "corpus_bleu_system_weighted_by_pair": round(w_sys, 4),
        "corpus_bleu_oracle_weighted_by_pair": round(w_orc, 4),
        "oracle_gap_weighted_by_pair": round(w_orc - w_sys, 4),
        "corpus_bleu_system_macro_avg_by_pair": round(macro_sys, 4),
        "corpus_bleu_oracle_macro_avg_by_pair": round(macro_orc, 4),
        "oracle_gap_macro_avg_by_pair": round(macro_orc - macro_sys, 4),
        "corpus_bleu_system_single_pair": pooled_sys,
        "corpus_bleu_oracle_single_pair": pooled_orc,
        "mean_sent_bleu_gain": round(sum(s.oracle_gain for s in per_sample) / len(per_sample), 4) if per_sample else 0.0,
        "oracle_win_rate": round(sum(1 for s in per_sample if s.oracle_gain > 1e-9) / len(per_sample), 4)
        if per_sample
        else 0.0,
        "note": "每语向单独 corpus BLEU（分词与 run_eval 一致）；全局用 weighted / macro 汇总，勿跨语向拼句算 BLEU。",
        "by_pair": by_pair_rows,
    }
    return per_sample, summary


def write_outputs(out_dir: Path, per_sample: list[SampleOracleRow], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oracle_bleu_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = out_dir / "oracle_bleu_by_pair.csv"
    if summary["by_pair"]:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary["by_pair"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["by_pair"])

    sent_path = out_dir / "oracle_bleu_per_sentence.csv"
    with sent_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "sample_id",
                "n_candidates",
                "system_candidate_id",
                "oracle_candidate_id",
                "sent_bleu_system",
                "sent_bleu_oracle",
                "sent_bleu_gain",
            ],
        )
        writer.writeheader()
        for s in per_sample:
            writer.writerow(
                {
                    "corpus": s.corpus,
                    "pair": s.pair,
                    "sample_id": s.sample_id,
                    "n_candidates": s.n_candidates,
                    "system_candidate_id": "" if s.system_cand_id is None else s.system_cand_id,
                    "oracle_candidate_id": s.oracle_cand_id,
                    "sent_bleu_system": round(s.sent_bleu_system, 4),
                    "sent_bleu_oracle": round(s.sent_bleu_oracle, 4),
                    "sent_bleu_gain": round(s.oracle_gain, 4),
                }
            )

    md_lines = [
        "# Oracle BLEU vs System BLEU",
        "",
        f"- Samples: **{summary['n_samples']}**",
        f"- System source: `{summary['system_source']}`",
        f"- BLEU tokenize policy: `{summary['bleu_tokenize_policy']}`",
        "",
        "## Weighted by sentence count over language pairs",
        "",
        f"| | BLEU |",
        f"|--|--:|",
        f"| System | {summary['corpus_bleu_system_weighted_by_pair']:.2f} |",
        f"| Oracle (upper bound) | {summary['corpus_bleu_oracle_weighted_by_pair']:.2f} |",
        f"| Gap (Oracle − System) | {summary['oracle_gap_weighted_by_pair']:.2f} |",
        "",
        "## Macro average over language pairs",
        "",
        f"| | BLEU |",
        f"|--|--:|",
        f"| System | {summary['corpus_bleu_system_macro_avg_by_pair']:.2f} |",
        f"| Oracle | {summary['corpus_bleu_oracle_macro_avg_by_pair']:.2f} |",
        f"| Gap | {summary['oracle_gap_macro_avg_by_pair']:.2f} |",
        "",
        f"- Mean per-sentence BLEU gain (oracle pick − system): **{summary['mean_sent_bleu_gain']:.4f}**",
        f"- Oracle win rate (sent BLEU strictly better): **{100 * summary['oracle_win_rate']:.1f}%**",
        "",
        summary["note_pooled"],
        "",
        "See `oracle_bleu_by_pair.csv` and `oracle_bleu_per_sentence.csv`.",
    ]
    (out_dir / "oracle_bleu_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    repo = eval_common.root()
    ap = argparse.ArgumentParser(description="Oracle BLEU vs system BLEU on flat candidates.jsonl.")
    ap.add_argument("--candidates-jsonl", type=Path, required=True)
    ap.add_argument(
        "--hypotheses-jsonl",
        type=Path,
        default=None,
        help="若提供，用其中 hypothesis 作为 system 侧（如 LLM rerank 后输出），否则用 --system-candidate-id。",
    )
    ap.add_argument("--system-candidate-id", type=int, default=0, help="无 hypotheses 时选用的 candidate_id。")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--bleu-tokenize",
        choices=("auto", "flores200", "legacy"),
        default="auto",
        help="与 run_eval.py 一致。",
    )
    ap.add_argument(
        "--flores200-spm-model",
        type=Path,
        default=repo / "models" / "sacrebleu" / "flores200_sacrebleu_tokenizer_spm.model",
    )
    args = ap.parse_args()

    if not args.candidates_jsonl.is_file():
        raise SystemExit(f"未找到: {args.candidates_jsonl}")

    rows = read_jsonl(args.candidates_jsonl)
    if not rows:
        raise SystemExit("candidates.jsonl 为空")

    groups = group_candidates(rows)
    system_by_key = None
    if args.hypotheses_jsonl:
        if not args.hypotheses_jsonl.is_file():
            raise SystemExit(f"未找到: {args.hypotheses_jsonl}")
        system_by_key = load_system_hypotheses(args.hypotheses_jsonl)

    flores_spm = args.flores200_spm_model if args.flores200_spm_model.is_file() else None
    per_sample, summary = analyze(
        groups,
        system_by_key=system_by_key,
        system_candidate_id=args.system_candidate_id,
        bleu_policy=args.bleu_tokenize,
        flores200_spm=flores_spm,
    )
    summary["candidates_jsonl"] = str(args.candidates_jsonl.resolve())
    if args.hypotheses_jsonl:
        summary["hypotheses_jsonl"] = str(args.hypotheses_jsonl.resolve())

    write_outputs(args.out_dir, per_sample, summary)

    print(f"完成。输出目录: {args.out_dir.resolve()}")
    print(
        f"Weighted BLEU — system: {summary['corpus_bleu_system_weighted_by_pair']:.2f} | "
        f"oracle: {summary['corpus_bleu_oracle_weighted_by_pair']:.2f} | "
        f"gap: {summary['oracle_gap_weighted_by_pair']:.2f}"
    )
    print(
        f"Macro by pair — system: {summary['corpus_bleu_system_macro_avg_by_pair']:.2f} | "
        f"oracle: {summary['corpus_bleu_oracle_macro_avg_by_pair']:.2f} | "
        f"gap: {summary['oracle_gap_macro_avg_by_pair']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
