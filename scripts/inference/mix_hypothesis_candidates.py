#!/usr/bin/env python3
"""
合并多份扁平 candidates.jsonl（如 sample/rag × 两条推理链路），产出新的候选集供下游 LLM rerank 再 eval。

规则（对每个 eval 样本）：
  1) 按配置过滤：长度比在 [min_len_ratio, max_len_ratio] 之外则丢弃该行候选。
  2) 完全相同（归一化后）的译文只保留一条；若多条重复保留 QE 较高的那条对应的整行。
  3) 若仍多于 max_candidates 条，仅保留 QE 分数最高的 max_candidates 条。
  4) 若步骤 1 后一条都不剩：从合并前的全部候选中保留 QE 最高的一条。

配置：默认 configs/mix_hypothesis_candidates.json。支持：
  - default_len_ratio：全局兜底
  - pairs：eval_pair 粒度（例 eng_Latn->zho_Hans）
  - corpus_pairs：eval_corpus|eval_pair 粒度（优先级高于 pairs）

 per 语向区间可用 scripts/analysis/build_mix_len_ratio_config.py 基于 analyze_length_ratios 统计生成。

QE：默认 comet_qe；若缺失可加 --comet-qe-model 现算。

示例：
  conda run -n lowres python scripts/inference/mix_hypothesis_candidates.py \\
    --mix-config configs/mix_hypothesis_candidates.json \\
    --input run/exp_a/candidates.jsonl \\
    --input run/exp_b/candidates.jsonl \\
    --output-dir eval_multilingual/my_mix/run_001
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent / "run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import run_eval as eval_common  # noqa: E402

# 与 scripts/prepare/filter_augmented_mt_by_qe_stats.py 对齐
# 注意：原泰文片段 [\u0e00-\u0e7f]+ 会把整段连续泰文当 1 个 token，导致英→泰长度比严重偏小，
# 所有候选会被默认 [min_len_ratio, max_len_ratio] 砍光后触发 fallback 只剩 1 条。
# 这里改成：泰文段先用 pythainlp.word_tokenize 切，再让非泰文按下面正则匹配。
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]+")
TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?|\d+(?:[.,]\d+)*",
    re.UNICODE,
)

_THAI_WORD_TOKENIZE = None


def _thai_word_tokenize(text: str) -> list[str]:
    """惰性加载 pythainlp 并按 newmm 引擎切词，过滤空白。"""
    global _THAI_WORD_TOKENIZE
    if _THAI_WORD_TOKENIZE is None:
        try:
            from pythainlp.tokenize import word_tokenize as _wt
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "泰语长度比统计需要 PyThaiNLP：在当前 conda 环境内 pip install pythainlp"
            ) from e
        _THAI_WORD_TOKENIZE = _wt
    return [t for t in _THAI_WORD_TOKENIZE(text, engine="newmm") if t and not t.isspace()]


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


def token_len(text: str) -> int:
    s = text or ""
    if not s:
        return 0
    # 不含泰文：原有正则即可
    if not _THAI_RE.search(s):
        return len(TOKEN_RE.findall(s))
    # 含泰文：把泰文段抠出来交给 pythainlp，剩余部分用 TOKEN_RE 计数
    count = 0
    last = 0
    for m in _THAI_RE.finditer(s):
        if m.start() > last:
            count += len(TOKEN_RE.findall(s[last : m.start()]))
        count += len(_thai_word_tokenize(m.group(0)))
        last = m.end()
    if last < len(s):
        count += len(TOKEN_RE.findall(s[last:]))
    return count


def len_ratio(source_text: str, hypothesis: str) -> float:
    s = token_len(source_text)
    t = token_len(hypothesis)
    return t / s if s > 0 else 0.0


def normalize_hypothesis(text: str, *, normalize: bool) -> str:
    s = text or ""
    if normalize:
        s = re.sub(r"\s+", " ", s.strip())
    else:
        s = s.strip()
    return s


def pair_from_item(row: dict[str, Any]) -> str:
    return str(row.get("eval_pair") or f"{row.get('src_lang', '')}->{row.get('tgt_lang', '')}")


def _parse_pairs_map(raw_pairs: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(raw_pairs, dict):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for k, v in raw_pairs.items():
        if not isinstance(v, dict):
            continue
        out[str(k)] = (float(v["min_len_ratio"]), float(v["max_len_ratio"]))
    return out


def load_mix_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"mix config 不存在: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("mix config JSON 必须为 object")

    if "default_len_ratio" in raw and isinstance(raw["default_len_ratio"], dict):
        dlr = raw["default_len_ratio"]
        dmin = float(dlr["min_len_ratio"])
        dmax = float(dlr["max_len_ratio"])
    elif "min_len_ratio" in raw and "max_len_ratio" in raw:
        # 兼容旧版扁平写法
        dmin = float(raw["min_len_ratio"])
        dmax = float(raw["max_len_ratio"])
    else:
        raise ValueError("mix config 需包含 default_len_ratio 或顶层 min_len_ratio/max_len_ratio")

    mc = raw.get("max_candidates")
    if mc is None:
        raise ValueError("mix config 缺少 max_candidates")
    out: dict[str, Any] = {
        "defaults": (dmin, dmax),
        "pairs": _parse_pairs_map(raw.get("pairs")),
        "corpus_pairs": _parse_pairs_map(raw.get("corpus_pairs")),
        "max_candidates": int(mc),
        "qe_score_field": str(raw.get("qe_score_field", "comet_qe")),
        "hypothesis_normalize_for_dedupe": bool(raw.get("hypothesis_normalize_for_dedupe", True)),
    }
    return out


def resolve_len_bounds_for_row(row: dict[str, Any], mix: dict[str, Any]) -> tuple[float, float, str]:
    """返回 (min_len_ratio, max_len_ratio, 命中来源)。"""

    pair = pair_from_item(row)
    corpus = str(row.get("eval_corpus") or row.get("dataset") or "").strip()
    if corpus:
        ck = f"{corpus}|{pair}"
        if ck in mix["corpus_pairs"]:
            lo, hi = mix["corpus_pairs"][ck]
            return lo, hi, "corpus_pair"
    if pair in mix["pairs"]:
        lo, hi = mix["pairs"][pair]
        return lo, hi, "pair"
    lo, hi = mix["defaults"]
    return lo, hi, "default"


def mix_config_summary_for_meta(mix: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable 摘要。"""
    return {
        "defaults": {"min_len_ratio": mix["defaults"][0], "max_len_ratio": mix["defaults"][1]},
        "pairs": {k: {"min_len_ratio": v[0], "max_len_ratio": v[1]} for k, v in mix["pairs"].items()},
        "corpus_pairs": {k: {"min_len_ratio": v[0], "max_len_ratio": v[1]} for k, v in mix["corpus_pairs"].items()},
        "max_candidates": mix["max_candidates"],
        "qe_score_field": mix["qe_score_field"],
        "hypothesis_normalize_for_dedupe": mix["hypothesis_normalize_for_dedupe"],
    }


def get_qe(row: dict[str, Any], qe_field: str) -> float | None:
    v = row.get(qe_field)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def comet_score_rows(rows: list[dict[str, Any]], qe_field: str, cfg: argparse.Namespace, out_model_dir: Path) -> None:
    """原地写入 qe_field。rows 每项需含 source_text、hypothesis。"""
    ckpt, torch_mod, load_fn = eval_common.prepare_comet_checkpoint(
        str(cfg.comet_qe_model), out_model_dir, encoder_path=cfg.comet_encoder_model
    )
    if not ckpt or torch_mod is None or load_fn is None:
        raise RuntimeError(f"无法加载 COMET-QE：{cfg.comet_qe_model}")
    gpus = 1 if torch_mod.cuda.is_available() else 0
    ckpt = eval_common.patch_comet_checkpoint_pretrained_model(ckpt, cfg.comet_encoder_model)
    model = eval_common.load_comet_model(load_fn, ckpt)
    data = [{"src": str(r["source_text"]), "mt": str(r["hypothesis"])} for r in rows]
    pred = model.predict(data, batch_size=cfg.comet_batch_size, gpus=gpus)
    scores = pred.get("scores", []) if isinstance(pred, dict) else getattr(pred, "scores", [])
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError(f"COMET-QE 长度不一致：scores={len(scores) if isinstance(scores, list) else '?'} rows={len(rows)}")
    for row, score in zip(rows, scores, strict=True):
        row[qe_field] = float(score)


def select_for_sample(rows: list[dict[str, Any]], mix: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """rows：同一样本在多个输入文件中合并后的所有扁平行。"""
    ref = rows[0]
    min_lr, max_lr, bounds_src = resolve_len_bounds_for_row(ref, mix)
    max_k = int(mix["max_candidates"])
    qe_field = str(mix["qe_score_field"])
    norm_dedupe = bool(mix["hypothesis_normalize_for_dedupe"])

    stats = {
        "input_rows": len(rows),
        "len_ok_count": 0,
        "used_len_empty_fallback": False,
        "fallback_reason": "",
        "len_bounds_source": bounds_src,
        "min_len_ratio_used": min_lr,
        "max_len_ratio_used": max_lr,
    }

    def row_qe(row: dict[str, Any]) -> float:
        v = get_qe(row, qe_field)
        if v is None:
            return float("-inf")
        return v

    if not rows:
        return [], stats

    if all(get_qe(r, qe_field) is None for r in rows):
        raise ValueError(f"样本 {eval_common.result_key(rows[0])} 无任何有效 {qe_field}")

    len_filtered: list[dict[str, Any]] = []
    for r in rows:
        hyp = normalize_hypothesis(str(r.get("hypothesis", "")), normalize=norm_dedupe)
        if not hyp:
            continue
        lr = len_ratio(str(r.get("source_text", "")), hyp)
        if min_lr <= lr <= max_lr:
            len_filtered.append(r)

    stats["len_ok_count"] = len(len_filtered)

    pool = len_filtered
    used_fallback = False
    if not pool:
        pool = list(rows)
        used_fallback = True
        stats["used_len_empty_fallback"] = True
        stats["fallback_reason"] = "len_ratio_filter_removed_all"

    # 去重：相同归一化 hypothesis 保留 QE 更高的整行
    best_by_hyp: dict[str, dict[str, Any]] = {}
    for r in pool:
        h = normalize_hypothesis(str(r.get("hypothesis", "")), normalize=norm_dedupe)
        if not h:
            continue
        rq = row_qe(r)
        prev = best_by_hyp.get(h)
        if prev is None or rq > row_qe(prev):
            best_by_hyp[h] = r

    deduped = list(best_by_hyp.values())
    if not deduped:
        # 极端情况：hypothesis 经归一化为空全部被跳过 —— 在全量候选中按 QE 取 1 条
        fb = sorted(list(rows), key=row_qe, reverse=True)[:1]
        return fb, stats

    deduped.sort(key=row_qe, reverse=True)

    if used_fallback:
        deduped = deduped[:1]
    elif len(deduped) > max_k:
        deduped = deduped[:max_k]

    return deduped, stats


def strip_candidate_specific_fields(row: dict[str, Any]) -> dict[str, Any]:
    """输出与 generate_reranked_hypotheses 扁平候选行对齐，去掉旧 candidate_id 等。"""
    skip = {"candidate_id", "hypothesis"}
    base = {k: v for k, v in row.items() if k not in skip}
    return base


def main() -> int:
    repo_root = eval_common.root()
    default_cfg = repo_root / "configs" / "mix_hypothesis_candidates.json"
    ap = argparse.ArgumentParser(description="Merge flat candidates.jsonl for mixed reranking.")
    ap.add_argument("--mix-config", type=Path, default=default_cfg, help=f"默认 {default_cfg}")
    ap.add_argument("--input", type=Path, action="append", dest="inputs", default=[], metavar="PATH", help="可重复传入多份扁平 candidates.jsonl")
    ap.add_argument("--output-dir", type=Path, required=True, help="写出 candidates.jsonl 与 mix_meta.json")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="可选：与推理脚本相同 manifest，用于样本顺序对齐（默认字典序遍历 key）",
    )
    ap.add_argument(
        "--comet-qe-model",
        default="",
        help="若任一候选缺 qe_score_field，需指定以在本脚本内批量打分（与 generate_reranked_hypotheses 一致的路径）",
    )
    ap.add_argument("--comet-encoder-model", type=Path, default=repo_root / "models" / "xlm-roberta-large")
    ap.add_argument("--comet-batch-size", type=int, default=16)
    ap.add_argument("--offline-eval-assets", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if not args.inputs:
        raise SystemExit("至少需要一份 --input candidates.jsonl")

    mix_cfg = load_mix_config(args.mix_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    file_row_counts: list[tuple[str, int]] = []
    for p in args.inputs:
        if not p.is_file():
            raise FileNotFoundError(p)
        part = read_jsonl(p)
        file_row_counts.append((str(p), len(part)))
        for row in part:
            merged_by_key[eval_common.result_key(row)].append(dict(row))

    all_rows_flat: list[dict[str, Any]] = [r for lst in merged_by_key.values() for r in lst]
    qe_field = str(mix_cfg["qe_score_field"])
    need_scores = any(get_qe(r, qe_field) is None for r in all_rows_flat)
    if need_scores:
        if not args.comet_qe_model or not str(args.comet_qe_model).strip():
            raise SystemExit(f"输入中存在缺失 {qe_field} 的行，请设置 --comet-qe-model 或确保上游已写入分数")
        enc = args.comet_encoder_model if args.comet_encoder_model.is_dir() else None
        eval_common.configure_offline_transformers(enc, bool(args.offline_eval_assets))
        eval_common.quiet_http_logging()
        print(f"为 {sum(1 for r in all_rows_flat if get_qe(r, qe_field) is None)} 行运行 COMET-QE …", file=sys.stderr)
        comet_score_rows(
            all_rows_flat,
            qe_field,
            args,
            args.output_dir / "comet_qe_mix_model",
        )

    out_candidates: list[dict[str, Any]] = []
    per_sample_summaries: list[dict[str, Any]] = []

    if args.manifest:
        manifest = eval_common.load_json(args.manifest)
        ip = eval_common.root() / manifest["items_jsonl"]
        manifest_items = eval_common.read_items_jsonl(ip)
        keys_order = [eval_common.result_key(it) for it in manifest_items]
        keys_iter = [k for k in keys_order if k in merged_by_key]
        tail = sorted(k for k in merged_by_key.keys() if k not in set(keys_iter))
        keys_iter.extend(tail)
    else:
        keys_iter = sorted(merged_by_key.keys())

    for key in tqdm(keys_iter, desc="mix-samples"):
        group = merged_by_key[key]
        picked, stats = select_for_sample(group, mix_cfg)
        corpus, src_lang, tgt_lang, sample_id = key
        for i, src_row in enumerate(picked):
            base = strip_candidate_specific_fields(src_row)
            out_row = {
                **base,
                "candidate_id": i,
                "hypothesis": normalize_hypothesis(
                    str(src_row.get("hypothesis", "")), normalize=bool(mix_cfg["hypothesis_normalize_for_dedupe"])
                ),
            }
            if qe_field in src_row:
                out_row[qe_field] = src_row[qe_field]
            out_candidates.append(out_row)

        per_sample_summaries.append(
            {
                "corpus": corpus,
                "pair": pair_from_item(group[0]),
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "sample_id": sample_id,
                **stats,
                "output_candidates": len(picked),
            }
        )

    write_jsonl(args.output_dir / "candidates.jsonl", out_candidates)
    meta = {
        "mix_config_resolved": mix_config_summary_for_meta(mix_cfg),
        "mix_config_path": str(args.mix_config.resolve()),
        "manifest_used": str(args.manifest.resolve()) if args.manifest else None,
        "inputs": [str(Path(x).resolve()) for x in args.inputs],
        "input_row_counts": file_row_counts,
        "samples": len(merged_by_key),
        "output_rows": len(out_candidates),
        "comet_qe_model_used": str(args.comet_qe_model) if need_scores else None,
    }
    (args.output_dir / "mix_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "mix_per_sample.jsonl", per_sample_summaries)

    print(f"完成。candidates.jsonl → {args.output_dir / 'candidates.jsonl'}")
    print(f"元数据 → {args.output_dir / 'mix_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
