#!/usr/bin/env python3
"""
合并多份扁平 candidates.jsonl（如 sample/rag × 两条推理链路），产出新的候选集供下游 LLM rerank 再 eval。

规则（对每个 eval 样本）：
  1) 按配置过滤：长度比在 [min_len_ratio, max_len_ratio] 之外则丢弃该行候选。
  2) 完全相同（归一化后）的译文只保留一条；若多条重复保留 QE 较高的那条对应的整行。
  3) 若仍多于 max_candidates 条，仅保留 QE 分数最高的 max_candidates 条。
  4) 若步骤 1 后一条都不剩：从合并前的全部候选中保留 QE 最高的一条。

配置：默认读取仓库根目录下 configs/mix_hypothesis_candidates.json（可用 --mix-config 指定）。

QE 字段：默认 comet_qe。若上游是 LLM rerank 跑出来的 candidates 且无分数，可加
  --comet-qe-model ... 在本次脚本里批量打分。

示例：
  conda run -n lowres python scripts/inference/mix_hypothesis_candidates.py \\
    --mix-config configs/mix_hypothesis_candidates.json \\
    --input run/exp_a_sample_llm/candidates.jsonl \\
    --input run/exp_b_rag_llm/candidates.jsonl \\
    --output-dir eval_multilingual/my_mix/run_001 \\
    --comet-qe-model models/Unbabel_wmt22-cometkiwi-da
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
TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[\u0e00-\u0e7f]+|[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?|\d+(?:[.,]\d+)*",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_len(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


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


def load_mix_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"mix config 不存在: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("mix config JSON 必须为 object")
    allowed = {
        "min_len_ratio",
        "max_len_ratio",
        "max_candidates",
        "qe_score_field",
        "hypothesis_normalize_for_dedupe",
    }
    out = {k: v for k, v in cfg.items() if k in allowed}
    for required in ("min_len_ratio", "max_len_ratio", "max_candidates"):
        if required not in out:
            raise ValueError(f"mix config 缺少字段: {required}")
    out.setdefault("qe_score_field", "comet_qe")
    out.setdefault("hypothesis_normalize_for_dedupe", True)
    return out


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
    min_lr = float(mix["min_len_ratio"])
    max_lr = float(mix["max_len_ratio"])
    max_k = int(mix["max_candidates"])
    qe_field = str(mix["qe_score_field"])
    norm_dedupe = bool(mix["hypothesis_normalize_for_dedupe"])

    stats = {
        "input_rows": len(rows),
        "len_ok_count": 0,
        "used_len_empty_fallback": False,
        "fallback_reason": "",
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
        "mix_config_resolved": mix_cfg,
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
