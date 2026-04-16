#!/usr/bin/env python3
"""
Evaluate the offline Apertium English-Spanish rule-based MT package with the
same BLEU/COMET aggregation format used by scripts/run/run_eval.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

import run_eval as eval_common


SUPPORTED_MODES = {
    ("eng_Latn", "spa_Latn"): "en-es",
    ("spa_Latn", "eng_Latn"): "es-en",
}


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def apertium_mode(src_lang: str, tgt_lang: str) -> str | None:
    return SUPPORTED_MODES.get((src_lang, tgt_lang))


def normalize_for_stdin(text: str) -> str:
    return " ".join((text or "").splitlines()).strip()


def translate_one(jar_path: Path, mode: str, text: str, timeout_s: int) -> str:
    proc = subprocess.run(
        ["java", "-jar", str(jar_path), "apertium", mode],
        input=normalize_for_stdin(text) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Apertium failed for mode={mode} rc={proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def write_metrics(
    *,
    results: list[dict[str, Any]],
    run_dir: Path,
    bleu_tokenize: str,
    comet_model_arg: str,
    comet_batch_size: int,
) -> None:
    metrics: dict[str, Any] = {
        "by_corpus": {},
        "overall": {},
    }

    try:
        import sacrebleu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"sacrebleu 未安装: {e}") from e

    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in results:
        key = f"{r.get('eval_corpus','?')}|{r.get('eval_pair') or r['src_lang'] + '->' + r['tgt_lang']}"
        grouped[key].append((r["hypothesis"], r["reference_text"], r["source_text"]))

    bleu_rows: list[dict[str, Any]] = []
    corpus_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, triples in grouped.items():
        hyps = [t[0] for t in triples]
        refs = [t[1] for t in triples]
        corp, pair = key.split("|", 1)
        bleu_score, bleu_tok = eval_common.corpus_bleu_with_fallbacks(
            hyps, refs, corp, pair, bleu_tokenize
        )
        if corp not in metrics["by_corpus"]:
            metrics["by_corpus"][corp] = {"by_pair": {}, "overall": {}}
        row = {
            "corpus": corp,
            "pair": pair,
            "bleu": bleu_score,
            "bleu_tokenizer": bleu_tok,
            "num": len(hyps),
        }
        bleu_rows.append(row)
        corpus_to_rows[corp].append(row)
        metrics["by_corpus"][corp]["by_pair"][pair] = {
            "bleu": row["bleu"],
            "bleu_tokenizer": bleu_tok,
            "num": row["num"],
        }

    total_n = sum(r["num"] for r in bleu_rows)
    metrics["overall"]["bleu"] = (
        float(sum(r["bleu"] * r["num"] for r in bleu_rows) / total_n) if total_n else 0.0
    )

    corpus_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        corpus_group[str(r.get("eval_corpus", "?"))].append(r)
    for corp, rows_c in corpus_to_rows.items():
        n_c = sum(r["num"] for r in rows_c)
        bleu_c = float(sum(r["bleu"] * r["num"] for r in rows_c) / n_c) if n_c else 0.0
        metrics["by_corpus"][corp]["overall"]["bleu"] = bleu_c
    for corp, rr in corpus_group.items():
        if corp not in metrics["by_corpus"]:
            metrics["by_corpus"][corp] = {"by_pair": {}, "overall": {}}
        metrics["by_corpus"][corp]["overall"]["num"] = len(rr)

    comet_scores_by_key: dict[str, float] = {}
    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        print(f"unbabel-comet/torch 不可用，跳过 COMET: {e}", file=sys.stderr)
    else:
        if comet_model_arg.lower() in ("none", "off", "disable", "disabled"):
            print("COMET 已禁用（--comet-model=none）。", file=sys.stderr)
        else:
            gpus = 1 if torch.cuda.is_available() else 0
            comet_ckpt: str | None = None
            p = Path(comet_model_arg)
            if p.exists():
                if p.is_dir():
                    cand = p / "checkpoints" / "model.ckpt"
                    if cand.is_file():
                        comet_ckpt = str(cand)
                    else:
                        ckpts = list(p.glob("**/*.ckpt"))
                        if ckpts:
                            comet_ckpt = str(ckpts[0])
                else:
                    comet_ckpt = str(p)
            else:
                try:
                    comet_ckpt = download_model(
                        comet_model_arg,
                        saving_directory=str(run_dir / "comet_ckpt"),
                    )
                except Exception as e:
                    print(f"COMET 模型下载失败，将跳过 COMET: {e}", file=sys.stderr)

            if comet_ckpt:
                comet_model = load_from_checkpoint(comet_ckpt)
                all_data = [
                    {"src": r["source_text"], "mt": r["hypothesis"], "ref": r["reference_text"]}
                    for r in results
                ]
                out_all = comet_model.predict(all_data, batch_size=comet_batch_size, gpus=gpus)
                scores_all = out_all.get("scores", [])
                if not isinstance(scores_all, list) or len(scores_all) != len(results):
                    print("COMET 输出 scores 长度异常，将跳过 COMET。", file=sys.stderr)
                else:
                    metrics["overall"]["comet"] = float(sum(scores_all) / max(1, len(scores_all)))
                    sums: dict[str, float] = defaultdict(float)
                    cnts: dict[str, int] = defaultdict(int)
                    sums_corpus: dict[str, float] = defaultdict(float)
                    cnts_corpus: dict[str, int] = defaultdict(int)
                    for r, s in zip(results, scores_all, strict=True):
                        corp = str(r.get("eval_corpus", "?"))
                        pair = str(r.get("eval_pair") or (r["src_lang"] + "->" + r["tgt_lang"]))
                        k = f"{corp}|{pair}"
                        sv = float(s)
                        sums[k] += sv
                        cnts[k] += 1
                        sums_corpus[corp] += sv
                        cnts_corpus[corp] += 1
                    for k, total in sums.items():
                        comet_scores_by_key[k] = float(total / max(1, cnts[k]))
                        corp, pair = k.split("|", 1)
                        metrics["by_corpus"][corp]["by_pair"].setdefault(pair, {})
                        metrics["by_corpus"][corp]["by_pair"][pair]["comet"] = comet_scores_by_key[k]
                    for corp, total in sums_corpus.items():
                        metrics["by_corpus"][corp]["overall"]["comet"] = float(
                            total / max(1, cnts_corpus[corp])
                        )

    metrics["bleu_tokenize_policy"] = bleu_tokenize

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(run_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["corpus", "pair", "bleu", "bleu_tokenizer", "comet", "num"],
        )
        w.writeheader()
        for row in bleu_rows:
            key = f"{row['corpus']}|{row['pair']}"
            w.writerow({**row, "comet": comet_scores_by_key.get(key, "")})

    for corp, rows in corpus_to_rows.items():
        with open(run_dir / f"metrics_{corp}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["pair", "bleu", "bleu_tokenizer", "comet", "num"],
            )
            w.writeheader()
            for row in sorted(rows, key=lambda x: x["pair"]):
                key = f"{corp}|{row['pair']}"
                w.writerow(
                    {
                        "pair": row["pair"],
                        "bleu": row["bleu"],
                        "bleu_tokenizer": row["bleu_tokenizer"],
                        "comet": comet_scores_by_key.get(key, ""),
                        "num": row["num"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apertium en-es/es-en + BLEU/COMET 评估")
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=root() / "evaluation_config.json",
        help="evaluation_config.json 路径",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root() / "datasets" / "eval_manifest.json",
        help="expand_language_pairs 生成的 manifest",
    )
    parser.add_argument(
        "--jar-path",
        type=Path,
        default=root() / "models" / "apertium-en-es" / "apertium-en-es.jar",
        help="Apertium English-Spanish standalone JAR",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="apertium_en_es",
        help="输出子目录名",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=60,
        help="单句翻译超时秒数",
    )
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--comet-model",
        type=str,
        default="models/Unbabel_wmt22-comet-da",
        help="同 run_eval.py；设为 none 可只跑 BLEU。",
    )
    parser.add_argument(
        "--bleu-tokenize",
        type=str,
        choices=("auto", "flores200", "legacy"),
        default="auto",
        help="同 run_eval.py。",
    )
    args = parser.parse_args()

    if not args.jar_path.is_file():
        print(f"未找到 Apertium JAR: {args.jar_path}", file=sys.stderr)
        return 1

    eval_cfg = eval_common.load_json(args.eval_config)
    manifest = eval_common.load_json(args.manifest)
    items_path = root() / manifest["items_jsonl"]
    if not items_path.is_file():
        print(f"未找到 items_jsonl: {items_path}", file=sys.stderr)
        return 1

    all_items = eval_common.read_items_jsonl(items_path)
    items = [
        it
        for it in all_items
        if apertium_mode(str(it["src_lang"]), str(it["tgt_lang"])) is not None
    ]
    skipped = len(all_items) - len(items)
    if not items:
        print("manifest 中没有 Apertium en-es/es-en 支持的评估方向。", file=sys.stderr)
        return 1
    if skipped:
        print(f"跳过不支持的语向样本: {skipped}; 评估样本: {len(items)}", file=sys.stderr)

    base_out = root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = base_out / f"{args.model_tag}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for it in tqdm(items, desc="translate"):
        mode = apertium_mode(str(it["src_lang"]), str(it["tgt_lang"]))
        if mode is None:
            continue
        hyp = translate_one(args.jar_path, mode, str(it["source_text"]), args.timeout_s)
        results.append({**it, "hypothesis": hyp, "apertium_mode": mode})

    with open(run_dir / "hypotheses.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_metrics(
        results=results,
        run_dir=run_dir,
        bleu_tokenize=args.bleu_tokenize,
        comet_model_arg=args.comet_model,
        comet_batch_size=args.comet_batch_size,
    )

    print(f"完成。输出目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
