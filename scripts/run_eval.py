#!/usr/bin/env python3
"""
通过 vLLM 的 OpenAI 兼容 HTTP 接口批量翻译并计算 BLEU（sacrebleu）与 COMET。
使用仓库 venv：.venv/bin/python scripts/run_eval.py ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(p: Path) -> dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_items_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mt_user_content(src_lang: str, tgt_lang: str, text: str) -> str:
    return (
        f"Translate from {src_lang} to {tgt_lang}. Output only the translation, no explanations.\n\n{text}"
    )


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    fam = (model_family or "").lower()
    if fam in ("qwen3", "qwen", "qwen3-4b"):
        # vLLM / Qwen3：关闭思考模式（若服务端不支持该字段会被忽略）
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def call_translate(
    client: OpenAI,
    model: str,
    src_lang: str,
    tgt_lang: str,
    text: str,
    max_tokens: int,
    model_family: str,
) -> str:
    extra = build_extra_body(model_family)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional machine translation engine.",
            },
            {"role": "user", "content": mt_user_content(src_lang, tgt_lang, text)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra:
        kwargs["extra_body"] = extra
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    out = (choice.message.content or "").strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM OpenAI 兼容接口 + BLEU/COMET 评估")
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
        "--served-model-name",
        type=str,
        default=os.environ.get("SERVED_MODEL_NAME", "base"),
        help="与 vLLM --served-model-name 一致",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
        help="vLLM OpenAI API base，含 /v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default=os.environ.get("EVAL_MODEL_FAMILY", "generic"),
        help="qwen3 时关闭思考模式；smollm / generic 等按普通对话",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=os.environ.get("EVAL_MODEL_TAG", "run"),
        help="输出子目录名",
    )
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=int(os.environ.get("COMET_BATCH_SIZE", "8")),
    )
    args = parser.parse_args()

    eval_cfg = load_json(args.eval_config)
    manifest = load_json(args.manifest)
    items_path = root() / manifest["items_jsonl"]
    if not items_path.is_file():
        print(f"未找到 items_jsonl: {items_path}", file=sys.stderr)
        return 1

    items = read_items_jsonl(items_path)
    if not items:
        print("评估条目为空", file=sys.stderr)
        return 1

    max_tokens = int(eval_cfg.get("max_tokens", 512))
    max_workers = int(eval_cfg.get("max_workers", 8))
    base_out = root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = base_out / f"{args.model_tag}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    results: list[dict[str, Any]] = []
    lock_results: list[dict[str, Any]] = []

    def _one(it: dict[str, Any]) -> dict[str, Any]:
        src = it["source_text"]
        ref = it["reference_text"]
        sl = it["src_lang"]
        tl = it["tgt_lang"]
        hyp = call_translate(
            client,
            args.served_model_name,
            sl,
            tl,
            src,
            max_tokens,
            args.model_family,
        )
        return {
            **it,
            "hypothesis": hyp,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, it) for it in items]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="translate"):
            lock_results.append(fut.result())

    # 保序：按原 items 顺序输出（as_completed 无序）
    def key_fn(x: dict[str, Any]) -> tuple:
        return (
            x.get("eval_corpus", x.get("dataset", "")),
            x.get("src_lang", ""),
            x.get("tgt_lang", ""),
            str(x.get("sample_id", "")),
        )
    by_key = {key_fn(r): r for r in lock_results}
    for it in items:
        results.append(by_key[key_fn(it)])

    hyp_path = run_dir / "hypotheses.jsonl"
    with open(hyp_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics: dict[str, Any] = {"by_pair": {}, "overall": {}}

    # BLEU 按 eval_pair + corpus 聚合
    try:
        import sacrebleu
    except ImportError as e:
        print(f"sacrebleu 未安装: {e}", file=sys.stderr)
        return 1

    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in results:
        key = f"{r.get('eval_corpus','?')}|{r.get('eval_pair') or r['src_lang'] + '->' + r['tgt_lang']}"
        grouped[key].append((r["hypothesis"], r["reference_text"], r["source_text"]))

    bleu_rows: list[dict[str, Any]] = []
    for key, triples in grouped.items():
        hyps = [t[0] for t in triples]
        refs = [t[1] for t in triples]
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        corp, pair = key.split("|", 1)
        row = {
            "corpus": corp,
            "pair": pair,
            "bleu": float(bleu.score),
            "num": len(hyps),
        }
        bleu_rows.append(row)
        metrics["by_pair"][key] = {"bleu": row["bleu"], "num": row["num"]}

    overall_h = [r["hypothesis"] for r in results]
    overall_r = [r["reference_text"] for r in results]
    overall_bleu = sacrebleu.corpus_bleu(overall_h, [overall_r])
    metrics["overall"]["bleu"] = float(overall_bleu.score)

    # COMET：按组计算均值；无 GPU 时 gpus=0
    comet_scores_by_key: dict[str, float] = {}
    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        print(f"unbabel-comet/torch 不可用，跳过 COMET: {e}", file=sys.stderr)
    else:
        gpus = 1 if torch.cuda.is_available() else 0
        ckpt = download_model("Unbabel/wmt22-comet-da", saving_directory=str(run_dir / "comet_ckpt"))
        comet_model = load_from_checkpoint(ckpt)
        for key, triples in grouped.items():
            data = [{"src": t[2], "mt": t[0], "ref": t[1]} for t in triples]
            out = comet_model.predict(data, batch_size=args.comet_batch_size, gpus=gpus)
            scores = out.get("scores", [])
            comet_scores_by_key[key] = float(sum(scores) / max(1, len(scores)))
            metrics["by_pair"][key]["comet"] = comet_scores_by_key[key]

        all_data = [{"src": r["source_text"], "mt": r["hypothesis"], "ref": r["reference_text"]} for r in results]
        out_all = comet_model.predict(all_data, batch_size=args.comet_batch_size, gpus=gpus)
        scores_all = out_all.get("scores", [])
        metrics["overall"]["comet"] = float(sum(scores_all) / max(1, len(scores_all)))

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "pair", "bleu", "comet", "num"])
        w.writeheader()
        for row in bleu_rows:
            key = f"{row['corpus']}|{row['pair']}"
            w.writerow(
                {
                    **row,
                    "comet": comet_scores_by_key.get(key, ""),
                }
            )

    print(f"完成。输出目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
