#!/usr/bin/env python3
"""
Evaluate Zhipu GLM models on FLORES via the official zai client.

The script:
1. Reads the existing eval manifest/items_jsonl.
2. Optionally filters to a target corpus (default: flores).
3. Calls Zhipu chat.completions for each sample with English MT instructions.
4. Writes hypotheses.jsonl and standard BLEU/COMET metrics.

Thinking is disabled by default and should remain off for MT evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

import run_eval as eval_common
from run_eval_apertium import write_metrics


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_client(api_key: str) -> Any:
    from zai import ZhipuAiClient

    return ZhipuAiClient(api_key=api_key)


def is_zhipu_content_filter_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        '"code":"1301"' in text
        or "'code':'1301'" in text
        or "contentFilter" in text
        or "可能包含不安全或敏感内容" in text
    )


def is_zhipu_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        '"code":"1302"' in text
        or "'code':'1302'" in text
        or "Error code: 429" in text
        or "达到速率限制" in text
        or "请求频率" in text
    )


def call_translate_zhipu(
    *,
    api_key: str,
    model: str,
    src_lang: str,
    tgt_lang: str,
    text: str,
    max_tokens: int,
    temperature: float,
    retries: int,
) -> str:
    client = build_client(api_key)
    messages = [
        {"role": "system", "content": "You are a professional machine translation engine."},
        {"role": "user", "content": eval_common.mt_user_content(src_lang, tgt_lang, text)},
    ]
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                thinking={"type": "disabled"},
                max_tokens=max_tokens,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content.strip()
            if isinstance(msg, dict):
                return str(msg.get("content", "")).strip()
            return str(content or "").strip()
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            if is_zhipu_rate_limit_error(e):
                time.sleep(min(5.0 * (attempt + 1), 30.0))
            else:
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    assert last_err is not None
    raise last_err


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Zhipu API on FLORES with BLEU/COMET.")
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
        "--api-key",
        type=str,
        default=os.environ.get("ZHIPU_API_KEY", ""),
        help="智谱 API key；默认读取 ZHIPU_API_KEY。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("ZHIPU_MODEL", "glm-5.1"),
        help="智谱模型名，例如 glm-5.1。",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=os.environ.get("EVAL_MODEL_TAG", "zhipu_api"),
        help="输出子目录名",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=None,
        help="可直接指定输出目录；配合 --resume 时建议固定这个目录。",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="flores",
        help="只评这个 corpus；默认 flores。",
    )
    parser.add_argument("--max-workers", type=int, default=0, help="并发数；默认读取 evaluation_config.json。")
    parser.add_argument("--max-tokens", type=int, default=0, help="生成长度；默认读取 evaluation_config.json。")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--comet-batch-size", type=int, default=int(os.environ.get("COMET_BATCH_SIZE", "8")))
    parser.add_argument(
        "--comet-model",
        type=str,
        default=os.environ.get("COMET_MODEL", "models/Unbabel_wmt22-comet-da"),
        help="同 run_eval.py；设为 none 可只跑 BLEU。",
    )
    parser.add_argument(
        "--bleu-tokenize",
        type=str,
        choices=("auto", "flores200", "legacy"),
        default=os.environ.get("BLEU_TOKENIZE", "auto"),
        help="同 run_eval.py。",
    )
    parser.add_argument(
        "--http-log-level",
        type=str,
        default=os.environ.get("HTTP_LOG_LEVEL", "WARNING"),
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
    )
    parser.add_argument(
        "--flores200-spm-model",
        type=Path,
        default=Path(os.environ.get("FLORES200_SPM_MODEL", str(eval_common.default_flores200_spm_path()))),
    )
    parser.add_argument(
        "--comet-encoder-model",
        type=Path,
        default=Path(os.environ.get("COMET_ENCODER_MODEL", str(root() / "models" / "xlm-roberta-large"))),
    )
    parser.add_argument(
        "--offline-eval-assets",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OFFLINE_EVAL_ASSETS", "1") not in ("0", "false", "False"),
    )
    parser.add_argument("--dump-segmentation", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若输出目录已有 hypotheses.jsonl，则跳过已完成样本并继续生成。",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key or ZHIPU_API_KEY.")

    eval_common.quiet_http_logging(str(args.http_log_level))
    eval_common.configure_offline_transformers(args.comet_encoder_model, bool(args.offline_eval_assets))
    flores200_spm_model = args.flores200_spm_model if args.flores200_spm_model.is_file() else None

    eval_cfg = eval_common.load_json(args.eval_config)
    manifest = eval_common.load_json(args.manifest)
    items_path = root() / manifest["items_jsonl"]
    if not items_path.is_file():
        print(f"未找到 items_jsonl: {items_path}", file=sys.stderr)
        return 1

    items = eval_common.read_items_jsonl(items_path)
    corpus = str(args.corpus).strip().lower()
    if corpus:
        items = [it for it in items if str(it.get("eval_corpus", "")).strip().lower() == corpus]
    if not items:
        print(f"评估条目为空（corpus={args.corpus}）。", file=sys.stderr)
        return 1

    max_tokens = int(args.max_tokens or eval_cfg.get("max_tokens", 512))
    max_workers = int(args.max_workers or eval_cfg.get("max_workers", 8))
    base_out = root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = args.output_run_dir or (base_out / f"{args.model_tag}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = run_dir / "hypotheses.jsonl"

    existing_results: list[dict[str, Any]] = []
    existing_by_key: dict[tuple, dict[str, Any]] = {}
    pending_items = list(items)
    if args.resume and hyp_path.is_file():
        try:
            existing_results = eval_common.load_hypotheses_jsonl(hyp_path)
            existing_by_key = {eval_common.result_key(r): r for r in existing_results}
            pending_items = [it for it in items if eval_common.result_key(it) not in existing_by_key]
            print(
                f"resume: found {len(existing_results)} existing hypotheses, remaining {len(pending_items)} samples",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"resume: failed to read existing hypotheses, start fresh: {e}", file=sys.stderr)
            existing_results = []
            existing_by_key = {}
            pending_items = list(items)

    results_unordered: list[dict[str, Any]] = []
    skipped_content_filter: list[dict[str, Any]] = []

    def _one(it: dict[str, Any]) -> dict[str, Any]:
        src = str(it["source_text"])
        ref = str(it["reference_text"])
        sl = str(it["src_lang"])
        tl = str(it["tgt_lang"])
        hyp = call_translate_zhipu(
            api_key=args.api_key,
            model=args.model,
            src_lang=sl,
            tgt_lang=tl,
            text=src,
            max_tokens=max_tokens,
            temperature=float(args.temperature),
            retries=int(args.max_retries),
        )
        out = {**it, "hypothesis": hyp}
        if args.dump_segmentation:
            pair = str(it.get("eval_pair") or (sl + "->" + tl))
            tok = eval_common.sacrebleu_tokenize_for_group(str(it.get("eval_corpus", "?")), pair, str(args.bleu_tokenize))
            if tok == eval_common._THAI_SACREbleu_TOK:
                out["hypothesis_segmented"] = eval_common._segment_thai_pythai_words([hyp])[0]
                out["reference_segmented"] = eval_common._segment_thai_pythai_words([ref])[0]
                out["segmentation_tokenizer"] = tok
            elif tok in ("zh", "flores200"):
                try:
                    out["hypothesis_segmented"] = eval_common._segment_with_sacrebleu_tokenizer(
                        [hyp],
                        tok,
                        flores200_spm_model=flores200_spm_model,
                    )[0]
                    out["reference_segmented"] = eval_common._segment_with_sacrebleu_tokenizer(
                        [ref],
                        tok,
                        flores200_spm_model=flores200_spm_model,
                    )[0]
                    out["segmentation_tokenizer"] = (
                        "flores200_local_spm"
                        if tok == "flores200" and flores200_spm_model
                        else tok
                    )
                except Exception as e:
                    out["segmentation_tokenizer"] = f"{tok}_dump_failed"
                    out["segmentation_error"] = str(e)
        return out

    if pending_items:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_item = {ex.submit(_one, it): it for it in pending_items}
            for fut in tqdm(as_completed(fut_to_item), total=len(fut_to_item), desc="translate"):
                try:
                    row = fut.result()
                except Exception as e:
                    if is_zhipu_content_filter_error(e):
                        it = fut_to_item[fut]
                        skipped_content_filter.append(
                            {
                                "eval_corpus": it.get("eval_corpus"),
                                "eval_pair": it.get("eval_pair"),
                                "sample_id": it.get("sample_id"),
                                "src_lang": it.get("src_lang"),
                                "tgt_lang": it.get("tgt_lang"),
                                "source_text": it.get("source_text"),
                                "reference_text": it.get("reference_text"),
                                "error": str(e),
                            }
                        )
                        continue
                    raise
                results_unordered.append(row)
        if args.resume and existing_results:
            merged = list(existing_results)
            merged.extend(results_unordered)
            merged.sort(key=eval_common.result_key)
            eval_common.write_hypotheses_jsonl(hyp_path, merged)
        else:
            eval_common.write_hypotheses_jsonl(hyp_path, results_unordered)
    elif not hyp_path.is_file():
        eval_common.write_hypotheses_jsonl(hyp_path, [])

    all_results = eval_common.load_hypotheses_jsonl(hyp_path)
    by_key = {eval_common.result_key(r): r for r in all_results}
    results = [by_key[eval_common.result_key(it)] for it in items if eval_common.result_key(it) in by_key]

    meta = {
        "provider": "zhipu",
        "model": args.model,
        "thinking": "disabled",
        "num_samples": len(results),
        "manifest": str(args.manifest),
        "eval_config": str(args.eval_config),
        "corpus": args.corpus,
        "resume": bool(args.resume),
        "max_retries": int(args.max_retries),
        "skipped_content_filter": len(skipped_content_filter),
    }
    with open(run_dir / "generation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if skipped_content_filter:
        skipped_path = run_dir / "skipped_content_filter.jsonl"
        with open(skipped_path, "w", encoding="utf-8") as f:
            for row in skipped_content_filter:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"skipped due to content filter: {len(skipped_content_filter)} -> {skipped_path}", file=sys.stderr)

    write_metrics(
        results=results,
        run_dir=run_dir,
        bleu_tokenize=args.bleu_tokenize,
        comet_model_arg=args.comet_model,
        comet_batch_size=args.comet_batch_size,
        flores200_spm_model=flores200_spm_model,
        comet_encoder_model=args.comet_encoder_model if args.comet_encoder_model.is_dir() else None,
        offline_eval_assets=bool(args.offline_eval_assets),
    )

    print(f"完成。输出目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
