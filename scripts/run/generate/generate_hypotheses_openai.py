#!/usr/bin/env python3
"""
使用 OpenAI 兼容 API 按 eval manifest 批量生成 hypothesis，并写成与 run_eval.py 对齐的 hypotheses.jsonl。

示例：
  python scripts/run/generate/generate_hypotheses_openai.py \
    --base-url "" \
    --api-key "$OPENAI_API_KEY" \
    --model your-model-name \
    --model-tag ext_api_qwen
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

from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import run_eval as eval_common


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI 兼容 API 批量生成 hypotheses.jsonl")
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=eval_common.root() / "evaluation_config.json",
        help="evaluation_config.json 路径",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=eval_common.root() / "datasets" / "eval_manifest.json",
        help="expand_language_pairs 生成的 manifest",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", ""),
        help="OpenAI 兼容 API base，留空则走官方默认端点。",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="API 侧模型名",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default=os.environ.get("EVAL_MODEL_FAMILY", "generic"),
        help="qwen3 时会自动关闭 thinking，与 run_eval.py 保持一致。",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=os.environ.get("EVAL_MODEL_TAG", "openai_api"),
        help="输出子目录名",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=None,
        help="可直接指定输出目录；默认 eval output_dir/model_tag_timestamp",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="并发数；默认读取 evaluation_config.json 中的 max_workers。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="生成长度；默认读取 evaluation_config.json 中的 max_tokens。",
    )
    parser.add_argument(
        "--http-log-level",
        type=str,
        default=os.environ.get("HTTP_LOG_LEVEL", "WARNING"),
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        help="OpenAI/httpx/httpcore 日志级别。",
    )
    parser.add_argument(
        "--dump-segmentation",
        action="store_true",
        help="额外写出分词后的 hypothesis/reference，便于人工排查 BLEU。",
    )
    parser.add_argument(
        "--flores200-spm-model",
        type=Path,
        default=Path(os.environ.get("FLORES200_SPM_MODEL", str(eval_common.default_flores200_spm_path()))),
        help="本地 FLORES200/spBLEU SentencePiece 模型路径。",
    )
    args = parser.parse_args()

    eval_common.quiet_http_logging(str(args.http_log_level))
    eval_cfg = eval_common.load_json(args.eval_config)
    manifest = eval_common.load_json(args.manifest)
    items_path = eval_common.root() / manifest["items_jsonl"]
    if not items_path.is_file():
        print(f"未找到 items_jsonl: {items_path}", file=sys.stderr)
        return 1

    items = eval_common.read_items_jsonl(items_path)
    if not items:
        print("评估条目为空", file=sys.stderr)
        return 1

    max_tokens = int(args.max_tokens or eval_cfg.get("max_tokens", 512))
    max_workers = int(args.max_workers or eval_cfg.get("max_workers", 8))
    base_out = eval_common.root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = args.output_run_dir or (base_out / f"{args.model_tag}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    client_kwargs: dict[str, Any] = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    flores200_spm_model = args.flores200_spm_model if args.flores200_spm_model.is_file() else None
    results_unordered: list[dict[str, Any]] = []

    def _one(it: dict[str, Any]) -> dict[str, Any]:
        src = it["source_text"]
        ref = it["reference_text"]
        sl = it["src_lang"]
        tl = it["tgt_lang"]
        hyp = eval_common.call_translate(
            client,
            args.model,
            sl,
            tl,
            src,
            max_tokens,
            args.model_family,
        )
        out = {**it, "hypothesis": hyp}
        if args.dump_segmentation:
            corp = str(it.get("eval_corpus", "?"))
            pair = str(it.get("eval_pair") or (sl + "->" + tl))
            tok = eval_common.sacrebleu_tokenize_for_group(corp, pair, "auto")
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

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, it) for it in items]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="translate"):
            results_unordered.append(fut.result())

    by_key = {eval_common.result_key(r): r for r in results_unordered}
    results = [by_key[eval_common.result_key(it)] for it in items]

    hyp_path = run_dir / "hypotheses.jsonl"
    eval_common.write_hypotheses_jsonl(hyp_path, results)
    meta = {
        "model": args.model,
        "model_family": args.model_family,
        "base_url": args.base_url,
        "num_samples": len(results),
        "manifest": str(args.manifest),
        "eval_config": str(args.eval_config),
    }
    with open(run_dir / "generation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"完成。hypotheses: {hyp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
