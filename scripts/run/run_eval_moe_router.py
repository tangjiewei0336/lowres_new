#!/usr/bin/env python3
"""
Evaluate a pair-level LoRA MoE deployment served by vLLM/OpenAI-compatible API.

The script reads training/moe_router_manifest.json, routes each eval sample to
its matching adapter_name by (src_lang, tgt_lang), writes hypotheses.jsonl, and
then reuses the standard BLEU/COMET aggregation logic.
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

import run_eval as eval_common
from run_eval_apertium import write_metrics


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_router_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    experts = data.get("experts")
    if not isinstance(experts, list) or not experts:
        raise SystemExit(f"No experts found in manifest: {path}")
    return data


def build_router(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    router: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest.get("experts", []):
        if not isinstance(item, dict):
            continue
        src = str(item.get("src_lang", "")).strip()
        tgt = str(item.get("tgt_lang", "")).strip()
        name = str(item.get("adapter_name", "")).strip()
        if not src or not tgt or not name:
            continue
        router[(src, tgt)] = item
    if not router:
        raise SystemExit("Manifest has no usable expert routes.")
    return router


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE router + BLEU/COMET 评估")
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
        help="expand_language_pairs 生成的 eval manifest",
    )
    parser.add_argument(
        "--router-manifest",
        type=Path,
        default=root() / "training" / "moe_router_manifest.json",
        help="MoE language-pair router manifest",
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
        default=os.environ.get("EVAL_MODEL_FAMILY", "qwen3"),
        help="qwen3/qwen3.5 时关闭 thinking。",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=os.environ.get("EVAL_MODEL_TAG", "moe_router"),
        help="输出子目录名",
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
        help="OpenAI/httpx/httpcore 日志级别。",
    )
    parser.add_argument(
        "--flores200-spm-model",
        type=Path,
        default=Path(os.environ.get("FLORES200_SPM_MODEL", str(eval_common.default_flores200_spm_path()))),
        help="本地 FLORES200/spBLEU SentencePiece 模型路径。",
    )
    parser.add_argument(
        "--comet-encoder-model",
        type=Path,
        default=Path(os.environ.get("COMET_ENCODER_MODEL", str(root() / "models" / "xlm-roberta-large"))),
        help="COMET encoder 本地路径。",
    )
    parser.add_argument(
        "--offline-eval-assets",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OFFLINE_EVAL_ASSETS", "1") not in ("0", "false", "False"),
        help="默认开启：让 COMET/FLORES 评测优先使用本地离线资产。",
    )
    parser.add_argument(
        "--dump-segmentation",
        action="store_true",
        help="在 hypotheses.jsonl 中额外写出分词后的 hypothesis/reference。",
    )
    args = parser.parse_args()

    eval_common.quiet_http_logging(str(args.http_log_level))
    eval_common.configure_offline_transformers(args.comet_encoder_model, bool(args.offline_eval_assets))
    flores200_spm_model = args.flores200_spm_model if args.flores200_spm_model.is_file() else None
    if flores200_spm_model:
        print(f"BLEU: 使用本地 FLORES200 SPM: {flores200_spm_model}", file=sys.stderr)

    eval_cfg = eval_common.load_json(args.eval_config)
    manifest = eval_common.load_json(args.manifest)
    items_path = root() / manifest["items_jsonl"]
    if not items_path.is_file():
        print(f"未找到 items_jsonl: {items_path}", file=sys.stderr)
        return 1
    items = eval_common.read_items_jsonl(items_path)
    if not items:
        print("评估条目为空", file=sys.stderr)
        return 1

    router_manifest = load_router_manifest(args.router_manifest)
    router = build_router(router_manifest)

    missing_pairs = sorted({
        f"{it['src_lang']}->{it['tgt_lang']}"
        for it in items
        if (str(it["src_lang"]), str(it["tgt_lang"])) not in router
    })
    if missing_pairs:
        print(
            "MoE router manifest 缺少以下语向: " + ", ".join(missing_pairs),
            file=sys.stderr,
        )
        return 1

    max_tokens = int(args.max_tokens or eval_cfg.get("max_tokens", 512))
    max_workers = int(args.max_workers or eval_cfg.get("max_workers", 8))
    base_out = root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = base_out / f"{args.model_tag}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    results_unordered: list[dict[str, Any]] = []

    def _one(it: dict[str, Any]) -> dict[str, Any]:
        sl = str(it["src_lang"])
        tl = str(it["tgt_lang"])
        route = router[(sl, tl)]
        model_name = str(route["adapter_name"])
        src = str(it["source_text"])
        ref = str(it["reference_text"])
        hyp = eval_common.call_translate(
            client,
            model_name,
            sl,
            tl,
            src,
            max_tokens,
            args.model_family,
        )
        out = {**it, "hypothesis": hyp, "routed_model": model_name}
        if args.dump_segmentation:
            corp = str(it.get("eval_corpus", "?"))
            pair = str(it.get("eval_pair") or (sl + "->" + tl))
            tok = eval_common.sacrebleu_tokenize_for_group(corp, pair, str(args.bleu_tokenize))
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

    eval_common.write_hypotheses_jsonl(run_dir / "hypotheses.jsonl", results)

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
