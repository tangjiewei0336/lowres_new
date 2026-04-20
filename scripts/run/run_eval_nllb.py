#!/usr/bin/env python3
"""
Evaluate facebook/nllb-200-3.3B, facebook/nllb-moe-54b, or a local NLLB
checkpoint on the existing eval manifest, then write hypotheses.jsonl and
BLEU/COMET metrics in the same format as scripts/run/run_eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

import run_eval as eval_common
from run_eval_apertium import write_metrics


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_model_path(model_name_or_path: str) -> str:
    p = Path(model_name_or_path)
    if p.exists():
        return str(p)
    local_defaults = {
        "facebook/nllb-200-3.3B": root() / "models" / "facebook_nllb-200-3.3B",
        "facebook/nllb-moe-54b": root() / "models" / "facebook_nllb-moe-54b",
    }
    local_default = local_defaults.get(model_name_or_path)
    if local_default is not None and local_default.exists():
        return str(local_default)
    return model_name_or_path


def choose_device(torch_mod: Any, device_arg: str) -> Any:
    if device_arg != "auto":
        return torch_mod.device(device_arg)
    if torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return torch_mod.device("mps")
    return torch_mod.device("cpu")


def choose_dtype(torch_mod: Any, dtype_arg: str, device: Any) -> Any:
    if dtype_arg == "fp16":
        return torch_mod.float16
    if dtype_arg == "bf16":
        return torch_mod.bfloat16
    if dtype_arg == "fp32":
        return torch_mod.float32
    if device.type == "cuda":
        return torch_mod.float16
    return torch_mod.float32


def infer_input_device(torch_mod: Any, model: Any, fallback_device: Any) -> Any:
    """Pick a concrete tensor device for inputs, including accelerate device_map models."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and str(emb.weight.device) != "meta":
            return emb.weight.device
    except Exception:
        pass
    for param in model.parameters():
        if str(param.device) != "meta":
            return param.device
    if torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    return fallback_device


def forced_bos_token_id(tokenizer: Any, tgt_lang: str) -> int:
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and tgt_lang in lang_code_to_id:
        return int(lang_code_to_id[tgt_lang])
    tok_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if tok_id is None or tok_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer 不支持目标语言代码: {tgt_lang}")
    return int(tok_id)


def grouped_items(items: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        out[(str(it["src_lang"]), str(it["tgt_lang"]))].append(it)
    return out


def translate_batch(
    *,
    torch_mod: Any,
    model: Any,
    tokenizer: Any,
    device: Any,
    src_lang: str,
    tgt_lang: str,
    texts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    tokenizer.src_lang = src_lang
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch_mod.inference_mode():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id(tokenizer, tgt_lang),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return [
        s.strip()
        for s in tokenizer.batch_decode(generated, skip_special_tokens=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="NLLB-200 / NLLB-MoE + BLEU/COMET 评估")
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
        "--model-name-or-path",
        default=str(root() / "models" / "facebook_nllb-200-3.3B"),
        help=(
            "本地模型目录或 Hugging Face repo id，例如 facebook/nllb-200-3.3B "
            "或 facebook/nllb-moe-54b。"
        ),
    )
    parser.add_argument("--model-tag", default="nllb_200_3_3b")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu/mps/cuda:0 等")
    parser.add_argument(
        "--device-map",
        default="none",
        help="传给 from_pretrained 的 device_map；54B MoE 常用 auto。默认 none。",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=root() / "models" / "offload" / "nllb_moe_54b",
        help="device_map/offload 使用的磁盘目录。",
    )
    parser.add_argument(
        "--no-low-cpu-mem-usage",
        action="store_true",
        help="关闭 transformers low_cpu_mem_usage。",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp16", "bf16", "fp32"),
        default="auto",
    )
    parser.add_argument("--comet-batch-size", type=int, default=8)
    parser.add_argument(
        "--comet-model",
        default="models/Unbabel_wmt22-comet-da",
        help="同 run_eval.py；设为 none 可只跑 BLEU。",
    )
    parser.add_argument(
        "--bleu-tokenize",
        choices=("auto", "flores200", "legacy"),
        default="auto",
        help="同 run_eval.py。",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        print(
            f"缺少 NLLB 评估依赖: {e}\n"
            "请先安装 requirements.txt，或至少安装 torch、transformers、sentencepiece、accelerate。",
            file=sys.stderr,
        )
        return 1

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

    model_path = resolve_model_path(args.model_name_or_path)
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, args.dtype, device)
    device_map = None if args.device_map.lower() in ("none", "off", "false") else args.device_map
    print(
        f"加载 NLLB: {model_path} device={device} dtype={dtype} device_map={device_map}",
        file=sys.stderr,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": not args.no_low_cpu_mem_usage,
    }
    if device_map is not None:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        model_kwargs["device_map"] = device_map
        model_kwargs["offload_folder"] = str(args.offload_folder)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
    input_device = infer_input_device(torch, model, device)
    if device_map is None:
        model.to(device)
        input_device = device
    model.eval()

    base_out = root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = base_out / f"{args.model_tag}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(eval_cfg.get("max_tokens", 512))

    results_by_key: dict[tuple, dict[str, Any]] = {}
    groups = grouped_items(items)
    total_batches = sum((len(rows) + args.batch_size - 1) // args.batch_size for rows in groups.values())

    with tqdm(total=total_batches, desc="translate") as pbar:
        for (src_lang, tgt_lang), rows in sorted(groups.items()):
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                hyps = translate_batch(
                    torch_mod=torch,
                    model=model,
                    tokenizer=tokenizer,
                    device=input_device,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    texts=[str(it["source_text"]) for it in batch],
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=max_new_tokens,
                    num_beams=args.num_beams,
                )
                for it, hyp in zip(batch, hyps, strict=True):
                    key = (
                        it.get("eval_corpus", it.get("dataset", "")),
                        it.get("src_lang", ""),
                        it.get("tgt_lang", ""),
                        str(it.get("sample_id", "")),
                    )
                    results_by_key[key] = {**it, "hypothesis": hyp}
                pbar.update(1)

    results: list[dict[str, Any]] = []
    for it in items:
        key = (
            it.get("eval_corpus", it.get("dataset", "")),
            it.get("src_lang", ""),
            it.get("tgt_lang", ""),
            str(it.get("sample_id", "")),
        )
        results.append(results_by_key[key])

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
