from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from agent_runtime import LangGraphDictionaryAgentRuntime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_items_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def result_key(x: dict) -> tuple:
    return (
        x.get("eval_corpus", x.get("dataset", "")),
        x.get("src_lang", ""),
        x.get("tgt_lang", ""),
        str(x.get("sample_id", "")),
    )


async def main_async() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the dictionary tool-calling translation agent on FLORES.")
    ap.add_argument(
        "--eval-config",
        type=Path,
        default=repo_root() / "evaluation_config.json",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=repo_root() / "datasets" / "eval_manifest.json",
    )
    ap.add_argument("--corpus", default="flores")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--model-family", default="qwen3")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument(
        "--lexicon-dir",
        type=Path,
        default=repo_root() / "training" / "data" / "dictionaries" / "moe_lexicon",
    )
    ap.add_argument("--model-tag", default="dictionary_tool_agent")
    ap.add_argument("--output-run-dir", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true", default=False, help="打印每条样本组装后的提示词。")
    ap.add_argument("--comet-batch-size", type=int, default=8)
    ap.add_argument("--comet-model", default="models/Unbabel_wmt22-comet-da")
    ap.add_argument("--bleu-tokenize", choices=("auto", "flores200", "legacy"), default="auto")
    ap.add_argument(
        "--flores200-spm-model",
        type=Path,
        default=repo_root() / "models" / "sacrebleu" / "flores200_sacrebleu_tokenizer_spm.model",
    )
    ap.add_argument(
        "--comet-encoder-model",
        type=Path,
        default=repo_root() / "models" / "xlm-roberta-large",
    )
    ap.add_argument("--offline-eval-assets", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    scripts_run_dir = repo_root() / "scripts" / "run"
    if str(scripts_run_dir) not in sys.path:
        sys.path.insert(0, str(scripts_run_dir))
    import run_eval as eval_common
    from run_eval_apertium import write_metrics

    eval_common.configure_offline_transformers(args.comet_encoder_model, bool(args.offline_eval_assets))
    eval_cfg = load_json(args.eval_config)
    manifest = load_json(args.manifest)
    items_path = repo_root() / manifest["items_jsonl"]
    if not items_path.is_file():
        print(f"未找到 items_jsonl: {items_path}", file=sys.stderr)
        return 1
    items = read_items_jsonl(items_path)
    corpus = str(args.corpus).strip().lower()
    if corpus:
        items = [it for it in items if str(it.get("eval_corpus", "")).strip().lower() == corpus]
    if not items:
        print(f"评估条目为空（corpus={args.corpus}）。", file=sys.stderr)
        return 1

    base_out = repo_root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = args.output_run_dir or (base_out / f"{args.model_tag}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = run_dir / "hypotheses.jsonl"

    existing_results: list[dict] = []
    existing_by_key: dict[tuple, dict] = {}
    pending_items = list(items)
    if args.resume and hyp_path.is_file():
        try:
            existing_results = eval_common.load_hypotheses_jsonl(hyp_path)
            existing_by_key = {result_key(r): r for r in existing_results}
            pending_items = [it for it in items if result_key(it) not in existing_by_key]
            print(
                f"resume: found {len(existing_results)} existing hypotheses, remaining {len(pending_items)} samples",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"resume: failed to read existing hypotheses, start fresh: {e}", file=sys.stderr)
            existing_results = []
            existing_by_key = {}
            pending_items = list(items)

    new_results: list[dict] = []
    async with LangGraphDictionaryAgentRuntime(
        model=args.model,
        model_family=args.model_family,
        base_url=args.base_url,
        api_key=args.api_key,
        lexicon_dir=args.lexicon_dir,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        debug=bool(args.debug),
    ) as runtime:
        for it in tqdm(pending_items, total=len(pending_items), desc="translate"):
            hyp = await runtime.translate(
                text=str(it["source_text"]),
                src_lang=str(it["src_lang"]),
                tgt_lang=str(it["tgt_lang"]),
            )
            new_results.append({**it, "hypothesis": hyp})
            if len(new_results) % 20 == 0:
                merged = list(existing_results) + list(new_results)
                merged.sort(key=result_key)
                eval_common.write_hypotheses_jsonl(hyp_path, merged)

    merged = list(existing_results) + list(new_results)
    merged.sort(key=result_key)
    eval_common.write_hypotheses_jsonl(hyp_path, merged)

    by_key = {result_key(r): r for r in merged}
    results = [by_key[result_key(it)] for it in items if result_key(it) in by_key]

    meta = {
        "provider": "dictionary_tool_agent",
        "model": args.model,
        "model_family": args.model_family,
        "base_url": args.base_url,
        "lexicon_dir": str(args.lexicon_dir),
        "num_samples": len(results),
        "manifest": str(args.manifest),
        "eval_config": str(args.eval_config),
        "corpus": args.corpus,
        "resume": bool(args.resume),
    }
    with open(run_dir / "generation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    flores200_spm_model = args.flores200_spm_model if args.flores200_spm_model.is_file() else None
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


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
