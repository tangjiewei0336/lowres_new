#!/usr/bin/env python3
"""
Generate translation candidates and select the final hypothesis by reranking.

This script covers inference-time combinations:
  candidate-mode = sample | rag | mixed (mixed = read flat candidates, see --from-flat-candidates-jsonl)
  reranker       = llm | comet-qe

Examples:
  # P(y|x), five candidates, LLM reranker
  conda run -n lowres python scripts/inference/generate_reranked_hypotheses.py \
    --candidate-mode sample --reranker llm \
    --base-url "$OPENAI_API_BASE" --api-key "$OPENAI_API_KEY" \
    --model qwen3-8b --model-family qwen --model-tag qwen3_8b_sample_llm

  # RAG candidates from augmented FineWeb pairs, COMET-QE reranker
  conda run -n lowres python scripts/inference/generate_reranked_hypotheses.py \
    --candidate-mode rag --reranker comet-qe \
    --aug-data-dir training/data/multilingual/fineweb2_synth \
    --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --base-url "$OPENAI_API_BASE" --api-key "$OPENAI_API_KEY" \
    --model qwen3-8b --model-family qwen --model-tag qwen3_8b_rag_cometqe

  # Rerank only from mix_hypothesis_candidates.py output (combined sample+RAG flats)
  conda run -n lowres python scripts/inference/generate_reranked_hypotheses.py \
    --from-flat-candidates-jsonl eval_multilingual/my_mix/run_001/candidates.jsonl \
    --reranker llm --model qwen3-8b --model-family qwen --model-tag qwen3_8b_mixed_llm
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent / "run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import run_eval as eval_common  # noqa: E402


@dataclass(frozen=True)
class Example:
    src: str
    tgt: str
    score: float = 0.0


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


def pair_from_item(item: dict[str, Any]) -> str:
    return str(item.get("eval_pair") or f"{item['src_lang']}->{item['tgt_lang']}")


def pair_file_name(src_lang: str, tgt_lang: str) -> str:
    return f"fineweb_synth_{src_lang}__{tgt_lang}.jsonl"


def resolve_text_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    src = row.get("input") or row.get("source_text") or row.get("src") or row.get("source")
    tgt = row.get("output") or row.get("target_text") or row.get("tgt") or row.get("target")
    if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
        return src.strip(), tgt.strip()
    return None


def load_pair_examples(path: Path, *, limit: int = 0) -> list[Example]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: list[Example] = []
    max_rows = limit if limit and limit > 0 else None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pair = resolve_text_pair(row)
            if pair is None:
                continue
            out.append(Example(src=pair[0], tgt=pair[1]))
            if max_rows and len(out) >= max_rows:
                break
    return out


class FaissRetriever:
    def __init__(
        self,
        examples: list[Example],
        embedding_model: str,
        device: str | None = None,
        *,
        index: Any | None = None,
    ):
        if not examples:
            raise ValueError("empty examples")
        try:
            import faiss  # type: ignore
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "RAG mode needs faiss and sentence-transformers. Install them in lowres, "
                "for example: pip install faiss-cpu sentence-transformers"
            ) from e
        self.faiss = faiss
        self.np = np
        self.examples = examples
        self.model = SentenceTransformer(embedding_model, device=device)
        if index is None:
            vectors = self.model.encode(
                [x.src for x in examples],
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            vectors = np.asarray(vectors, dtype="float32")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        else:
            self.index = index

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        embedding_model: str,
        device: str | None = None,
    ) -> "FaissRetriever":
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise RuntimeError("RAG mode needs faiss. Install it in lowres: pip install faiss-cpu") from e
        meta_path = index_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            built_with = meta.get("embedding_model")
            if built_with and built_with != embedding_model:
                raise RuntimeError(
                    f"Embedding model mismatch for {index_dir}: index uses {built_with}, "
                    f"but --embedding-model is {embedding_model}"
                )
        examples_path = index_dir / "examples.jsonl"
        index_path = index_dir / "index.faiss"
        if not examples_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"Missing index.faiss or examples.jsonl in {index_dir}")
        examples = []
        for row in read_jsonl(examples_path):
            pair = resolve_text_pair(row)
            if pair is not None:
                examples.append(Example(src=pair[0], tgt=pair[1]))
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(examples):
            raise RuntimeError(f"Index/example length mismatch in {index_dir}: {index.ntotal} vs {len(examples)}")
        return cls(examples, embedding_model, device=device, index=index)

    def search(self, text: str, k: int) -> list[Example]:
        vec = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        vec = self.np.asarray(vec, dtype="float32")
        scores, idxs = self.index.search(vec, k)
        out: list[Example] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist(), strict=False):
            if idx < 0:
                continue
            ex = self.examples[idx]
            out.append(Example(src=ex.src, tgt=ex.tgt, score=float(score)))
        return out


def openai_extra(model_family: str) -> dict[str, Any] | None:
    return eval_common.build_extra_body(model_family)


def chat_text(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    model_family: str,
    temperature: float = 0.0,
    n: int = 1,
) -> list[str]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if n > 1:
        kwargs["n"] = n
    extra = openai_extra(model_family)
    if extra:
        kwargs["extra_body"] = extra
    resp = client.chat.completions.create(**kwargs)
    texts = [eval_common.extract_message_text(choice.message).strip() for choice in resp.choices]
    return [x for x in texts if x]


def unique_keep_order(texts: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def sample_candidates(
    client: OpenAI,
    item: dict[str, Any],
    *,
    model: str,
    model_family: str,
    max_tokens: int,
    num_candidates: int,
    temperature: float,
) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You are a professional machine translation engine."},
        {"role": "user", "content": eval_common.mt_user_content(item["src_lang"], item["tgt_lang"], item["source_text"])},
    ]
    try:
        texts = chat_text(
            client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            model_family=model_family,
            temperature=temperature,
            n=num_candidates,
        )
    except Exception:
        texts = []
        for _ in range(num_candidates):
            texts.extend(
                chat_text(
                    client,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    model_family=model_family,
                    temperature=temperature,
                    n=1,
                )
            )
    texts = unique_keep_order(texts, num_candidates)
    return [
        {"candidate_id": i, "hypothesis": text, "candidate_source": "sample"}
        for i, text in enumerate(texts)
    ]


def rag_user_content(src_lang: str, tgt_lang: str, source_text: str, example: Example) -> str:
    return (
        f"{eval_common.english_translation_instruction(src_lang, tgt_lang)}\n\n"
        "Use the retrieved example only as style and terminology guidance.\n"
        "Do not copy the example unless it is truly the same sentence.\n\n"
        f"Retrieved example source:\n{example.src}\n\n"
        f"Retrieved example translation:\n{example.tgt}\n\n"
        f"Source to translate:\n{source_text}"
    )


def rag_candidates(
    client: OpenAI,
    item: dict[str, Any],
    examples: list[Example],
    *,
    model: str,
    model_family: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        messages = [
            {"role": "system", "content": "You are a professional machine translation engine."},
            {
                "role": "user",
                "content": rag_user_content(item["src_lang"], item["tgt_lang"], item["source_text"], ex),
            },
        ]
        text = chat_text(
            client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            model_family=model_family,
            temperature=0.0,
            n=1,
        )[0]
        out.append(
            {
                "candidate_id": i,
                "hypothesis": text,
                "candidate_source": "rag",
                "retrieval_score": ex.score,
                "retrieved_source": ex.src,
                "retrieved_target": ex.tgt,
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in out:
        hyp = re.sub(r"\s+", " ", str(cand["hypothesis"]).strip())
        if hyp and hyp not in seen:
            seen.add(hyp)
            cand["hypothesis"] = hyp
            cand["candidate_id"] = len(deduped)
            deduped.append(cand)
    return deduped


def llm_rerank(
    client: OpenAI,
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    model: str,
    model_family: str,
    max_tokens: int,
) -> tuple[int, str]:
    numbered = "\n".join(f"[{i}] {c['hypothesis']}" for i, c in enumerate(candidates))
    prompt = (
        "Choose the best translation candidate for the source sentence.\n"
        "Judge adequacy, fluency, terminology, and whether the output is only the translation.\n"
        "Return only the candidate number, with no explanation.\n\n"
        f"Source language: {item['src_lang']}\n"
        f"Target language: {item['tgt_lang']}\n"
        f"Source:\n{item['source_text']}\n\n"
        f"Candidates:\n{numbered}"
    )
    texts = chat_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict machine-translation reranker."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        model_family=model_family,
        temperature=0.0,
        n=1,
    )
    raw = texts[0] if texts else ""
    m = re.search(r"\d+", raw)
    idx = int(m.group(0)) if m else 0
    if idx < 0 or idx >= len(candidates):
        idx = 0
    return idx, raw


_FLAT_ROW_CAND_FIELDS = frozenset(
    {
        "hypothesis",
        "candidate_id",
        "candidate_source",
        "retrieval_score",
        "retrieved_source",
        "retrieved_target",
        "comet_qe",
    }
)


def split_candidate_flat_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """扁平行拆解为样本 base 与单项 candidate 字典。"""
    cand: dict[str, Any] = {}
    base = dict(row)
    for k in _FLAT_ROW_CAND_FIELDS:
        if k in base:
            cand[k] = base.pop(k)
    return base, cand


def _candidate_id_sort_key(row: dict[str, Any]) -> int:
    raw = row.get("candidate_id", 999)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 999


def candidate_rows_from_flat_jsonl(path: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 mix 脚本等产出的扁平 candidates.jsonl 恢复 generate 所使用的嵌套结构。"""
    flat_lines = read_jsonl(path)
    if not flat_lines:
        raise SystemExit(f"--from-flat-candidates-jsonl 为空: {path}")
    groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in flat_lines:
        groups[eval_common.result_key(row)].append(row)

    ordered_keys = [eval_common.result_key(it) for it in items]
    keys_done: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []

    def build_one_row(key: tuple[Any, ...]) -> dict[str, Any]:
        grp = groups[key]
        grp_sorted = sorted(grp, key=_candidate_id_sort_key)
        base0, _ = split_candidate_flat_row(grp_sorted[0])
        src0 = base0.get("source_text")
        cands_out: list[dict[str, Any]] = []
        for i, r in enumerate(grp_sorted):
            base_i, cand = split_candidate_flat_row(r)
            if base_i.get("source_text") != src0:
                raise RuntimeError(
                    f"同一 result_key 下 source_text 不一致: key={key} sample_id={base0.get('sample_id')}"
                )
            cand = dict(cand)
            cand["candidate_id"] = i
            if "hypothesis" not in cand:
                cand["hypothesis"] = ""
            cands_out.append(cand)
        return {**base0, "candidates": cands_out}

    for key in ordered_keys:
        if key not in groups:
            raise SystemExit(f"扁平候选文件缺少评测样本（key={key}），请核对 manifest / items-jsonl")
        out.append(build_one_row(key))
        keys_done.add(key)

    rest = sorted(k for k in groups if k not in keys_done)
    for key in rest:
        print(
            f"[warn] 扁平候选中存在 manifest/items 之外的 key={key}，已跳过（以免影响 hypotheses 顺序）",
            file=sys.stderr,
        )
    return out


def comet_qe_scores(
    rows: list[dict[str, Any]],
    *,
    model_arg: str,
    out_dir: Path,
    batch_size: int,
    encoder_model: Path | None,
) -> list[float]:
    ckpt, torch_mod, load_fn = eval_common.prepare_comet_checkpoint(model_arg, out_dir, encoder_path=encoder_model)
    if not ckpt or torch_mod is None or load_fn is None:
        raise RuntimeError(f"Could not load COMET-QE model: {model_arg}")
    gpus = 1 if torch_mod.cuda.is_available() else 0
    ckpt = eval_common.patch_comet_checkpoint_pretrained_model(ckpt, encoder_model)
    model = eval_common.load_comet_model(load_fn, ckpt)
    data = [{"src": r["source_text"], "mt": r["hypothesis"]} for r in rows]
    pred = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = pred.get("scores", []) if isinstance(pred, dict) else getattr(pred, "scores", [])
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError(f"COMET-QE output length mismatch: got {len(scores)} for {len(rows)}")
    return [float(x) for x in scores]


def main() -> int:
    parser = argparse.ArgumentParser(description="Candidate generation + reranking for MT inference.")
    parser.add_argument("--eval-config", type=Path, default=eval_common.root() / "evaluation_config.json")
    parser.add_argument("--manifest", type=Path, default=eval_common.root() / "datasets" / "eval_manifest.json")
    parser.add_argument("--items-jsonl", type=Path, default=None)
    parser.add_argument(
        "--from-flat-candidates-jsonl",
        type=Path,
        default=None,
        help="跳过生成阶段，直接读取扁平 candidates.jsonl（如 mix_hypothesis_candidates 产出），再接 reranker。此时 --candidate-mode 默认记作 mixed（仅写入 meta）。",
    )
    parser.add_argument("--output-run-dir", type=Path, default=None)
    parser.add_argument("--model-tag", default=os.environ.get("EVAL_MODEL_TAG", "rerank"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-family", default=os.environ.get("EVAL_MODEL_FAMILY", "qwen"))
    parser.add_argument(
        "--candidate-mode",
        choices=["sample", "rag", "mixed"],
        default=None,
        help="mixed 通常在提供 --from-flat-candidates-jsonl 时使用（或未指定时将由该选项自动设为 mixed）。",
    )
    parser.add_argument("--reranker", choices=["llm", "comet-qe"], required=True)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--rerank-max-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--aug-data-dir", type=Path, default=eval_common.root() / "training" / "data" / "multilingual" / "fineweb2_synth")
    parser.add_argument("--aug-file-template", default="fineweb_synth_{src}__{tgt}.jsonl")
    parser.add_argument("--rag-index-dir", type=Path, default=eval_common.root() / "indexes" / "faiss_aug_fineweb")
    parser.add_argument("--build-rag-index-on-the-fly", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rag-index-limit", type=int, default=0, help="Optional max augmented examples per pair.")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--comet-qe-model", default="models/Unbabel_wmt22-cometkiwi-da")
    parser.add_argument("--comet-encoder-model", type=Path, default=eval_common.root() / "models" / "xlm-roberta-large")
    parser.add_argument("--comet-batch-size", type=int, default=8)
    parser.add_argument("--offline-eval-assets", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.from_flat_candidates_jsonl:
        args.candidate_mode = args.candidate_mode or "mixed"
        if args.candidate_mode != "mixed":
            print(
                f"[warn] 已加载 --from-flat-candidates-jsonl，将 candidate_mode={args.candidate_mode} 重写为 mixed（仅写入 meta）",
                file=sys.stderr,
            )
            args.candidate_mode = "mixed"
    elif args.candidate_mode == "mixed":
        raise SystemExit("--candidate-mode mixed 仅支持与 --from-flat-candidates-jsonl 一起使用")
    elif args.candidate_mode is None:
        raise SystemExit("请指定 --candidate-mode（sample|rag），或传入 --from-flat-candidates-jsonl")

    eval_common.quiet_http_logging()
    eval_cfg = eval_common.load_json(args.eval_config)
    manifest = eval_common.load_json(args.manifest)
    items_path = args.items_jsonl or (eval_common.root() / manifest["items_jsonl"])
    items = eval_common.read_items_jsonl(items_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No eval items found: {items_path}")

    max_tokens = int(args.max_tokens or eval_cfg.get("max_tokens", 512))
    max_workers = int(args.max_workers or eval_cfg.get("max_workers", 8))
    base_out = eval_common.root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = args.output_run_dir or (
        base_out / f"{args.model_tag}_{args.candidate_mode}_{args.reranker}_{int(time.time())}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    client_kwargs: dict[str, Any] = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    retrievers: dict[str, FaissRetriever] = {}
    if args.candidate_mode == "rag" and not args.from_flat_candidates_jsonl:
        by_pair: dict[str, tuple[str, str]] = {}
        for it in items:
            by_pair[pair_from_item(it)] = (it["src_lang"], it["tgt_lang"])
        for pair, (src, tgt) in sorted(by_pair.items()):
            index_dir = args.rag_index_dir / f"{src}__{tgt}"
            if (index_dir / "index.faiss").is_file() and (index_dir / "examples.jsonl").is_file():
                print(f"Loading FAISS retriever for {pair}: {index_dir}", file=sys.stderr)
                retrievers[pair] = FaissRetriever.from_index_dir(
                    index_dir,
                    args.embedding_model,
                    device=args.embedding_device,
                )
                continue
            if not args.build_rag_index_on_the_fly:
                raise SystemExit(
                    f"Missing RAG index for {pair}: {index_dir}. "
                    "Run scripts/inference/build_faiss_rag_index.py first."
                )
            path = args.aug_data_dir / args.aug_file_template.format(src=src, tgt=tgt)
            examples = load_pair_examples(path, limit=args.rag_index_limit)
            print(f"Building FAISS retriever on the fly for {pair}: {path} rows={len(examples)}", file=sys.stderr)
            retrievers[pair] = FaissRetriever(examples, args.embedding_model, device=args.embedding_device)

    def make_candidates(it: dict[str, Any]) -> dict[str, Any]:
        if args.candidate_mode == "sample":
            cands = sample_candidates(
                client,
                it,
                model=args.model,
                model_family=args.model_family,
                max_tokens=max_tokens,
                num_candidates=args.num_candidates,
                temperature=args.sample_temperature,
            )
        else:
            pair = pair_from_item(it)
            examples = retrievers[pair].search(it["source_text"], args.num_candidates)
            cands = rag_candidates(
                client,
                it,
                examples,
                model=args.model,
                model_family=args.model_family,
                max_tokens=max_tokens,
            )
        if not cands:
            raise RuntimeError(f"No candidates for {pair_from_item(it)} sample_id={it.get('sample_id')}")
        return {**it, "candidates": cands}

    if args.from_flat_candidates_jsonl:
        if not args.from_flat_candidates_jsonl.is_file():
            raise FileNotFoundError(args.from_flat_candidates_jsonl)
        # 仅用 manifest 的顺序重建；与 mix 脚本 --manifest 一致时可保证 deterministic
        candidate_rows = candidate_rows_from_flat_jsonl(args.from_flat_candidates_jsonl, items)
    else:
        candidate_rows_unordered: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(make_candidates, it) for it in items]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"generate-{args.candidate_mode}"):
                candidate_rows_unordered.append(fut.result())
        by_key = {eval_common.result_key(r): r for r in candidate_rows_unordered}
        candidate_rows = [by_key[eval_common.result_key(it)] for it in items]

    flat_candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        base = {k: v for k, v in row.items() if k != "candidates"}
        for cand in row["candidates"]:
            flat_candidates.append({**base, **cand})
    write_jsonl(run_dir / "candidates.jsonl", flat_candidates)

    selected: dict[tuple[Any, ...], tuple[int, str, float | None]] = {}
    if args.reranker == "llm":
        def rerank_one(row: dict[str, Any]) -> tuple[tuple[Any, ...], int, str]:
            idx, raw = llm_rerank(
                client,
                row,
                row["candidates"],
                model=args.model,
                model_family=args.model_family,
                max_tokens=args.rerank_max_tokens,
            )
            return eval_common.result_key(row), idx, raw

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(rerank_one, row) for row in candidate_rows]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="rerank-llm"):
                key, idx, raw = fut.result()
                selected[key] = (idx, raw, None)
    else:
        encoder = args.comet_encoder_model if args.comet_encoder_model.is_dir() else None
        eval_common.configure_offline_transformers(encoder, bool(args.offline_eval_assets))
        scores = comet_qe_scores(
            flat_candidates,
            model_arg=args.comet_qe_model,
            out_dir=run_dir / "comet_qe_model",
            batch_size=args.comet_batch_size,
            encoder_model=encoder,
        )
        score_i = 0
        for row in candidate_rows:
            best_idx = 0
            best_score = float("-inf")
            for local_idx, cand in enumerate(row["candidates"]):
                score = scores[score_i]
                cand["comet_qe"] = score
                flat_candidates[score_i]["comet_qe"] = score
                if score > best_score:
                    best_score = score
                    best_idx = local_idx
                score_i += 1
            selected[eval_common.result_key(row)] = (best_idx, "comet-qe", best_score)
        write_jsonl(run_dir / "candidates.jsonl", flat_candidates)

    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = eval_common.result_key(row)
        idx, raw, score = selected[key]
        cand = row["candidates"][idx]
        final = {k: v for k, v in row.items() if k != "candidates"}
        final["hypothesis"] = cand["hypothesis"]
        final["selected_candidate_id"] = cand["candidate_id"]
        final["candidate_mode"] = args.candidate_mode
        final["reranker"] = args.reranker
        if score is not None:
            final["selected_comet_qe"] = score
        final_rows.append(final)
        summary_rows.append(
            {
                "corpus": final.get("eval_corpus") or final.get("dataset") or "",
                "pair": pair_from_item(final),
                "sample_id": final.get("sample_id", ""),
                "selected_candidate_id": cand["candidate_id"],
                "selected_score": "" if score is None else score,
                "reranker_raw": raw,
                "hypothesis": cand["hypothesis"],
            }
        )

    eval_common.write_hypotheses_jsonl(run_dir / "hypotheses.jsonl", final_rows)
    with (run_dir / "rerank_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus",
                "pair",
                "sample_id",
                "selected_candidate_id",
                "selected_score",
                "reranker_raw",
                "hypothesis",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "model": args.model,
        "model_family": args.model_family,
        "base_url": args.base_url,
        "candidate_mode": args.candidate_mode,
        "reranker": args.reranker,
        "num_candidates": args.num_candidates,
        "items_jsonl": str(items_path),
        "num_samples": len(final_rows),
        "aug_data_dir": str(args.aug_data_dir),
        "rag_index_dir": str(args.rag_index_dir) if args.candidate_mode == "rag" else None,
        "embedding_model": args.embedding_model if args.candidate_mode == "rag" and not args.from_flat_candidates_jsonl else None,
        "from_flat_candidates_jsonl": str(args.from_flat_candidates_jsonl.resolve()) if args.from_flat_candidates_jsonl else None,
        "comet_qe_model": args.comet_qe_model if args.reranker == "comet-qe" else None,
    }
    (run_dir / "generation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成。run_dir: {run_dir}")
    print(f"hypotheses: {run_dir / 'hypotheses.jsonl'}")
    print(f"candidates: {run_dir / 'candidates.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
