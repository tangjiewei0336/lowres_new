#!/usr/bin/env python3
"""
Build reusable FAISS indexes for RAG-based translation candidate generation.

Input rows are expected to look like augmented FineWeb / LLaMAFactory MT rows:
  {"input": "...", "output": "..."}
The loader also accepts source_text/target_text, src/tgt, and source/target.

Example:
  conda run -n lowres python scripts/inference/build_faiss_rag_index.py \
    --aug-data-dir training/data/multilingual/fineweb2_synth \
    --out-dir indexes/faiss_aug_fineweb \
    --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


@dataclass(frozen=True)
class Example:
    src: str
    tgt: str


PAIR_RE = re.compile(r"(.+?)__([^./]+)$")


def resolve_text_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    src = row.get("input") or row.get("source_text") or row.get("src") or row.get("source")
    tgt = row.get("output") or row.get("target_text") or row.get("tgt") or row.get("target")
    if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
        return src.strip(), tgt.strip()
    return None


def infer_pair_from_path(path: Path, prefix: str) -> tuple[str, str]:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    match = PAIR_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot infer src/tgt from filename: {path.name}")
    return match.group(1), match.group(2)


def load_examples(path: Path, *, limit: int = 0, min_chars: int = 1) -> list[Example]:
    examples: list[Example] = []
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
            src, tgt = pair
            if len(src) < min_chars or len(tgt) < min_chars:
                continue
            examples.append(Example(src=src, tgt=tgt))
            if max_rows and len(examples) >= max_rows:
                break
    return examples


def write_examples(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def build_one(
    *,
    input_path: Path,
    out_dir: Path,
    src_lang: str,
    tgt_lang: str,
    embedding_model: str,
    embedding_device: str | None,
    batch_size: int,
    limit: int,
    min_chars: int,
    overwrite: bool,
) -> None:
    pair_dir = out_dir / f"{src_lang}__{tgt_lang}"
    index_path = pair_dir / "index.faiss"
    examples_path = pair_dir / "examples.jsonl"
    meta_path = pair_dir / "meta.json"
    if index_path.exists() and examples_path.exists() and not overwrite:
        print(f"skip existing {src_lang}->{tgt_lang}: {pair_dir}")
        return

    try:
        import faiss  # type: ignore
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "FAISS index build needs faiss and sentence-transformers. "
            "Install in lowres: pip install faiss-cpu sentence-transformers"
        ) from e

    examples = load_examples(input_path, limit=limit, min_chars=min_chars)
    if not examples:
        raise RuntimeError(f"No usable examples in {input_path}")

    pair_dir.mkdir(parents=True, exist_ok=True)
    print(f"encode {src_lang}->{tgt_lang}: rows={len(examples)} model={embedding_model}")
    model = SentenceTransformer(embedding_model, device=embedding_device)
    vectors = model.encode(
        [x.src for x in examples],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    write_examples(examples_path, examples)
    meta = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "input_path": str(input_path),
        "num_examples": len(examples),
        "embedding_model": embedding_model,
        "embedding_dim": int(vectors.shape[1]),
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {pair_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FAISS RAG indexes from augmented MT JSONL files.")
    parser.add_argument("--aug-data-dir", type=Path, default=Path("training/data/multilingual/fineweb2_synth"))
    parser.add_argument("--glob", default="fineweb_synth_*__*.jsonl")
    parser.add_argument("--filename-prefix", default="fineweb_synth_")
    parser.add_argument("--input", type=Path, action="append", dest="inputs", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("indexes/faiss_aug_fineweb"))
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Optional max examples per pair.")
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = args.inputs or sorted(
        p for p in args.aug_data_dir.glob(args.glob)
        if p.is_file() and "previews" not in p.parts
    )
    if not inputs:
        raise SystemExit(f"No input files found under {args.aug_data_dir} with glob {args.glob}")

    for path in tqdm(inputs, desc="pairs"):
        src_lang, tgt_lang = infer_pair_from_path(path, args.filename_prefix)
        build_one(
            input_path=path,
            out_dir=args.out_dir,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            batch_size=args.batch_size,
            limit=args.limit,
            min_chars=args.min_chars,
            overwrite=bool(args.overwrite),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
