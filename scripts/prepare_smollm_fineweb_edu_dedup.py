#!/usr/bin/env python3
"""
从 Hugging Face 拉取 HuggingFaceTB/smollm-corpus 的 fineweb-edu-dedup 子集：
- 将 Hub 缓存指向 datasets/cache/huggingface（与 prepare_datasets 中 MODELSCOPE_CACHE 类似）
- 按 prepare_datasets.py 约定写出 processed/ 与 previews/（前 PREVIEW_N 条样本 JSONL）
- 扫描 parquet 分片，统计 metadata.language 条数并写入 JSON

可用 --max-shards N 只下载/统计前 N 个分片（按文件名排序），节省磁盘与流量；统计 JSON 中会标注 partial 与分片列表。

完整语料以 Hub 上的 parquet 分片形式存在；全量 1.9 亿行不适合再导出为单一 JSONL。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

PREVIEW_N = 50

DATASET_ID = "HuggingFaceTB/smollm-corpus"
SUBSET = "fineweb-edu-dedup"
SPLIT = "train"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_preview(src_path: Path, preview_path: Path, n: int = PREVIEW_N) -> None:
    lines: list[str] = []
    with open(src_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.rstrip("\n"))
    ensure_dir(preview_path.parent)
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def _json_safe_metadata(meta: Any) -> Any:
    if isinstance(meta, dict):
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if hasattr(v, "isoformat"):
                out[k] = v.isoformat()
            else:
                out[k] = v
        return out
    return meta


def _norm_row(
    text: str,
    sid: str,
    metadata: dict[str, Any] | Any,
) -> dict[str, Any]:
    md = metadata if isinstance(metadata, dict) else {}
    return {
        "dataset": "smollm_corpus",
        "subset": SUBSET,
        "split": SPLIT,
        "sample_id": sid,
        "text": text,
        "metadata": _json_safe_metadata(md),
    }


def _list_parquet_files() -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    paths = sorted(
        f for f in files if f.startswith(f"{SUBSET}/") and f.endswith(".parquet") and "train-" in f
    )
    if not paths:
        raise RuntimeError(f"未找到 {SUBSET} 的 parquet 分片")
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="准备 smollm-corpus / fineweb-edu-dedup 样本与语言统计")
    ap.add_argument(
        "--max-shards",
        type=int,
        default=None,
        metavar="N",
        help="仅处理前 N 个 parquet 分片（按路径排序）；省略则处理全部分片（体积与磁盘占用很大）",
    )
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(root() / "datasets" / "cache" / "huggingface"))

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    processed = root() / "datasets" / "processed"
    previews = root() / "datasets" / "previews"
    ensure_dir(processed)
    ensure_dir(previews)

    parquet_paths = _list_parquet_files()
    total_shards = len(parquet_paths)
    if args.max_shards is not None:
        if args.max_shards < 1:
            raise SystemExit("--max-shards 须 >= 1")
        parquet_paths = parquet_paths[: args.max_shards]
    partial = args.max_shards is not None and len(parquet_paths) < total_shards
    print(
        f"将处理 {len(parquet_paths)}/{total_shards} 个 parquet 分片"
        + (f"（--max-shards {args.max_shards}）" if args.max_shards is not None else "（全量）")
        + f"，缓存目录 HF_HOME={os.environ['HF_HOME']}"
    )

    counts: Counter[str] = Counter()
    preview_rows: list[dict[str, Any]] = []

    for pi, rel in enumerate(tqdm(parquet_paths, desc="分片")):
        local = hf_hub_download(DATASET_ID, rel, repo_type="dataset")
        pf = pq.ParquetFile(local)

        for batch in pf.iter_batches(columns=["metadata"], batch_size=65_536):
            meta_col = batch.column("metadata")
            langs = pc.struct_field(meta_col, ["language"])
            for lang in langs.to_pylist():
                key = lang if isinstance(lang, str) and lang else "(missing)"
                counts[key] += 1

        if pi == 0:
            t0 = pq.read_table(local, columns=["text", "id", "metadata"]).slice(0, PREVIEW_N)
            for i in range(t0.num_rows):
                preview_rows.append(
                    _norm_row(
                        str(t0["text"][i].as_py()),
                        str(t0["id"][i].as_py()),
                        t0["metadata"][i].as_py(),
                    )
                )

    base = f"smollm_corpus_{SUBSET}_{SPLIT}"
    sample_path = processed / f"{base}.sample_{PREVIEW_N}.jsonl"
    write_jsonl(sample_path, preview_rows)
    write_preview(sample_path, previews / f"{base}.sample_{PREVIEW_N}.jsonl.preview_{PREVIEW_N}.jsonl")

    total = sum(counts.values())
    stats = {
        "dataset_id": DATASET_ID,
        "subset": SUBSET,
        "split": SPLIT,
        "partial": partial,
        "shards_total_on_hub": total_shards,
        "shards_processed": len(parquet_paths),
        "shard_paths": parquet_paths,
        "total_rows": total,
        "unique_languages": len(counts),
        "by_language": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
    }
    if partial:
        stats["note"] = (
            "仅在部分分片上统计 language；全量分布需省略 --max-shards 或增大 N（磁盘与流量需求极大）。"
        )
    suffix = f"_partial_{len(parquet_paths)}shards" if partial else ""
    stats_path = processed / f"{SUBSET}_{SPLIT}_language_counts{suffix}.json"
    ensure_dir(stats_path.parent)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"语言统计已写入 {stats_path}（共 {total} 行，{len(counts)} 种语言码）")
    print(f"样本 JSONL: {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
