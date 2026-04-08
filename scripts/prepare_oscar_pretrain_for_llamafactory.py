#!/usr/bin/env python3
"""
从 Hugging Face「oscar-corpus/OSCAR-2301」流式读取单语言子集，导出为 LLaMAFactory **预训练（pt）** 本地 jsonl。
每行一个 JSON 对象：{"text": "<文档正文>"}。

重要（访问）：
- OSCAR-2301 在 Hub 上为 **带人工审核的门控**；须在数据集页同意条款并完成表单，且本机已登录：
    huggingface-cli login
  或设置环境变量 HF_TOKEN / HUGGING_FACE_HUB_TOKEN。
- 若 Hub 侧暂停授权，load_dataset 会失败，与脚本无关。

FLORES 风格语言标签 → OSCAR-2301 配置名（ISO 639-1，与 Hub 子集一致）：

  spa_Latn → es, ind_Latn → id, vie_Latn → vi, tha_Thai → th, tgl_Latn → tl

字段：OSCAR-23.01 行为主列为 **content**（若不存在则依次尝试 text / oscar）。

依赖：建议在 lowres 环境中安装 **zstandard**（部分 Parquet 分片解压需要）：
  pip install zstandard

LLaMAFactory 注册示例（dataset_info.json 片段，file_name 相对 dataset_dir）：

  "oscar_pt_spa_Latn": {
    "file_name": "oscar_pt_spa_Latn.jsonl",
    "formatting": "alpaca",
    "columns": { "prompt": "text", "response": null }
  }

用法：
  conda activate lowres
  python scripts/prepare_oscar_pretrain_for_llamafactory.py --limit 50000
  python scripts/prepare_oscar_pretrain_for_llamafactory.py --flores-lang spa_Latn vie_Latn --limit 10000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator

PREVIEW_N = 50

FLORES_TO_OSCAR_SUBSET: dict[str, str] = {
    "spa_Latn": "es",
    "ind_Latn": "id",
    "vie_Latn": "vi",
    "tha_Thai": "th",
    "tgl_Latn": "tl",
}


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_doc_text(row: dict[str, Any], content_keys: list[str]) -> str | None:
    for k in content_keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def iter_oscar_rows(
    repo_id: str,
    subset: str,
    *,
    split: str,
    token: str | bool | None,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, subset, split=split, streaming=True, token=token)
    for row in ds:
        if not isinstance(row, dict):
            row = dict(row)
        yield row


def main() -> int:
    ap = argparse.ArgumentParser(description="OSCAR-2301 → LLaMAFactory 预训练 jsonl（单语）")
    ap.add_argument(
        "--repo-id",
        default="oscar-corpus/OSCAR-2301",
        help="Hub 数据集 id",
    )
    ap.add_argument(
        "--split",
        default="train",
        help="split 名称（默认 train）",
    )
    ap.add_argument(
        "--flores-lang",
        action="append",
        dest="flores_langs",
        metavar="CODE",
        help=f"FLORES 语言码，可多次指定。默认: {', '.join(FLORES_TO_OSCAR_SUBSET)}",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=50_000,
        help="每种语言最多写入的文档条数（0 表示不限制，慎用）",
    )
    ap.add_argument(
        "--content-key",
        action="append",
        dest="content_keys",
        metavar="NAME",
        help="正文列名，按顺序尝试；默认 content text oscar",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="跳过短于此字符数的文档",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认 <repo>/training/data）",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="显式传递 Hub token；默认用环境变量 HF_TOKEN / HUGGING_FACE_HUB_TOKEN，否则 True（走 huggingface-cli 缓存）",
    )
    args = ap.parse_args()

    flores_list = args.flores_langs or list(FLORES_TO_OSCAR_SUBSET.keys())
    content_keys = args.content_keys or ["content", "text", "oscar"]

    out_dir = args.out_dir or (root() / "training" / "data")
    prev_dir = out_dir / "previews"
    ensure_dir(out_dir)
    ensure_dir(prev_dir)

    token: str | bool | None
    if args.hf_token:
        token = args.hf_token
    elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    else:
        token = True

    limit = args.limit
    min_chars = args.min_chars

    for fl in flores_list:
        if fl not in FLORES_TO_OSCAR_SUBSET:
            raise SystemExit(
                f"未知 FLORES 语言码 {fl!r}。已知: {', '.join(sorted(FLORES_TO_OSCAR_SUBSET))}"
            )
        subset = FLORES_TO_OSCAR_SUBSET[fl]
        out_path = out_dir / f"oscar_pt_{fl}.jsonl"
        prev_path = prev_dir / f"oscar_pt_{fl}.preview_{PREVIEW_N}.jsonl"

        n_written = 0
        preview_lines: list[str] = []
        print(f"[{fl} / {subset}] → {out_path} (limit={limit or '无限制'})")

        with open(out_path, "w", encoding="utf-8") as fo:
            try:
                rows = iter_oscar_rows(args.repo_id, subset, split=args.split, token=token)
            except Exception as e:
                raise SystemExit(
                    f"无法打开 {args.repo_id} 子集 {subset}: {e}\n"
                    "请确认已在 Hub 接受 OSCAR 门控并完成登录（见本脚本顶部说明）。"
                ) from e

            for row in rows:
                if limit and n_written >= limit:
                    break
                text = pick_doc_text(row, content_keys)
                if text is None or len(text) < min_chars:
                    continue
                line = json.dumps({"text": text}, ensure_ascii=False)
                fo.write(line + "\n")
                if n_written < PREVIEW_N:
                    preview_lines.append(line)
                n_written += 1
                if n_written % 10_000 == 0:
                    print(f"  ... {n_written} 条")

        prev_path.write_text(
            "\n".join(preview_lines) + ("\n" if preview_lines else ""),
            encoding="utf-8",
        )
        print(f"  完成: {n_written} 条；预览 {prev_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
