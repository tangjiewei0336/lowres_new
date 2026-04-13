#!/usr/bin/env python3
"""
将 training/data 下多个 nllb_mt_<src>__<tgt>.jsonl 合并为一个文件，供 LLaMAFactory 使用。

默认输入：training/data/multilingual/nllb/nllb_mt_*.jsonl（忽略 *_all.jsonl）
输出：
  training/data/multilingual/nllb/nllb_mt_all.jsonl
  training/data/multilingual/nllb/previews/nllb_mt_all.preview_50.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

PREVIEW_N = 50


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="合并 NLLB Alpaca jsonl 为单文件")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=root() / "training" / "data" / "multilingual" / "nllb",
        help="包含 nllb_mt_*.jsonl 的目录",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出文件路径（默认 <data-dir>/nllb_mt_all.jsonl）",
    )
    args = ap.parse_args()

    data_dir = args.data_dir
    out_path = args.out or (data_dir / "nllb_mt_all.jsonl")
    prev_dir = data_dir / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev_path = prev_dir / f"{out_path.stem}.preview_{PREVIEW_N}.jsonl"

    files = sorted(p for p in data_dir.glob("nllb_mt_*.jsonl") if p.name != out_path.name)
    if not files:
        raise SystemExit(f"未找到输入文件: {data_dir}/nllb_mt_*.jsonl")

    written = 0
    preview_lines: list[str] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for p in files:
            with open(p, encoding="utf-8") as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    fo.write(line + "\n")
                    if written < PREVIEW_N:
                        preview_lines.append(line)
                    written += 1

    prev_path.write_text("\n".join(preview_lines) + ("\n" if preview_lines else ""), encoding="utf-8")
    print(f"合并完成: {out_path} 共 {written} 条")
    print(f"预览: {prev_path}")
    print("包含文件：")
    for p in files:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
