#!/usr/bin/env python3
"""
将多个 Alpaca jsonl 按权重无放回抽样后合并、打乱，写出单一 jsonl。

用途：
  - 训练框架不支持多数据集 interleave 时，预先按固定比例合成一个文件；
  - 或缩小总规模同时保持句级 / 篇章的大致配比。

推荐优先使用 LLaMAFactory 内置混合（无需合并文件）：
  dataset: nllb_mt,nllb_draft_refine
  mix_strategy: interleave_under   # 或 interleave_over
  interleave_probs: 0.7,0.3
见 training/llamafactory_nllb_mt_refine_mix_lora.yaml

示例：
  python scripts/merge_alpaca_jsonl_weighted.py \\
    --inputs training/data/multilingual/nllb/nllb_mt_all.jsonl \\
              training/data/draft_refine/nllb/nllb_draft_refine_all.jsonl \\
    --weights 0.7 0.3 --total 100000 \\
    --out training/data/multilingual/nllb_mt_refine_mix_100k.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PREVIEW_N = 50


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_lines(path: Path) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
    return out


def allocate_counts(lengths: list[int], weights: list[float], total: int) -> list[int]:
    """在不超过各文件行数的前提下，使总条数尽量接近 total，比例接近 weights。"""
    if len(lengths) != len(weights):
        raise ValueError("lengths 与 weights 长度须一致")
    if total <= 0:
        raise ValueError("total 须为正整数")
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("weights 之和须为正")
    wn = [w / wsum for w in weights]
    n = [min(lengths[i], max(0, int(total * wn[i]))) for i in range(len(lengths))]

    while sum(n) > total:
        j = max(range(len(n)), key=lambda i: n[i])
        if n[j] <= 0:
            break
        n[j] -= 1

    while sum(n) < total:
        progressed = False
        for i in range(len(n)):
            if sum(n) >= total:
                break
            if n[i] < lengths[i]:
                n[i] += 1
                progressed = True
        if not progressed:
            break

    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="按权重抽样合并 Alpaca jsonl")
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="输入 jsonl 路径（顺序与 --weights 一致）",
    )
    ap.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="非负权重，会与 dataset 顺序一一对应并归一化",
    )
    ap.add_argument("--total", type=int, required=True, help="目标合并总条数（受各文件行数上限约束）")
    ap.add_argument("--out", type=Path, required=True, help="输出 jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-bad-lines",
        action="store_true",
        help="跳过无法解析为 JSON 的行",
    )
    args = ap.parse_args()

    if len(args.inputs) != len(args.weights):
        raise SystemExit("--inputs 与 --weights 个数须相同")
    if any(w < 0 for w in args.weights):
        raise SystemExit("--weights 须非负且不全为 0")

    pools: list[list[str]] = []
    lengths: list[int] = []
    for p in args.inputs:
        lines = read_lines(p)
        if args.skip_bad_lines:
            kept: list[str] = []
            for line in lines:
                try:
                    json.loads(line)
                    kept.append(line)
                except json.JSONDecodeError:
                    continue
            lines = kept
        pools.append(lines)
        lengths.append(len(lines))

    if any(x == 0 for x in lengths):
        raise SystemExit("某一输入文件为空（或无有效行）")

    counts = allocate_counts(lengths, args.weights, args.total)
    rng = random.Random(args.seed)
    merged: list[str] = []
    for lines, k in zip(pools, counts):
        if k <= 0:
            continue
        merged.extend(rng.sample(lines, k))

    rng.shuffle(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for line in merged:
            fo.write(line + "\n")

    prev_dir = args.out.parent / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev_path = prev_dir / f"{args.out.stem}.preview_{PREVIEW_N}.jsonl"
    prev_path.write_text(
        "\n".join(merged[:PREVIEW_N]) + ("\n" if merged[:PREVIEW_N] else ""),
        encoding="utf-8",
    )

    print(f"写出 {args.out} 共 {len(merged)} 条（计划各档 {counts}，输入上限 {lengths}）")
    print(f"预览 {prev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
