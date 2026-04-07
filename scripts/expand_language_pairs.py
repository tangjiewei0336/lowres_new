#!/usr/bin/env python3
"""
根据 evaluation_config.json 展开语言对，合并 datasets/processed 下已生成的 jsonl，
输出 eval_manifest.json（含 items_jsonl 路径）与 eval_items_all.jsonl。
NTREX 仅包含源为英语的方向（prepare 阶段已只生成 eng_Latn -> *）。
"""
from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def expand_pairs(groups: list[list[str]], bidirectional: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for group in groups:
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            pairs.append((a, b))
            if bidirectional:
                pairs.append((b, a))
    return pairs


def flores_filename(split: str, src: str, tgt: str) -> str:
    return f"flores_{split}_{src}__{tgt}.jsonl"


def ntrex_filename(split: str, src: str, tgt: str) -> str:
    return f"ntrex_{split}_{src}__{tgt}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def apply_limit(items: list[dict[str, Any]], limit: int | None, seed: int | None) -> list[dict[str, Any]]:
    if not limit or limit >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    keep = set(idx[:limit])
    return [items[i] for i in range(len(items)) if i in keep]


def main() -> int:
    cfg = load_json(root() / "evaluation_config.json")
    split = cfg["split"]
    bidir = bool(cfg.get("bidirectional", True))
    groups = cfg["language_pair_groups"]
    limit = cfg.get("limit")
    seed = cfg.get("seed")

    pairs = expand_pairs(groups, bidir)
    processed = root() / "datasets" / "processed"
    merged_path = processed / "eval_items_all.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    source_files: list[str] = []
    all_items: list[dict[str, Any]] = []

    for src, tgt in pairs:
        fp = processed / flores_filename(split, src, tgt)
        rows = read_jsonl(fp)
        if rows:
            source_files.append(str(fp.relative_to(root())))
        for r in rows:
            r = dict(r)
            r["eval_pair"] = f"{src}->{tgt}"
            r["eval_corpus"] = "flores"
            all_items.append(r)

    for src, tgt in pairs:
        if src != "eng_Latn":
            continue
        np = processed / ntrex_filename(split, src, tgt)
        rows = read_jsonl(np)
        if rows:
            source_files.append(str(np.relative_to(root())))
        for r in rows:
            r = dict(r)
            r["eval_pair"] = f"{src}->{tgt}"
            r["eval_corpus"] = "ntrex"
            all_items.append(r)

    all_items = apply_limit(all_items, limit, seed)

    with open(merged_path, "w", encoding="utf-8") as f:
        for r in all_items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "split": split,
        "bidirectional": bidir,
        "language_pair_groups": groups,
        "items_jsonl": str(merged_path.relative_to(root())),
        "item_count": len(all_items),
        "source_files": source_files,
        "limit_applied": limit,
        "seed": seed,
    }
    out_manifest = root() / "datasets" / "eval_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"写出 {manifest['items_jsonl']} 共 {len(all_items)} 条；元数据 {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
