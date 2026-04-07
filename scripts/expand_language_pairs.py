#!/usr/bin/env python3
"""
根据 evaluation_config.json 展开语言对，合并 datasets/processed 下已生成的 jsonl，
输出 eval_manifest.json（含 items_jsonl 路径）与 eval_items_all.jsonl。
NTREX 为英语中心：合并配置中出现的每种非英语语言对应的 eng_Latn -> tgt（prepare 阶段已生成）。
"""
from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path
from typing import Any


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def expand_pairs(groups: list[list[str]], bidirectional: bool) -> list[tuple[str, str]]:
    """
    配对规则：从任意两组中各取一个语言，做笛卡尔积（不是组内两两组合）。
    """
    pairs: list[tuple[str, str]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            if not g1 or not g2:
                continue
            for a, b in product(g1, g2):
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

def _stable_int_seed(seed: int | None, key: str) -> int | None:
    """将全局 seed + key 组合成稳定的子 seed（不依赖 Python hash 随机化）。"""
    if seed is None:
        return None
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (int(seed) ^ h) & 0x7FFFFFFF


def apply_limit_per_pair(
    items: list[dict[str, Any]],
    per_pair_limit: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    """对每个 (eval_corpus, eval_pair) 单独限量抽样，保持可复现。"""
    if not per_pair_limit:
        return items
    buckets: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        corp = str(it.get("eval_corpus", "?"))
        pair = str(it.get("eval_pair", it.get("src_lang", "?") + "->" + it.get("tgt_lang", "?")))
        k = f"{corp}|{pair}"
        buckets.setdefault(k, []).append(it)
    out: list[dict[str, Any]] = []
    for k, bucket in buckets.items():
        out.extend(apply_limit(bucket, per_pair_limit, _stable_int_seed(seed, k)))
    return out


def main() -> int:
    cfg = load_json(root() / "evaluation_config.json")
    split = cfg["split"]
    bidir = bool(cfg.get("bidirectional", True))
    groups = cfg["language_pair_groups"]
    limit = cfg.get("limit")
    per_pair_limit = cfg.get("per_pair_limit")
    seed = cfg.get("seed")

    pairs = expand_pairs(groups, bidir)
    all_langs = sorted({x for group in groups for x in group})
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

    # NTREX：仅 english-centered 且在“跨组配对”中出现的方向 eng_Latn -> tgt
    eng = "eng_Latn"
    ntrex_targets: list[str] = []
    for src, tgt in pairs:
        if src == eng and tgt != eng and tgt not in ntrex_targets:
            ntrex_targets.append(tgt)

    for tgt in ntrex_targets:
        np = processed / ntrex_filename(split, eng, tgt)
        rows = read_jsonl(np)
        if rows:
            source_files.append(str(np.relative_to(root())))
        for r in rows:
            r = dict(r)
            r["eval_pair"] = f"{eng}->{tgt}"
            r["eval_corpus"] = "ntrex"
            all_items.append(r)

    # 先做单语言对限量，再做全局限量（方便快速整体采样）
    all_items = apply_limit_per_pair(all_items, per_pair_limit, seed)
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
        "per_pair_limit": per_pair_limit,
        "seed": seed,
    }
    out_manifest = root() / "datasets" / "eval_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"写出 {manifest['items_jsonl']} 共 {len(all_items)} 条；元数据 {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
