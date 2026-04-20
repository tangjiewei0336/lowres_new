#!/usr/bin/env python3
"""
Build one multilingual SFT dataset from all MoE directions.

This is the non-MoE baseline data path: instead of one LoRA adapter per
translation direction, all selected directions are merged into a single Alpaca
JSONL and trained as one LoRA SFT adapter.

The script reuses training/moe_data_mix_config.json so the single-SFT baseline
and the pair-level MoE experiment can use the same per-direction source limits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


PREVIEW_N = 50


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_jsonl_lines(path: Path, *, skip_bad_lines: bool) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skip_bad_lines:
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
            lines.append(line)
    return lines


def sample_lines(lines: list[str], limit: int | None, rng: random.Random) -> list[str]:
    if limit is None or limit < 0 or limit >= len(lines):
        out = list(lines)
        rng.shuffle(out)
        return out
    if limit <= 0:
        return []
    return rng.sample(lines, limit)


def source_cfg_for_pair(default_sources: dict[str, Any], pair_item: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(default_sources, ensure_ascii=False))
    overrides = pair_item.get("sources") or {}
    if not isinstance(overrides, dict):
        raise SystemExit(f"pair sources must be an object: {pair_item!r}")
    for name, override in overrides.items():
        if not isinstance(override, dict):
            raise SystemExit(f"source override must be an object for {name}: {override!r}")
        base = out.setdefault(name, {})
        if not isinstance(base, dict):
            raise SystemExit(f"default source config must be an object for {name}: {base!r}")
        base.update(override)
    return out


def source_path(data_root: Path, template: str, src: str, tgt: str) -> Path:
    return data_root / template.format(src=src, tgt=tgt)


def dataset_info_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }


def update_dataset_info(path: Path, dataset_name: str, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{path} must contain a JSON object")
    else:
        data = {}
    data[dataset_name] = entry
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Build a single multilingual SFT dataset from all MoE directions.")
    ap.add_argument("--config", type=Path, default=root / "training/moe_data_mix_config.json")
    ap.add_argument("--data-root", type=Path, default=root / "training/data")
    ap.add_argument("--out-dir", type=Path, default=root / "training/data/multilingual/single_sft")
    ap.add_argument("--out-subdir", default="multilingual/single_sft")
    ap.add_argument("--out-name", default="qwen3_8b_all_directions_sft")
    ap.add_argument("--dataset-name", default="qwen3_8b_all_directions_sft")
    ap.add_argument(
        "--dataset-info-snippet",
        type=Path,
        default=root / "training/qwen3_8b_all_directions_sft_dataset_info.snippet.json",
    )
    ap.add_argument("--llamafactory-dataset-info", type=Path, default=root / "training/data/dataset_info.json")
    ap.add_argument("--no-update-llamafactory-dataset-info", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Fail if an enabled source file is missing.")
    ap.add_argument("--skip-bad-lines", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conf = json.loads(args.config.read_text(encoding="utf-8"))
    default_sources = conf.get("default_sources")
    pairs = conf.get("pairs")
    if not isinstance(default_sources, dict):
        raise SystemExit(f"{args.config} has no default_sources object")
    if not isinstance(pairs, list) or not pairs:
        raise SystemExit(f"{args.config} has no pairs list")

    default_seed = int(conf.get("default_seed", 42))
    rng = random.Random(default_seed)
    merged: list[str] = []
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "dataset_name": args.dataset_name,
        "file_name": f"{args.out_subdir.rstrip('/')}/{args.out_name}.jsonl",
        "pairs": [],
    }

    for idx, pair in enumerate(pairs):
        src = str(pair.get("src_lang", "")).strip()
        tgt = str(pair.get("tgt_lang", "")).strip()
        if not src or not tgt:
            raise SystemExit(f"Bad pair item: {pair!r}")
        pair_rng = random.Random(int(pair.get("seed", default_seed + idx)))
        sources = source_cfg_for_pair(default_sources, pair)
        pair_counts: dict[str, int] = {}
        pair_lines: list[str] = []

        for source_name, cfg in sources.items():
            if not isinstance(cfg, dict):
                raise SystemExit(f"Bad source config for {source_name}: {cfg!r}")
            enabled = bool(cfg.get("enabled", True))
            limit_raw = cfg.get("limit")
            limit = int(limit_raw) if limit_raw is not None else None
            if not enabled or limit == 0:
                pair_counts[source_name] = 0
                continue
            template = cfg.get("path_template")
            if not isinstance(template, str) or not template:
                raise SystemExit(f"Source {source_name} has no path_template")
            path = source_path(args.data_root, template, src, tgt)
            if not path.is_file():
                msg = f"missing {source_name} input for {src}->{tgt}: {path}"
                if args.strict:
                    raise SystemExit(msg)
                print(f"warn: {msg}", file=sys.stderr)
                pair_counts[source_name] = 0
                continue
            lines = read_jsonl_lines(path, skip_bad_lines=args.skip_bad_lines)
            picked = sample_lines(lines, limit, pair_rng)
            pair_lines.extend(picked)
            pair_counts[source_name] = len(picked)

        if args.dry_run:
            print(json.dumps({"src_lang": src, "tgt_lang": tgt, "source_counts": pair_counts}, ensure_ascii=False))
            continue

        merged.extend(pair_lines)
        manifest["pairs"].append(
            {
                "src_lang": src,
                "tgt_lang": tgt,
                "num_rows": len(pair_lines),
                "source_counts": pair_counts,
            }
        )
        print(f"collected {len(pair_lines):>7} {src}->{tgt} counts={pair_counts}")

    if args.dry_run:
        return 0

    rng.shuffle(merged)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.out_name}.jsonl"
    out_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")

    preview_dir = args.out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{args.out_name}.preview_{PREVIEW_N}.jsonl"
    preview = merged[:PREVIEW_N]
    preview_path.write_text("\n".join(preview) + ("\n" if preview else ""), encoding="utf-8")

    entry = dataset_info_entry(f"{args.out_subdir.rstrip('/')}/{args.out_name}.jsonl")
    snippet = {args.dataset_name: entry}
    args.dataset_info_snippet.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_info_snippet.write_text(json.dumps(snippet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_update_llamafactory_dataset_info:
        update_dataset_info(args.llamafactory_dataset_info, args.dataset_name, entry)

    manifest["num_rows"] = len(merged)
    manifest_path = args.out_dir / f"{args.out_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(merged)} rows: {out_path}")
    print(f"preview: {preview_path}")
    print(f"dataset_info snippet: {args.dataset_info_snippet}")
    if not args.no_update_llamafactory_dataset_info:
        print(f"updated LLaMA-Factory dataset_info: {args.llamafactory_dataset_info}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
