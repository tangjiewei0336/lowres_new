#!/usr/bin/env python3
"""
Build per-direction mixed MoE training files for LLaMA-Factory.

Default mix:
  nllb_moe + fineweb2_synth

Dictionary data is supported by the config schema but disabled by default.
Each output remains ordinary Alpaca JSONL:
  {"instruction": "...", "input": "...", "output": "..."}
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


def dataset_info_entry(src: str, tgt: str, out_subdir: str) -> dict[str, Any]:
    return {
        "file_name": f"{out_subdir.rstrip('/')}/mixed_moe_{src}__{tgt}.jsonl",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }


def update_dataset_info(path: Path, snippet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{path} must contain a JSON object")
    else:
        data = {}
    data.update(snippet)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def build_one_pair(
    *,
    data_root: Path,
    out_dir: Path,
    src: str,
    tgt: str,
    sources: dict[str, Any],
    seed: int,
    strict: bool,
    skip_bad_lines: bool,
) -> tuple[int, dict[str, int]]:
    rng = random.Random(seed)
    merged: list[str] = []
    counts: dict[str, int] = {}

    for source_name, cfg in sources.items():
        if not isinstance(cfg, dict):
            raise SystemExit(f"Bad source config for {source_name}: {cfg!r}")
        enabled = bool(cfg.get("enabled", True))
        limit_raw = cfg.get("limit")
        limit = int(limit_raw) if limit_raw is not None else None
        if not enabled or limit == 0:
            counts[source_name] = 0
            continue
        template = cfg.get("path_template")
        if not isinstance(template, str) or not template:
            raise SystemExit(f"Source {source_name} has no path_template")
        path = source_path(data_root, template, src, tgt)
        if not path.is_file():
            msg = f"missing {source_name} input for {src}->{tgt}: {path}"
            if strict:
                raise SystemExit(msg)
            print(f"warn: {msg}", file=sys.stderr)
            counts[source_name] = 0
            continue
        lines = read_jsonl_lines(path, skip_bad_lines=skip_bad_lines)
        picked = sample_lines(lines, limit, rng)
        merged.extend(picked)
        counts[source_name] = len(picked)

    rng.shuffle(merged)
    out_path = out_dir / f"mixed_moe_{src}__{tgt}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")

    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"mixed_moe_{src}__{tgt}.preview_{PREVIEW_N}.jsonl"
    preview = merged[:PREVIEW_N]
    preview_path.write_text("\n".join(preview) + ("\n" if preview else ""), encoding="utf-8")
    return len(merged), counts


def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Build mixed MoE Alpaca JSONL per language direction.")
    ap.add_argument("--config", type=Path, default=root / "training/moe_data_mix_config.json")
    ap.add_argument("--data-root", type=Path, default=root / "training/data")
    ap.add_argument("--out-dir", type=Path, default=root / "training/data/multilingual/mixed_moe")
    ap.add_argument("--out-subdir", default="multilingual/mixed_moe")
    ap.add_argument("--dataset-info-snippet", type=Path, default=root / "training/mixed_moe_dataset_info.snippet.json")
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
    snippet: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "out_subdir": args.out_subdir,
        "pairs": [],
    }
    total = 0

    for idx, pair in enumerate(pairs):
        src = str(pair.get("src_lang", "")).strip()
        tgt = str(pair.get("tgt_lang", "")).strip()
        if not src or not tgt:
            raise SystemExit(f"Bad pair item: {pair!r}")
        seed = int(pair.get("seed", default_seed + idx))
        sources = source_cfg_for_pair(default_sources, pair)
        if args.dry_run:
            print(json.dumps({"src_lang": src, "tgt_lang": tgt, "seed": seed, "sources": sources}, ensure_ascii=False))
            continue
        n, counts = build_one_pair(
            data_root=args.data_root,
            out_dir=args.out_dir,
            src=src,
            tgt=tgt,
            sources=sources,
            seed=seed,
            strict=args.strict,
            skip_bad_lines=args.skip_bad_lines,
        )
        total += n
        dataset_name = f"mixed_moe_{src}_to_{tgt}"
        snippet[dataset_name] = dataset_info_entry(src, tgt, args.out_subdir)
        manifest["pairs"].append(
            {
                "src_lang": src,
                "tgt_lang": tgt,
                "dataset": dataset_name,
                "file_name": f"{args.out_subdir.rstrip('/')}/mixed_moe_{src}__{tgt}.jsonl",
                "num_rows": n,
                "source_counts": counts,
            }
        )
        print(f"wrote {n:>7} {src}->{tgt} counts={counts}")

    if args.dry_run:
        return 0

    args.dataset_info_snippet.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_info_snippet.write_text(json.dumps(snippet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = args.out_dir / "mixed_moe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_update_llamafactory_dataset_info:
        update_dataset_info(args.llamafactory_dataset_info, snippet)
        print(f"updated LLaMA-Factory dataset_info: {args.llamafactory_dataset_info}")
    print(f"wrote dataset_info snippet: {args.dataset_info_snippet}")
    print(f"wrote manifest: {manifest_path}")
    print(f"done: {total} mixed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
