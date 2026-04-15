#!/usr/bin/env python3
"""
仅通过 ModelScope 将基座权重完整下载到仓库 models/ 目录：
  models/<展示名>_<YYYYMMDD_HHMMSS>/

默认处理 modelscope_sources.json 中的 smollm3_3b、hunyuan_mt1_5_1_8b、qwen3_4b、qwen3_8b、qwen3_8b_base、qwen3_4b_instruct_2507、qwen3_5_27b、qwen3_5_27b_instruct
（Qwen3-8B Base Hub: https://huggingface.co/Qwen/Qwen3-8B-Base ；Instruct 对应 Hub: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ；Qwen3.5-27B Hub: https://huggingface.co/Qwen/Qwen3.5-27B ）。
用法:
  conda activate lowres
  python scripts/download_models_to_models_dir.py
  python scripts/download_models_to_models_dir.py --only qwen3_4b
  python scripts/download_models_to_models_dir.py --only qwen3_8b
  python scripts/download_models_to_models_dir.py --only qwen3_8b_base
  python scripts/download_models_to_models_dir.py --only qwen3_4b_instruct_2507
  python scripts/download_models_to_models_dir.py --only qwen3_5_27b
  python scripts/download_models_to_models_dir.py --only qwen3_5_27b_instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        action="append",
        dest="only_keys",
        metavar="KEY",
        help="仅下载指定键（可重复），如 qwen3_4b、qwen3_8b、qwen3_8b_base、qwen3_4b_instruct_2507、qwen3_5_27b、qwen3_5_27b_instruct；默认全部",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的操作")
    args = parser.parse_args()

    os.environ.setdefault("MODELSCOPE_CACHE", str(root() / "datasets" / "cache" / "modelscope"))

    with open(root() / "modelscope_sources.json", encoding="utf-8") as f:
        cfg = json.load(f)
    models: dict[str, str] = cfg["models"]

    if args.dry_run:
        snapshot_download = None
    else:
        from modelscope import snapshot_download

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_root = root() / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    all_keys = list(models.keys())
    if args.only_keys:
        keys = [k for k in args.only_keys if k in models]
        missing = set(args.only_keys) - set(models.keys())
        if missing:
            print(f"警告: 以下 --only 键不在 modelscope_sources.json 的 models 中: {sorted(missing)}")
        if not keys:
            print("没有可下载的模型键。", file=sys.stderr)
            return 1
    else:
        keys = all_keys

    display = {
        "smollm3_3b": "SmolLM3-3B",
        "hunyuan_mt1_5_1_8b": "HY-MT1.5-1.8B",
        "qwen3_4b": "Qwen3-4B",
        "qwen3_8b": "Qwen3-8B",
        "qwen3_8b_base": "Qwen3-8B-Base",
        "qwen3_4b_instruct_2507": "Qwen3-4B-Instruct-2507",
        "qwen3_5_27b": "Qwen3.5-27B",
        "qwen3_5_27b_instruct": "Qwen3.5-27B-Instruct",
    }

    for key in keys:
        mid = models[key]
        label = display.get(key, key)
        dest = models_root / f"{label}_{ts}"
        print(f"[{key}] {mid} -> {dest}")
        if args.dry_run:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        assert snapshot_download is not None
        path = snapshot_download(mid, local_dir=str(dest))
        print(f"  完成: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
