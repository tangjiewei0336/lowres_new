#!/usr/bin/env python3
"""
将基座权重和评测模型完整下载到仓库 models/ 目录：
  models/<展示名>_<YYYYMMDD_HHMMSS>/

默认处理 modelscope_sources.json 中的 smollm3_3b、hunyuan_mt1_5_1_8b、qwen3_4b、qwen3_8b、qwen3_8b_base、qwen3_4b_instruct_2507、qwen3_5_27b、qwen3_5_27b_instruct
（Qwen3-8B Base Hub: https://huggingface.co/Qwen/Qwen3-8B-Base ；Instruct 对应 Hub: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ；Qwen3.5-27B Hub: https://huggingface.co/Qwen/Qwen3.5-27B ）。
另会处理 comet_models 中的 comet_wmt22_da，使用 huggingface_hub 下载到 models/Unbabel_wmt22-comet-da，
与 scripts/run/run_eval.py 的默认 --comet-model 路径一致。
用法:
  conda activate lowres
  python scripts/download_models_to_models_dir.py
  python scripts/download_models_to_models_dir.py --only qwen3_4b
  python scripts/download_models_to_models_dir.py --only qwen3_8b
  python scripts/download_models_to_models_dir.py --only qwen3_8b_base
  python scripts/download_models_to_models_dir.py --only qwen3_4b_instruct_2507
  python scripts/download_models_to_models_dir.py --only qwen3_5_27b
  python scripts/download_models_to_models_dir.py --only qwen3_5_27b_instruct
  python scripts/download_models_to_models_dir.py --only comet_wmt22_da
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


def comet_local_dir_for_key(models_root: Path, key: str) -> Path:
    if key == "comet_wmt22_da":
        return models_root / "Unbabel_wmt22-comet-da"
    return models_root / key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        action="append",
        dest="only_keys",
        metavar="KEY",
        help=(
            "仅下载指定键（可重复），如 qwen3_4b、qwen3_8b、qwen3_8b_base、"
            "qwen3_4b_instruct_2507、qwen3_5_27b、qwen3_5_27b_instruct、"
            "comet_wmt22_da；默认全部"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的操作")
    args = parser.parse_args()

    os.environ.setdefault("MODELSCOPE_CACHE", str(root() / "datasets" / "cache" / "modelscope"))

    with open(root() / "modelscope_sources.json", encoding="utf-8") as f:
        cfg = json.load(f)
    models: dict[str, str] = cfg["models"]
    comet_models: dict[str, str] = cfg.get("comet_models", {})

    if args.dry_run:
        snapshot_download = None
        hf_snapshot_download = None
    else:
        from modelscope import snapshot_download
        from huggingface_hub import snapshot_download as hf_snapshot_download

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_root = root() / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    all_keys = list(models.keys()) + list(comet_models.keys())
    if args.only_keys:
        known_keys = set(models.keys()) | set(comet_models.keys())
        keys = [k for k in args.only_keys if k in known_keys]
        missing = set(args.only_keys) - known_keys
        if missing:
            print(f"警告: 以下 --only 键不在 modelscope_sources.json 的 models/comet_models 中: {sorted(missing)}")
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
        "comet_wmt22_da": "Unbabel_wmt22-comet-da",
    }

    for key in keys:
        if key in comet_models:
            mid = comet_models[key]
            dest = comet_local_dir_for_key(models_root, key)
            print(f"[{key}] {mid} -> {dest}")
            if args.dry_run:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            assert hf_snapshot_download is not None
            path = hf_snapshot_download(
                repo_id=mid,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )
            print(f"  完成: {path}")
            continue

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
