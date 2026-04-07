#!/usr/bin/env python3
"""
小样本验证 ModelScope 数据集可加载、模型仓库可解析（尽量只拉取少量文件）。
需在 conda lowres 中运行: conda activate lowres && python scripts/check_modelscope_download.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_sources() -> dict:
    p = _root() / "modelscope_sources.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_dataset_flores(dataset_id: str) -> None:
    from modelscope.msdatasets import MsDataset

    # facebook/flores 等需执行数据集构建脚本（HuggingFace datasets 行为）
    ds = MsDataset.load(
        dataset_id, subset_name="eng_Latn", split="dev", trust_remote_code=True
    )
    n = len(ds)  # type: ignore[arg-type]
    if n < 1:
        raise RuntimeError(f"{dataset_id} eng_Latn dev 为空")
    row = ds[0]
    print(f"  FLORES OK: {dataset_id} eng_Latn dev 样本数={n}, 首条键={list(row.keys()) if isinstance(row, dict) else type(row)}")


def check_dataset_ntrex(dataset_id: str) -> None:
    from modelscope.msdatasets import MsDataset

    attempts: list[dict] = [
        {},
        {"split": "test"},
        {"split": "train"},
        {"split": "validation"},
        {"subset_name": "default", "split": "test"},
    ]
    last_err: Exception | None = None
    for kw in attempts:
        try:
            load_kw = {**kw, "trust_remote_code": True}
            ds = MsDataset.load(dataset_id, **load_kw)
            n = len(ds)  # type: ignore[arg-type]
            if n < 1:
                continue
            row = ds[0]
            keys = list(row.keys()) if isinstance(row, dict) else str(type(row))
            print(f"  NTREX OK: {dataset_id} {kw or '{no kwargs}'} 样本数={n}, 首条键={keys}")
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"NTREX 加载失败 dataset_id={dataset_id} 最后错误: {last_err}")


def check_model_repo(model_id: str) -> None:
    from modelscope import snapshot_download

    # 尽量只拉配置与小文件（若平台忽略 pattern，仍可能较大）
    cache_dir = os.environ.get("MODELSCOPE_CACHE") or str(_root() / "datasets" / "cache" / "models")
    os.makedirs(cache_dir, exist_ok=True)
    path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_file_pattern=["configuration.json", "config.json", "tokenizer_config.json", "generation_config.json"],
    )
    print(f"  模型元数据 OK: {model_id} -> {path}")


def main() -> int:
    os.environ.setdefault("MODELSCOPE_CACHE", str(_root() / "datasets" / "cache" / "modelscope"))
    src = _load_sources()
    ds_cfg = src["datasets"]
    models = src["models"]
    print("MODELSCOPE_CACHE=", os.environ["MODELSCOPE_CACHE"])
    try:
        print("检查 FLORES...")
        check_dataset_flores(ds_cfg["flores"]["dataset_id"])
        print("检查 NTREX...")
        check_dataset_ntrex(ds_cfg["ntrex"]["dataset_id"])
        print("检查模型仓库（轻量文件）...")
        for name, mid in models.items():
            print(f"  [{name}] {mid}")
            check_model_repo(mid)
    except Exception as e:
        print(f"ModelScope 检查失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    print("ModelScope 下载/加载检查完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
