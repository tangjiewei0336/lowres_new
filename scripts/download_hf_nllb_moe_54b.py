#!/usr/bin/env python3
"""
Download facebook/nllb-moe-54b from Hugging Face into this repository's
models/ directory.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download facebook/nllb-moe-54b to models/")
    parser.add_argument(
        "--repo-id",
        default="facebook/nllb-moe-54b",
        help="Hugging Face repo id.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=root() / "models" / "facebook_nllb-moe-54b",
        help="Target directory under this project.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved target path.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(root() / "datasets" / "cache" / "huggingface"))
    args.local_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.repo_id} -> {args.local_dir}")
    if args.dry_run:
        return 0

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"完成: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
