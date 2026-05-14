#!/usr/bin/env python3
"""
Download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 to models/.

Usage:
  python scripts/download_paraphrase_multilingual_minilm.py

  conda run -n lowres python scripts/download_paraphrase_multilingual_minilm.py \\
    --models-dir /path/to/models

Offline cache only (already fetched):
  HF_HUB_OFFLINE=1 python scripts/download_paraphrase_multilingual_minilm.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LOCAL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Download {REPO_ID} into models/.")
    ap.add_argument("--models-dir", type=Path, default=root() / "models")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument(
        "--local-name",
        default=DEFAULT_LOCAL_NAME,
        help="Subdirectory under --models-dir for the snapshot.",
    )
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    dest = args.models_dir / args.local_name
    print(f"[hf] {args.repo_id} -> {dest}")
    if args.dry_run:
        return 0

    from huggingface_hub import snapshot_download

    args.models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
