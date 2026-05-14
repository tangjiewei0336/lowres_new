#!/usr/bin/env python3
"""
Download the COMET-QE model used by reranking / sentence quality analysis.

Default:
  Unbabel/wmt22-cometkiwi-da -> models/Unbabel_wmt22-cometkiwi-da

Example:
  conda run -n lowres python scripts/download_comet_qe_model.py

With local encoder for offline COMET loading:
  conda run -n lowres python scripts/download_comet_qe_model.py --with-encoder
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


COMET_QE_REPO = "Unbabel/wmt22-cometkiwi-da"
XLM_ROBERTA_REPO = "FacebookAI/xlm-roberta-large"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Download COMET-QE model to models/.")
    ap.add_argument("--models-dir", type=Path, default=root() / "models")
    ap.add_argument("--repo-id", default=COMET_QE_REPO)
    ap.add_argument("--local-name", default="Unbabel_wmt22-cometkiwi-da")
    ap.add_argument("--with-encoder", action="store_true", help="Also download FacebookAI/xlm-roberta-large.")
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    qe_dir = args.models_dir / args.local_name
    encoder_dir = args.models_dir / "xlm-roberta-large"
    print(f"[comet-qe] {args.repo_id} -> {qe_dir}")
    if args.with_encoder:
        print(f"[encoder] {XLM_ROBERTA_REPO} -> {encoder_dir}")
    if args.dry_run:
        return 0

    from huggingface_hub import snapshot_download

    args.models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(qe_dir),
        local_dir_use_symlinks=False,
    )
    if args.with_encoder:
        snapshot_download(
            repo_id=XLM_ROBERTA_REPO,
            local_dir=str(encoder_dir),
            local_dir_use_symlinks=False,
        )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
