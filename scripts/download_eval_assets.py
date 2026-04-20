#!/usr/bin/env python3
"""
Download evaluation assets on an internet-connected machine.

Assets:
  - COMET: Unbabel/wmt22-comet-da -> models/Unbabel_wmt22-comet-da
  - COMET encoder: FacebookAI/xlm-roberta-large -> models/xlm-roberta-large
  - spBLEU/FLORES200 SPM:
      OpenNMT/nllb-200-onmt/flores200_sacrebleu_tokenizer_spm.model
      -> models/sacrebleu/flores200_sacrebleu_tokenizer_spm.model

The FLORES200 SPM file avoids sacrebleu's runtime tinyurl download path, which
can fail on machines with restricted TLS/certificates.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path


COMET_REPO = "Unbabel/wmt22-comet-da"
XLM_ROBERTA_REPO = "FacebookAI/xlm-roberta-large"
SPBLEU_REPO = "OpenNMT/nllb-200-onmt"
SPBLEU_FILENAME = "flores200_sacrebleu_tokenizer_spm.model"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_to_tar(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    if path.exists():
        tf.add(path, arcname=arcname)


def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Download COMET and spBLEU/FLORES200 evaluation assets.")
    ap.add_argument("--models-dir", type=Path, default=root / "models")
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    ap.add_argument("--skip-comet", action="store_true")
    ap.add_argument("--skip-xlm-roberta", action="store_true")
    ap.add_argument("--skip-spbleu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Optional .tar.gz bundle path to create after download, e.g. eval_assets.tar.gz.",
    )
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    comet_dir = args.models_dir / "Unbabel_wmt22-comet-da"
    xlm_roberta_dir = args.models_dir / "xlm-roberta-large"
    spbleu_dir = args.models_dir / "sacrebleu"
    spbleu_path = spbleu_dir / SPBLEU_FILENAME

    print(f"models_dir={args.models_dir}")
    if not args.skip_comet:
        print(f"[comet] {COMET_REPO} -> {comet_dir}")
    if not args.skip_xlm_roberta:
        print(f"[xlm-roberta] {XLM_ROBERTA_REPO} -> {xlm_roberta_dir}")
    if not args.skip_spbleu:
        print(f"[spbleu] {SPBLEU_REPO}/{SPBLEU_FILENAME} -> {spbleu_path}")
    if args.dry_run:
        return 0

    from huggingface_hub import hf_hub_download, snapshot_download

    args.models_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_comet:
        snapshot_download(
            repo_id=COMET_REPO,
            local_dir=str(comet_dir),
            local_dir_use_symlinks=False,
        )

    if not args.skip_xlm_roberta:
        snapshot_download(
            repo_id=XLM_ROBERTA_REPO,
            local_dir=str(xlm_roberta_dir),
            local_dir_use_symlinks=False,
        )

    if not args.skip_spbleu:
        spbleu_dir.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=SPBLEU_REPO,
                filename=SPBLEU_FILENAME,
                local_dir=str(spbleu_dir),
                local_dir_use_symlinks=False,
            )
        )
        if downloaded.resolve() != spbleu_path.resolve():
            shutil.copyfile(downloaded, spbleu_path)

    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.bundle, "w:gz") as tf:
            add_to_tar(tf, comet_dir, "models/Unbabel_wmt22-comet-da")
            add_to_tar(tf, xlm_roberta_dir, "models/xlm-roberta-large")
            add_to_tar(tf, spbleu_path, f"models/sacrebleu/{SPBLEU_FILENAME}")
        print(f"bundle={args.bundle}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
