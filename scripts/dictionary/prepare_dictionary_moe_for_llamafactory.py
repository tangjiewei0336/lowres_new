#!/usr/bin/env python3
"""
One-command wrapper for dictionary-based MoE MT data.

Steps:
  1. Download/normalize MUSE low-resource<->English dictionaries.
  2. Download/normalize CC-CEDICT Chinese<->English dictionary.
  3. Build low-resource<->Chinese dictionaries through English pivot.
  4. Export Alpaca JSONL and dataset_info snippet for LLaMA-Factory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    root = repo_root()
    py = sys.executable
    base = root / "scripts/dictionary"

    ap = argparse.ArgumentParser(description="Prepare dictionary-based MoE Alpaca JSONL.")
    ap.add_argument(
        "--limit-per-direction",
        type=int,
        default=None,
        help="Limit final pivot lexicons and Alpaca outputs per direction. Normalization remains full by default.",
    )
    ap.add_argument(
        "--normalize-limit-per-direction",
        type=int,
        default=None,
        help="Optional limit for raw dictionary normalization. Usually leave unset so English pivots can match.",
    )
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--skip-download-normalize", action="store_true")
    ap.add_argument("--include-meta", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    limit_args = []
    if args.limit_per_direction is not None:
        limit_args = ["--limit-per-direction", str(args.limit_per_direction)]
    normalize_limit_args = []
    if args.normalize_limit_per_direction is not None:
        normalize_limit_args = ["--limit-per-direction", str(args.normalize_limit_per_direction)]

    if not args.skip_download_normalize:
        force = ["--force-download"] if args.force_download else []
        # Keep CC-CEDICT available as a Chinese lexical source, then let MUSE zh-en
        # be the default English pivot. CC-CEDICT definitions are high-quality
        # explanations, but reverse lookup from English definitions is too noisy
        # for automatic low-resource<->Chinese pivot data.
        run([py, str(base / "prepare_cc_cedict_dictionary.py"), *normalize_limit_args, *force], dry_run=args.dry_run)
        run([py, str(base / "prepare_muse_dictionary.py"), *normalize_limit_args, *force], dry_run=args.dry_run)

    run([py, str(base / "build_pivot_dictionary.py"), *limit_args, "--overwrite"], dry_run=args.dry_run)

    meta = ["--include-meta"] if args.include_meta else []
    run([py, str(base / "build_dictionary_mt_jsonl.py"), *limit_args, *meta], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
