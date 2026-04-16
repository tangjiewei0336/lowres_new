#!/usr/bin/env python3
"""
Prepare NLLB pair data for the LoRA-MoE experiment.

This is a small wrapper around prepare_nllb_for_llamafactory.py with the
correct MoE defaults:
  --pairs-config training/moe_pair_limits.json
  --export-from-config
  --out-subdir multilingual/nllb_moe

It writes:
  training/data/multilingual/nllb_moe/nllb_mt_<src>__<tgt>.jsonl
  training/data/multilingual/nllb_moe/previews/*.preview_50.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare NLLB data under training/data/multilingual/nllb_moe.")
    ap.add_argument(
        "--pairs-config",
        type=Path,
        default=root() / "training" / "moe_pair_limits.json",
        help="MoE pairs config with a top-level pairs list.",
    )
    ap.add_argument(
        "--out-subdir",
        default="multilingual/nllb_moe",
        help="Output subdir relative to training/data.",
    )
    args = ap.parse_args()

    script = root() / "scripts" / "prepare" / "prepare_nllb_for_llamafactory.py"
    cmd = [
        sys.executable,
        str(script),
        "--pairs-config",
        str(args.pairs_config),
        "--export-from-config",
        "--out-subdir",
        str(args.out_subdir),
    ]
    print("运行:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
