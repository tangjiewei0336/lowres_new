#!/usr/bin/env python3
"""
Prepare English monolingual data from HuggingFaceFW/fineweb for LLaMA-Factory.

Default output intentionally matches the path expected by the Thai-English
synthetic generation script:
  training/data/monolingual/fineweb2_pt_eng_Latn.jsonl

Each output row:
  {"text": "..."}

The default FineWeb config is sample-10BT to avoid accidentally streaming the
full dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PREVIEW_N = 50


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def latin_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    latin = sum(1 for ch in chars if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    return latin / len(chars)


def resolve_text(row: dict[str, Any], keys: list[str], min_chars: int, min_latin_ratio: float) -> str | None:
    candidates: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, str):
            candidates.append(value)
    if not candidates:
        for value in row.values():
            if isinstance(value, str):
                candidates.append(value)

    for raw in candidates:
        text = normalize_ws(raw)
        if len(text) < min_chars:
            continue
        if latin_ratio(text) < min_latin_ratio:
            continue
        return text
    return None


def write_dataset_info(path: Path, out_file_name: str, dataset_name: str) -> None:
    ensure_dir(path.parent)
    entry = {
        dataset_name: {
            "file_name": out_file_name,
            "formatting": "alpaca",
            "columns": {"prompt": "text", "response": None},
        }
    }
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise SystemExit(f"dataset_info is not a JSON object: {path}")
    existing.update(entry)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Some scripts use both names.
    sibling = path.with_name("dataset_info.json" if path.name == "dataset.info" else "dataset.info")
    sibling.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_hf_load_dataset():
    """Import Hugging Face datasets even when repo-root ./datasets shadows it."""
    repo = str(root())
    old_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in ("", repo)]
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "未安装 Hugging Face datasets 包。请运行：\n"
            "  conda activate lowres\n"
            "  pip install datasets\n"
        ) from e
    finally:
        sys.path = old_path
    return load_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="FineWeb English -> LLaMA-Factory monolingual JSONL")
    ap.add_argument("--repo-id", default="HuggingFaceFW/fineweb")
    ap.add_argument(
        "--config",
        default="sample-10BT",
        help="FineWeb config. Default sample-10BT; use default/full configs only intentionally.",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=250_000, help="Max rows to write; 0 means unlimited.")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--min-latin-ratio", type=float, default=0.55)
    ap.add_argument("--text-key", action="append", dest="text_keys", default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root() / "training" / "data" / "monolingual",
    )
    ap.add_argument(
        "--out-name",
        default="fineweb2_pt_eng_Latn.jsonl",
        help="Output file name. Kept compatible with existing scripts by default.",
    )
    ap.add_argument("--dataset-name", default="fineweb2_pt_eng_Latn")
    ap.add_argument("--dataset-info-path", type=Path, default=None)
    ap.add_argument("--no-dataset-info", action="store_true")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument(
        "--no-hard-exit",
        action="store_true",
        help="Disable hard exit workaround for datasets streaming shutdown issues.",
    )
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(root() / "datasets" / "cache" / "huggingface"))

    token: str | bool | None
    if args.hf_token:
        token = args.hf_token
    elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    else:
        token = None

    out_dir: Path = args.out_dir
    prev_dir = out_dir / "previews"
    ensure_dir(out_dir)
    ensure_dir(prev_dir)

    out_path = out_dir / args.out_name
    prev_path = prev_dir / f"{out_path.stem}.preview_{PREVIEW_N}.jsonl"
    text_keys = args.text_keys or ["text", "content", "raw_content"]
    limit = args.limit if args.limit and args.limit > 0 else None

    load_dataset = import_hf_load_dataset()

    print(f"Loading {args.repo_id} config={args.config} split={args.split} streaming=True")
    print(f"Output: {out_path}")
    ds = load_dataset(
        args.repo_id,
        args.config,
        split=args.split,
        streaming=True,
        token=token,
    )

    written = 0
    preview_lines: list[str] = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for row in ds:
            if limit is not None and written >= limit:
                break
            if not isinstance(row, dict):
                row = dict(row)
            text = resolve_text(row, text_keys, args.min_chars, args.min_latin_ratio)
            if text is None:
                continue
            line = json.dumps({"text": text}, ensure_ascii=False)
            fo.write(line + "\n")
            if written < PREVIEW_N:
                preview_lines.append(line)
            written += 1
            if written % 10_000 == 0:
                print(f"... wrote {written} rows")

    prev_path.write_text("\n".join(preview_lines) + ("\n" if preview_lines else ""), encoding="utf-8")
    print(f"Done: {out_path}, rows={written}")
    print(f"Preview: {prev_path}")

    if not args.no_dataset_info:
        dataset_info_path = args.dataset_info_path or (out_dir / "dataset.info")
        write_dataset_info(dataset_info_path, out_path.name, args.dataset_name)
        print(f"Updated dataset info: {dataset_info_path} and {dataset_info_path.with_name('dataset_info.json')}")

    if not args.no_hard_exit:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
