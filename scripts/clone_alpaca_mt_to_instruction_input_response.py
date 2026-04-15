#!/usr/bin/env python3
"""
Clone Alpaca MT JSONL files into single-column Instruction/Input/Response JSONL.

Input rows:
  {"instruction": "...", "input": "...", "output": "..."}

Output rows:
  {"text": "Instruction: ...\\nInput: ...\\nResponse: ..."}

This is useful for training configs that expect a single prompt/completion text
column instead of Alpaca columns. Preview files are written alongside outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

PREVIEW_N = 50
DEFAULT_SOURCE_DIRS = (
    "training/data/multilingual/nllb_moe",
    "training/data/multilingual/fineweb2_synth",
)


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_preview(out_path: Path, preview_n: int) -> Path:
    prev_dir = out_path.parent / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev_path = prev_dir / f"{out_path.stem}.preview_{preview_n}.jsonl"
    lines: list[str] = []
    with open(out_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= preview_n:
                break
            lines.append(line.rstrip("\n"))
    prev_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return prev_path


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def format_block(instruction: str, input_text: str, output_text: str) -> str:
    return f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"


def infer_out_path(input_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    rel = input_path.relative_to(input_root)
    return output_root / rel.with_name(f"{rel.stem}{suffix}.jsonl")


def infer_dataset_name(out_path: Path, output_root: Path) -> str:
    rel = out_path.relative_to(output_root)
    stem = rel.with_suffix("").as_posix()
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", stem)
    return f"{safe}_text"


def convert_one(
    input_path: Path,
    out_path: Path,
    *,
    preview_n: int,
    overwrite: bool,
    limit: int | None,
    keep_meta: bool,
) -> tuple[int, Path]:
    if out_path.exists() and not overwrite:
        raise SystemExit(f"输出已存在，避免覆盖：{out_path}（可传 --overwrite）")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for row in read_jsonl(input_path):
            if limit is not None and written >= limit:
                break
            instruction = row.get("instruction")
            input_text = row.get("input")
            output_text = row.get("output")
            if not isinstance(instruction, str) or not isinstance(input_text, str) or not isinstance(output_text, str):
                continue
            instruction = clean_text(instruction)
            input_text = clean_text(input_text)
            output_text = clean_text(output_text)
            if not instruction or not input_text or not output_text:
                continue
            rec: dict[str, Any] = {"text": format_block(instruction, input_text, output_text)}
            if keep_meta and isinstance(row.get("meta"), dict):
                rec["meta"] = row["meta"]
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    prev_path = write_preview(out_path, preview_n)
    return written, prev_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Clone Alpaca MT JSONL into Instruction/Input/Response text JSONL.")
    ap.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="Input Alpaca JSONL file. Can be repeated.",
    )
    ap.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob under --input-dir when --input is not provided. Default: *.jsonl",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        dest="input_dirs",
        help=(
            "Directory to scan when --input is not provided. Can be repeated. "
            "Defaults to nllb_moe and fineweb2_synth when they exist."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Single output directory. If omitted, each input dir is cloned to <input_dir>_iir.",
    )
    ap.add_argument("--suffix", default="_iir", help="Suffix added before .jsonl for inferred output names.")
    ap.add_argument("--preview-n", type=int, default=PREVIEW_N)
    ap.add_argument("--limit", type=int, default=0, help="Optional per-file max rows; 0 means no limit.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-meta", action="store_true", help="Copy meta object into output rows when present.")
    ap.add_argument(
        "--dataset-info",
        type=Path,
        default=root() / "training" / "moe_iir_dataset_info.snippet.json",
        help="Write dataset_info snippet for cloned text datasets.",
    )
    ap.add_argument(
        "--update-dataset-info",
        type=Path,
        default=root() / "training" / "data" / "dataset_info.json",
        help="Create/update LLaMA-Factory dataset_info.json. Pass --no-update-dataset-info to skip.",
    )
    ap.add_argument("--no-update-dataset-info", action="store_true")
    args = ap.parse_args()

    repo = root()
    if args.inputs:
        jobs: list[tuple[Path, Path, Path]] = []
        output_root = args.output_dir or (repo / "training" / "data" / "multilingual" / "iir_clones")
        for p in args.inputs:
            input_root = p.parent
            jobs.append((p, input_root, output_root))
    else:
        raw_dirs = args.input_dirs
        if not raw_dirs:
            raw_dirs = [repo / x for x in DEFAULT_SOURCE_DIRS if (repo / x).is_dir()]
        jobs = []
        for input_dir in raw_dirs:
            input_dir = input_dir.resolve()
            output_root = args.output_dir or input_dir.with_name(f"{input_dir.name}_iir")
            files = sorted(
                p for p in input_dir.glob(args.glob)
                if p.is_file() and "previews" not in p.parts and not p.name.endswith(f"{args.suffix}.jsonl")
            )
            for p in files:
                jobs.append((p, input_dir, output_root))

    if not jobs:
        defaults = ", ".join(DEFAULT_SOURCE_DIRS)
        raise SystemExit(f"未找到输入文件。默认扫描：{defaults}；可用 --input-dir 或 --input 指定。")

    limit = args.limit if args.limit and args.limit > 0 else None
    total = 0
    dataset_entries: dict[str, dict[str, Any]] = {}
    for input_path, input_root, output_root in jobs:
        if not input_path.is_file():
            raise SystemExit(f"输入不存在：{input_path}")
        try:
            out_path = infer_out_path(input_path, input_root, output_root, args.suffix)
        except ValueError:
            out_path = output_root / f"{input_path.stem}{args.suffix}.jsonl"
        n, prev = convert_one(
            input_path=input_path,
            out_path=out_path,
            preview_n=args.preview_n,
            overwrite=args.overwrite,
            limit=limit,
            keep_meta=bool(args.keep_meta),
        )
        total += n
        print(f"{input_path} -> {out_path} rows={n}; preview={prev}")
        try:
            file_name = str(out_path.relative_to(repo / "training" / "data"))
        except ValueError:
            file_name = str(out_path)
        dataset_entries[infer_dataset_name(out_path, output_root)] = {
            "file_name": file_name,
            "formatting": "alpaca",
            "columns": {"prompt": "text", "response": None},
        }

    if dataset_entries:
        args.dataset_info.parent.mkdir(parents=True, exist_ok=True)
        args.dataset_info.write_text(json.dumps(dataset_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"dataset_info snippet: {args.dataset_info}")

        if not args.no_update_dataset_info:
            merged: dict[str, Any] = {}
            if args.update_dataset_info.is_file():
                merged = json.loads(args.update_dataset_info.read_text(encoding="utf-8"))
                if not isinstance(merged, dict):
                    raise SystemExit(f"dataset_info is not a JSON object: {args.update_dataset_info}")
            merged.update(dataset_entries)
            args.update_dataset_info.parent.mkdir(parents=True, exist_ok=True)
            args.update_dataset_info.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"updated dataset_info: {args.update_dataset_info}")

    print(f"完成：{len(jobs)} 个文件，共 {total} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
