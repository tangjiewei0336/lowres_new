#!/usr/bin/env python3
"""
One-off repair/generation script for Thai MoE data.

What it does:
1. Rewrite existing zho_Hans<->tha_Thai synthetic files so every instruction is English.
2. Generate missing eng_Latn<->tha_Thai synthetic FineWeb-2 pairs with Qwen3.5-27B.
3. Support --resume by counting existing rows and tracking source_hash in meta.

Default input files:
  training/data/monolingual/fineweb2_pt_eng_Latn.jsonl
  training/data/monolingual/fineweb2_pt_tha_Thai.jsonl

Default output files:
  training/data/multilingual/fineweb2_synth/fineweb_synth_eng_Latn__tha_Thai.jsonl
  training/data/multilingual/fineweb2_synth/fineweb_synth_tha_Thai__eng_Latn.jsonl

Existing zho_Hans<->tha_Thai files are repaired in place by default, with
atomic replacement and refreshed previews.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI
from tqdm import tqdm

PREVIEW_N = 50
ENG = "eng_Latn"
THA = "tha_Thai"
ZHO = "zho_Hans"

LANG_EN = {
    ENG: "English",
    THA: "Thai",
    ZHO: "Simplified Chinese",
}


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def thai_count(text: str) -> int:
    return sum(1 for ch in text if "\u0e00" <= ch <= "\u0e7f")


def latin_count(text: str) -> int:
    return sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))


def is_thai(text: str, *, min_thai_ratio: float) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    return bool(chars) and thai_count(text) / max(1, len(chars)) >= min_thai_ratio


def is_englishish(text: str, *, min_latin_ratio: float) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    return bool(chars) and latin_count(text) / max(1, len(chars)) >= min_latin_ratio


def split_candidates(text: str, *, max_chars: int) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?:\\n+|[.!?！？。；;])", text)
    out: list[str] = []
    buf = ""
    for part in parts:
        part = normalize_ws(part)
        if not part:
            continue
        if len(part) > max_chars:
            part = part[:max_chars].strip()
        if not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= max_chars:
            buf = f"{buf} {part}"
        else:
            out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    return out


def iter_clean_sources(
    path: Path,
    lang: str,
    *,
    min_chars: int,
    max_chars: int,
    min_thai_ratio: float,
    min_latin_ratio: float,
) -> Iterator[str]:
    seen: set[str] = set()
    for row in read_jsonl(path):
        raw = row.get("text")
        if not isinstance(raw, str):
            continue
        for cand in split_candidates(raw, max_chars=max_chars):
            if len(cand) < min_chars:
                continue
            if lang == THA and not is_thai(cand, min_thai_ratio=min_thai_ratio):
                continue
            if lang == ENG and not is_englishish(cand, min_latin_ratio=min_latin_ratio):
                continue
            h = sha1_text(cand)
            if h in seen:
                continue
            seen.add(h)
            yield cand
            break


def make_instruction(src: str, tgt: str) -> str:
    src_name = LANG_EN.get(src, src)
    tgt_name = LANG_EN.get(tgt, tgt)
    extra = " Do not use Traditional Chinese." if tgt == ZHO else ""
    return f"Translate the following {src_name} text into {tgt_name}. Output only the translation.{extra}"


def mt_user_content(src_lang: str, tgt_lang: str, text: str) -> str:
    src_name = LANG_EN.get(src_lang, src_lang)
    tgt_name = LANG_EN.get(tgt_lang, tgt_lang)
    return f"Translate the following {src_name} text into {tgt_name}. Output only the translation, no explanations.\n\n{text}"


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    if model_family.lower() in ("qwen3", "qwen", "qwen3.5", "qwen3-5", "qwen35"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def call_translate(
    client: OpenAI,
    model: str,
    src_lang: str,
    tgt_lang: str,
    text: str,
    *,
    max_tokens: int,
    model_family: str,
    request_timeout: float,
    max_retries: int,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional machine translation engine."},
            {"role": "user", "content": mt_user_content(src_lang, tgt_lang, text)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "timeout": request_timeout,
    }
    extra = build_extra_body(model_family)
    if extra:
        kwargs["extra_body"] = extra

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            if attempt >= max_retries:
                break
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    assert last is not None
    raise last


def preview_path(out_path: Path, preview_n: int) -> Path:
    prev_dir = out_path.parent / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    return prev_dir / f"{out_path.stem}.preview_{preview_n}.jsonl"


def refresh_preview(out_path: Path, preview_n: int) -> None:
    prev = preview_path(out_path, preview_n)
    rows: list[str] = []
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= preview_n:
                    break
                if line.strip():
                    rows.append(line.rstrip("\n"))
    prev.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def existing_source_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.is_file():
        return hashes
    for row in read_jsonl(path):
        meta = row.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("source_hash"), str):
            hashes.add(meta["source_hash"])
        elif isinstance(row.get("input"), str):
            hashes.add(sha1_text(row["input"].strip()))
    return hashes


def repair_instruction_file(path: Path, src_lang: str, tgt_lang: str, *, preview_n: int, dry_run: bool) -> int:
    if not path.is_file():
        print(f"[repair] missing, skip: {path}")
        return 0
    changed = 0
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as fo:
            for row in read_jsonl(path):
                new_inst = make_instruction(src_lang, tgt_lang)
                if row.get("instruction") != new_inst:
                    row["instruction"] = new_inst
                    changed += 1
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        if dry_run:
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(path)
            refresh_preview(path, preview_n)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"[repair] {path}: changed={changed}")
    return changed


def valid_translation(tgt_lang: str, out: str, *, min_thai_ratio: float, min_latin_ratio: float) -> bool:
    out = out.strip()
    if not out:
        return False
    if tgt_lang == THA:
        return is_thai(out, min_thai_ratio=min_thai_ratio)
    if tgt_lang == ENG:
        return is_englishish(out, min_latin_ratio=min_latin_ratio)
    return True


def generate_direction(
    *,
    client: OpenAI,
    served_model_name: str,
    base_url: str,
    input_path: Path,
    out_path: Path,
    src_lang: str,
    tgt_lang: str,
    limit: int,
    max_workers: int,
    max_inflight: int,
    max_tokens: int,
    model_family: str,
    request_timeout: float,
    max_retries: int,
    resume: bool,
    preview_n: int,
    refresh_preview_every: int,
    min_chars: int,
    max_chars: int,
    min_thai_ratio: float,
    min_latin_ratio: float,
    dry_run: bool,
) -> int:
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_n = count_jsonl(out_path) if resume else 0
    target_new = max(0, limit - existing_n)
    if target_new == 0:
        refresh_preview(out_path, preview_n)
        print(f"[{src_lang}->{tgt_lang}] already has {existing_n} rows; skip")
        return existing_n
    if dry_run:
        print(f"[{src_lang}->{tgt_lang}] would generate {target_new} rows -> {out_path}")
        return existing_n

    done = existing_source_hashes(out_path) if resume else set()
    source_iter = iter_clean_sources(
        input_path,
        src_lang,
        min_chars=min_chars,
        max_chars=max_chars,
        min_thai_ratio=min_thai_ratio,
        min_latin_ratio=min_latin_ratio,
    )
    mode = "a" if resume else "w"
    submitted = 0
    written = 0
    skipped_bad = 0

    def submit_one(ex: ThreadPoolExecutor, source_text: str) -> Future[tuple[str, str, str]]:
        source_hash = sha1_text(source_text)

        def _work() -> tuple[str, str, str]:
            translated = call_translate(
                client,
                served_model_name,
                src_lang,
                tgt_lang,
                source_text,
                max_tokens=max_tokens,
                model_family=model_family,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )
            return source_text, source_hash, translated

        return ex.submit(_work)

    with open(out_path, mode, encoding="utf-8") as fo, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures: set[Future[tuple[str, str, str]]] = set()
        pbar = tqdm(total=target_new, desc=f"{src_lang}->{tgt_lang}")

        def fill() -> None:
            nonlocal submitted
            while len(futures) < max_inflight and submitted < target_new:
                try:
                    source_text = next(source_iter)
                except StopIteration:
                    break
                h = sha1_text(source_text)
                if h in done:
                    continue
                done.add(h)
                futures.add(submit_one(ex, source_text))
                submitted += 1

        fill()
        while futures and written < target_new:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in completed:
                src_text, source_hash, translated = fut.result()
                if not valid_translation(
                    tgt_lang,
                    translated,
                    min_thai_ratio=min_thai_ratio,
                    min_latin_ratio=min_latin_ratio,
                ):
                    skipped_bad += 1
                    submitted -= 1
                    continue
                row = {
                    "instruction": make_instruction(src_lang, tgt_lang),
                    "input": src_text,
                    "output": translated,
                    "meta": {
                        "source": "fineweb2_synthetic_qwen3.5_27b_oneoff_eng_tha",
                        "source_file": str(input_path),
                        "source_hash": source_hash,
                        "generator": served_model_name,
                        "api_base": base_url,
                    },
                }
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)
                if written % refresh_preview_every == 0:
                    fo.flush()
                    refresh_preview(out_path, preview_n)
            fill()
        pbar.close()

    refresh_preview(out_path, preview_n)
    total = count_jsonl(out_path)
    print(f"[{src_lang}->{tgt_lang}] wrote={written}, total={total}, skipped_bad={skipped_bad}, preview={preview_path(out_path, preview_n)}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair zh-th instructions to English and generate eng<->tha FineWeb synthetic data.")
    ap.add_argument("--eng-input", type=Path, default=root() / "training" / "data" / "monolingual" / "fineweb2_pt_eng_Latn.jsonl")
    ap.add_argument("--thai-input", type=Path, default=root() / "training" / "data" / "monolingual" / "fineweb2_pt_tha_Thai.jsonl")
    ap.add_argument("--out-dir", type=Path, default=root() / "training" / "data" / "multilingual" / "fineweb2_synth")
    ap.add_argument("--limit-per-direction", type=int, default=100_000)
    ap.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "qwen3.5-27b-instruct"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model-family", default=os.environ.get("EVAL_MODEL_FAMILY", "qwen3.5"))
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--max-inflight", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--min-thai-ratio", type=float, default=0.35)
    ap.add_argument("--min-latin-ratio", type=float, default=0.55)
    ap.add_argument("--preview-n", type=int, default=PREVIEW_N)
    ap.add_argument("--refresh-preview-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-repair", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_repair:
        repair_instruction_file(out_dir / f"fineweb_synth_{THA}__{ZHO}.jsonl", THA, ZHO, preview_n=args.preview_n, dry_run=args.dry_run)
        repair_instruction_file(out_dir / f"fineweb_synth_{ZHO}__{THA}.jsonl", ZHO, THA, preview_n=args.preview_n, dry_run=args.dry_run)

    if args.skip_generate:
        return 0

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    common = dict(
        client=client,
        served_model_name=args.served_model_name,
        base_url=args.base_url,
        limit=int(args.limit_per_direction),
        max_workers=int(args.max_workers),
        max_inflight=int(args.max_inflight),
        max_tokens=int(args.max_tokens),
        model_family=str(args.model_family),
        request_timeout=float(args.request_timeout),
        max_retries=int(args.max_retries),
        resume=bool(args.resume),
        preview_n=int(args.preview_n),
        refresh_preview_every=max(1, int(args.refresh_preview_every)),
        min_chars=int(args.min_chars),
        max_chars=int(args.max_chars),
        min_thai_ratio=float(args.min_thai_ratio),
        min_latin_ratio=float(args.min_latin_ratio),
        dry_run=bool(args.dry_run),
    )
    generate_direction(
        **common,
        input_path=args.eng_input,
        out_path=out_dir / f"fineweb_synth_{ENG}__{THA}.jsonl",
        src_lang=ENG,
        tgt_lang=THA,
    )
    generate_direction(
        **common,
        input_path=args.thai_input,
        out_path=out_dir / f"fineweb_synth_{THA}__{ENG}.jsonl",
        src_lang=THA,
        tgt_lang=ENG,
    )
    print(f"Done. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
