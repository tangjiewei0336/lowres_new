#!/usr/bin/env python3
"""
Generate synthetic MT pairs from FineWeb-2 monolingual corpora for all configured directions.

This generalizes the previous Thai-specific scripts to all low-resource directions
used in the MoE setup. Inputs are monolingual JSONL files prepared by:

  scripts/prepare/prepare_fineweb2_monolingual_for_llamafactory.py

Default output format:
  {"instruction": "...", "input": "<src>", "output": "<tgt>", "meta": {...}}

Instruction language is always English.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from flores_lang_zh import english_translation_instruction  # noqa: E402


PREVIEW_N = 50
TRADITIONAL_CHARS = set(
    "體臺灣國語門風龍馬鳥魚貝車長東電會學萬與專業網頁點擊"
    "後來開關樂書時過無對發現為這個們還進廣場線愛雲氣區"
    "醫藥術聽說讀寫貓畫戰爭讓從應當實驗數據資料軟體硬體"
)


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


def cjk_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def thai_count(text: str) -> int:
    return sum(1 for ch in text if "\u0e00" <= ch <= "\u0e7f")


def latin_count(text: str) -> int:
    return sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))


def has_traditional_chars(text: str) -> bool:
    return any(ch in TRADITIONAL_CHARS for ch in text)


def is_simplified_chinese(text: str, *, min_cjk_ratio: float) -> bool:
    letters = [ch for ch in text if not ch.isspace()]
    return bool(letters) and not has_traditional_chars(text) and (cjk_count(text) / max(1, len(letters)) >= min_cjk_ratio)


def is_thai(text: str, *, min_thai_ratio: float) -> bool:
    letters = [ch for ch in text if not ch.isspace()]
    return bool(letters) and (thai_count(text) / max(1, len(letters)) >= min_thai_ratio)


def is_latinish(text: str, *, min_latin_ratio: float) -> bool:
    letters = [ch for ch in text if not ch.isspace()]
    return bool(letters) and (latin_count(text) / max(1, len(letters)) >= min_latin_ratio)


def split_candidates(text: str, *, max_chars: int) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    rough = re.split(r"(?:\\n+|[.!?！？。；;])", text)
    out: list[str] = []
    buf = ""
    for part in rough:
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


def valid_source_lang(text: str, lang: str, *, min_thai_ratio: float, min_cjk_ratio: float, min_latin_ratio: float) -> bool:
    if lang == "zho_Hans":
        return is_simplified_chinese(text, min_cjk_ratio=min_cjk_ratio)
    if lang == "tha_Thai":
        return is_thai(text, min_thai_ratio=min_thai_ratio)
    return is_latinish(text, min_latin_ratio=min_latin_ratio)


def valid_target_lang(text: str, lang: str, *, min_thai_ratio: float, min_cjk_ratio: float, min_latin_ratio: float) -> bool:
    if not text.strip():
        return False
    return valid_source_lang(
        text,
        lang,
        min_thai_ratio=min_thai_ratio,
        min_cjk_ratio=min_cjk_ratio,
        min_latin_ratio=min_latin_ratio,
    )


def iter_clean_sources(
    path: Path,
    lang: str,
    *,
    min_chars: int,
    max_chars: int,
    min_thai_ratio: float,
    min_cjk_ratio: float,
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
            if not valid_source_lang(
                cand,
                lang,
                min_thai_ratio=min_thai_ratio,
                min_cjk_ratio=min_cjk_ratio,
                min_latin_ratio=min_latin_ratio,
            ):
                continue
            key = sha1_text(cand)
            if key in seen:
                continue
            seen.add(key)
            yield cand
            break


def build_instruction(src_lang: str, tgt_lang: str) -> str:
    inst = english_translation_instruction(src_lang, tgt_lang)
    if tgt_lang == "zho_Hans":
        return inst[:-1] + " Do not use Traditional Chinese."
    return inst


def mt_user_content(src_lang: str, tgt_lang: str, text: str) -> str:
    return f"{build_instruction(src_lang, tgt_lang)}\n\n{text}"


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    fam = (model_family or "").lower()
    if fam in ("qwen3", "qwen", "qwen3.5", "qwen3-5", "qwen35"):
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
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    assert last_err is not None
    raise last_err


def preview_path(out_path: Path) -> Path:
    prev_dir = out_path.parent / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    return prev_dir / f"{out_path.stem}.preview_{PREVIEW_N}.jsonl"


def refresh_preview(out_path: Path) -> None:
    prev = preview_path(out_path)
    rows: list[str] = []
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= PREVIEW_N:
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
            continue
        inp = row.get("input")
        if isinstance(inp, str):
            hashes.add(sha1_text(inp.strip()))
    return hashes


def load_pairs(path: Path, default_limit: int) -> list[tuple[str, str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise SystemExit(f"pairs missing in {path}")
    out: list[tuple[str, str, int]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src_lang", "")).strip()
        tgt = str(item.get("tgt_lang", "")).strip()
        if not src or not tgt:
            continue
        limit = int(item.get("limit", default_limit) or default_limit)
        out.append((src, tgt, limit))
    if not out:
        raise SystemExit(f"no usable pairs in {path}")
    return out


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
    refresh_preview_every: int,
    min_chars: int,
    max_chars: int,
    min_thai_ratio: float,
    min_cjk_ratio: float,
    min_latin_ratio: float,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_hashes = existing_source_hashes(out_path) if resume else set()
    existing_n = count_jsonl(out_path) if resume else 0
    target_new = max(0, limit - existing_n)
    if target_new == 0:
        refresh_preview(out_path)
        print(f"[{src_lang}->{tgt_lang}] already has {existing_n} rows; skip.")
        return existing_n

    mode = "a" if resume else "w"
    submitted = 0
    written_new = 0
    skipped_bad = 0
    source_iter = iter_clean_sources(
        input_path,
        src_lang,
        min_chars=min_chars,
        max_chars=max_chars,
        min_thai_ratio=min_thai_ratio,
        min_cjk_ratio=min_cjk_ratio,
        min_latin_ratio=min_latin_ratio,
    )

    def submit_one(ex: ThreadPoolExecutor, source_text: str) -> Future[tuple[str, str, str]]:
        source_hash = sha1_text(source_text)

        def _work() -> tuple[str, str, str]:
            out = call_translate(
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
            return source_text, source_hash, out

        return ex.submit(_work)

    with open(out_path, mode, encoding="utf-8") as fo, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures: set[Future[tuple[str, str, str]]] = set()
        pbar = tqdm(total=target_new, desc=f"{src_lang}->{tgt_lang}")

        def fill() -> None:
            nonlocal submitted
            while len(futures) < max_inflight and submitted < target_new:
                try:
                    src_text = next(source_iter)
                except StopIteration:
                    break
                h = sha1_text(src_text)
                if h in done_hashes:
                    continue
                done_hashes.add(h)
                futures.add(submit_one(ex, src_text))
                submitted += 1

        fill()
        while futures and written_new < target_new:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                src_text, source_hash, translated = fut.result()
                if not valid_target_lang(
                    translated,
                    tgt_lang,
                    min_thai_ratio=min_thai_ratio,
                    min_cjk_ratio=min_cjk_ratio,
                    min_latin_ratio=min_latin_ratio,
                ):
                    skipped_bad += 1
                    submitted -= 1
                    continue
                row = {
                    "instruction": build_instruction(src_lang, tgt_lang),
                    "input": src_text,
                    "output": translated,
                    "meta": {
                        "source": "fineweb2_synthetic_model",
                        "source_file": str(input_path),
                        "source_hash": source_hash,
                        "generator": served_model_name,
                        "api_base": base_url,
                    },
                }
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_new += 1
                pbar.update(1)
                if written_new % refresh_preview_every == 0:
                    fo.flush()
                    refresh_preview(out_path)
            fill()
        pbar.close()

    refresh_preview(out_path)
    total = count_jsonl(out_path)
    print(
        f"[{src_lang}->{tgt_lang}] wrote {written_new} new rows, total {total}; "
        f"invalid translations skipped {skipped_bad}; preview {preview_path(out_path)}"
    )
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate FineWeb-2 synthetic MT pairs for all configured directions.")
    ap.add_argument(
        "--pairs-config",
        type=Path,
        default=root() / "training" / "moe_pair_limits.json",
        help="Pairs config with src_lang/tgt_lang/limit.",
    )
    ap.add_argument(
        "--mono-dir",
        type=Path,
        default=root() / "training" / "data" / "monolingual",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root() / "training" / "data" / "multilingual" / "fineweb2_synth",
    )
    ap.add_argument("--default-limit", type=int, default=100_000)
    ap.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "qwen3.5-27b-instruct"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model-family", default=os.environ.get("EVAL_MODEL_FAMILY", "qwen3.5"))
    ap.add_argument(
        "--direction-workers",
        type=int,
        default=int(os.environ.get("FINEWEB_DIRECTION_WORKERS", "4")),
        help="How many translation directions to process in parallel. Default 4 for 4-GPU vLLM.",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("FINEWEB_MAX_WORKERS_PER_DIRECTION", "16")),
        help="Per-direction request worker threads.",
    )
    ap.add_argument(
        "--max-inflight",
        type=int,
        default=int(os.environ.get("FINEWEB_MAX_INFLIGHT_PER_DIRECTION", "64")),
        help="Per-direction in-flight requests.",
    )
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--min-thai-ratio", type=float, default=0.35)
    ap.add_argument("--min-cjk-ratio", type=float, default=0.45)
    ap.add_argument("--min-latin-ratio", type=float, default=0.45)
    ap.add_argument("--refresh-preview-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--strict-missing-input", action="store_true", help="Fail if any source monolingual file is missing.")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs_config, int(args.default_limit))
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs: list[str] = []
    runnable: list[tuple[str, str, int, Path]] = []
    for src_lang, tgt_lang, limit in pairs:
        input_path = args.mono_dir / f"fineweb2_pt_{src_lang}.jsonl"
        if not input_path.is_file():
            msg = f"[{src_lang}->{tgt_lang}] missing input: {input_path}"
            if args.strict_missing_input:
                raise SystemExit(msg)
            print(msg)
            missing_inputs.append(msg)
            continue

        runnable.append((src_lang, tgt_lang, int(limit), input_path))

    if not runnable:
        print("No runnable directions found.")
        return 0

    def _run_one(job: tuple[str, str, int, Path]) -> tuple[str, str, int]:
        src_lang, tgt_lang, limit, input_path = job
        total = generate_direction(
            client=client,
            served_model_name=args.served_model_name,
            base_url=args.base_url,
            input_path=input_path,
            out_path=args.out_dir / f"fineweb_synth_{src_lang}__{tgt_lang}.jsonl",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            limit=int(limit),
            max_workers=int(args.max_workers),
            max_inflight=int(args.max_inflight),
            max_tokens=int(args.max_tokens),
            model_family=str(args.model_family),
            request_timeout=float(args.request_timeout),
            max_retries=int(args.max_retries),
            resume=bool(args.resume),
            refresh_preview_every=max(1, int(args.refresh_preview_every)),
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
            min_thai_ratio=float(args.min_thai_ratio),
            min_cjk_ratio=float(args.min_cjk_ratio),
            min_latin_ratio=float(args.min_latin_ratio),
        )
        return src_lang, tgt_lang, total

    direction_workers = max(1, int(args.direction_workers))
    summaries: list[tuple[str, str, int]] = []
    with ThreadPoolExecutor(max_workers=direction_workers) as ex:
        futs = [ex.submit(_run_one, job) for job in runnable]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="directions"):
            summaries.append(fut.result())

    print(f"Done. Output dir: {args.out_dir}")
    for src_lang, tgt_lang, total in sorted(summaries):
        print(f"{src_lang}->{tgt_lang}: total_rows={total}")
    if missing_inputs:
        print(f"Skipped {len(missing_inputs)} directions due to missing monolingual input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
