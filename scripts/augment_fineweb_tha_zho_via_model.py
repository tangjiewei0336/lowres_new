#!/usr/bin/env python3
"""
Generate synthetic Thai<->Simplified-Chinese MT pairs from FineWeb-2 monolingual JSONL.

Default inputs are produced by:
  scripts/prepare/prepare_fineweb2_monolingual_for_llamafactory.py

Default task:
  - Read Thai FineWeb text, translate it to Simplified Chinese, write tha_Thai -> zho_Hans.
  - Read Simplified-Chinese FineWeb text, translate it to Thai, write zho_Hans -> tha_Thai.
  - 100k rows per direction, 200k rows total.

Output format matches the existing Alpaca MT format:
  {"instruction": "...", "input": "<src>", "output": "<tgt>"}

The script appends incrementally and supports --resume. Preview files are written
under previews/ and refreshed during generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI
from tqdm import tqdm

PREVIEW_N = 50
THA = "tha_Thai"
ZHO = "zho_Hans"

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from flores_lang_zh import flores_code_to_zh_name  # noqa: E402

# Common Traditional-only characters. This intentionally favors precision over
# completeness; any hit rejects the text because the requested corpus is zho_Hans.
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


def has_traditional_chars(text: str) -> bool:
    return any(ch in TRADITIONAL_CHARS for ch in text)


def is_simplified_chinese(text: str, *, min_cjk_ratio: float) -> bool:
    if has_traditional_chars(text):
        return False
    letters = [ch for ch in text if not ch.isspace()]
    if not letters:
        return False
    return (cjk_count(text) / max(1, len(letters))) >= min_cjk_ratio


def is_thai(text: str, *, min_thai_ratio: float) -> bool:
    letters = [ch for ch in text if not ch.isspace()]
    if not letters:
        return False
    return (thai_count(text) / max(1, len(letters))) >= min_thai_ratio


def split_candidates(text: str, *, max_chars: int) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Prefer paragraph-ish chunks first, then sentence punctuation.
    rough = re.split(r"(?:\\n+|[。！？!?；;])", text)
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


def iter_clean_sources(
    path: Path,
    lang: str,
    *,
    min_chars: int,
    max_chars: int,
    min_thai_ratio: float,
    min_cjk_ratio: float,
) -> Iterator[str]:
    seen: set[str] = set()
    for row in read_jsonl(path):
        raw = row.get("text")
        if not isinstance(raw, str):
            continue
        for cand in split_candidates(raw, max_chars=max_chars):
            if len(cand) < min_chars:
                continue
            if lang == THA:
                if not is_thai(cand, min_thai_ratio=min_thai_ratio):
                    continue
            elif lang == ZHO:
                if not is_simplified_chinese(cand, min_cjk_ratio=min_cjk_ratio):
                    continue
            key = sha1_text(cand)
            if key in seen:
                continue
            seen.add(key)
            yield cand
            break


def make_instruction(src: str, tgt: str) -> str:
    return (
        f"请将以下 {flores_code_to_zh_name(src)} 文本翻译为 "
        f"{flores_code_to_zh_name(tgt)}，只输出译文。"
    )


def mt_user_content(src_lang: str, tgt_lang: str, text: str) -> str:
    if tgt_lang == ZHO:
        return (
            "Translate the following Thai text into Simplified Chinese. "
            "Output only the translation. Do not use Traditional Chinese.\n\n"
            f"{text}"
        )
    if tgt_lang == THA:
        return (
            "Translate the following Simplified Chinese text into Thai. "
            "Output only the Thai translation, no explanations.\n\n"
            f"{text}"
        )
    return f"Translate from {src_lang} to {tgt_lang}. Output only the translation.\n\n{text}"


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
    extra = build_extra_body(model_family)
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
    if extra:
        kwargs["extra_body"] = extra

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # OpenAI SDK exposes several transport exceptions.
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


def refresh_preview(out_path: Path) -> None:
    prev = preview_path(out_path)
    rows: list[dict[str, Any]] = []
    if out_path.is_file():
        for row in read_jsonl(out_path):
            rows.append(row)
            if len(rows) >= PREVIEW_N:
                break
    with open(prev, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def valid_translation(src_lang: str, tgt_lang: str, out: str, *, min_thai_ratio: float, min_cjk_ratio: float) -> bool:
    out = out.strip()
    if not out:
        return False
    if tgt_lang == ZHO:
        return is_simplified_chinese(out, min_cjk_ratio=min_cjk_ratio)
    if tgt_lang == THA:
        return is_thai(out, min_thai_ratio=min_thai_ratio)
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
    refresh_preview_every: int,
    min_chars: int,
    max_chars: int,
    min_thai_ratio: float,
    min_cjk_ratio: float,
    emit_reverse: bool,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reverse_path = out_path.parent / f"fineweb_synth_extra_reverse_{tgt_lang}__{src_lang}_from_{src_lang}__{tgt_lang}.jsonl"
    done_hashes = existing_source_hashes(out_path) if resume else set()
    existing_n = count_jsonl(out_path) if resume else 0
    target_new = max(0, limit - existing_n)
    if target_new == 0:
        refresh_preview(out_path)
        print(f"[{src_lang}->{tgt_lang}] already has {existing_n} rows; skip.")
        return existing_n

    mode = "a" if resume else "w"
    rev_mode = "a" if resume and reverse_path.is_file() else "w"
    written_new = 0
    skipped_bad = 0
    submitted = 0
    source_iter = iter_clean_sources(
        input_path,
        src_lang,
        min_chars=min_chars,
        max_chars=max_chars,
        min_thai_ratio=min_thai_ratio,
        min_cjk_ratio=min_cjk_ratio,
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
        rev_file = open(reverse_path, rev_mode, encoding="utf-8") if emit_reverse else None
        try:
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
                    if not valid_translation(
                        src_lang,
                        tgt_lang,
                        translated,
                        min_thai_ratio=min_thai_ratio,
                        min_cjk_ratio=min_cjk_ratio,
                    ):
                        skipped_bad += 1
                        # Replace invalid output with another source sample if possible.
                        submitted -= 1
                        continue

                    row = {
                        "instruction": make_instruction(src_lang, tgt_lang),
                        "input": src_text,
                        "output": translated,
                        "meta": {
                            "source": "fineweb2_synthetic_qwen3.5_27b",
                            "source_file": str(input_path),
                            "source_hash": source_hash,
                            "generator": served_model_name,
                            "api_base": base_url,
                        },
                    }
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_new += 1
                    pbar.update(1)

                    if rev_file is not None:
                        rev = {
                            "instruction": make_instruction(tgt_lang, src_lang),
                            "input": translated,
                            "output": src_text,
                            "meta": {
                                **row["meta"],
                                "source_hash": sha1_text(translated),
                                "reversed_from": source_hash,
                            },
                        }
                        rev_file.write(json.dumps(rev, ensure_ascii=False) + "\n")

                    if written_new % refresh_preview_every == 0:
                        fo.flush()
                        if rev_file is not None:
                            rev_file.flush()
                        refresh_preview(out_path)
                        if rev_file is not None:
                            refresh_preview(reverse_path)

                fill()
            pbar.close()
        finally:
            if rev_file is not None:
                rev_file.close()

    refresh_preview(out_path)
    if emit_reverse:
        refresh_preview(reverse_path)
    total = count_jsonl(out_path)
    print(
        f"[{src_lang}->{tgt_lang}] wrote {written_new} new rows, total {total}; "
        f"invalid translations skipped {skipped_bad}; preview {preview_path(out_path)}"
    )
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Thai<->Simplified-Chinese synthetic MT pairs from FineWeb-2.")
    ap.add_argument(
        "--thai-input",
        type=Path,
        default=root() / "training" / "data" / "monolingual" / "fineweb2_pt_tha_Thai.jsonl",
    )
    ap.add_argument(
        "--zho-input",
        type=Path,
        default=root() / "training" / "data" / "monolingual" / "fineweb2_pt_zho_Hans.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root() / "training" / "data" / "multilingual" / "fineweb2_synth",
    )
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
    ap.add_argument("--min-cjk-ratio", type=float, default=0.45)
    ap.add_argument("--refresh-preview-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--emit-reverse",
        action="store_true",
        help="Also write reversed examples from the generated pairs. This doubles training rows.",
    )
    args = ap.parse_args()

    if not args.thai_input.is_file():
        raise SystemExit(f"Thai input not found: {args.thai_input}")
    if not args.zho_input.is_file():
        raise SystemExit(f"Simplified Chinese input not found: {args.zho_input}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_direction(
        client=client,
        served_model_name=args.served_model_name,
        base_url=args.base_url,
        input_path=args.thai_input,
        out_path=out_dir / f"fineweb_synth_{THA}__{ZHO}.jsonl",
        src_lang=THA,
        tgt_lang=ZHO,
        limit=int(args.limit_per_direction),
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
        emit_reverse=bool(args.emit_reverse),
    )
    generate_direction(
        client=client,
        served_model_name=args.served_model_name,
        base_url=args.base_url,
        input_path=args.zho_input,
        out_path=out_dir / f"fineweb_synth_{ZHO}__{THA}.jsonl",
        src_lang=ZHO,
        tgt_lang=THA,
        limit=int(args.limit_per_direction),
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
        emit_reverse=bool(args.emit_reverse),
    )
    print(f"Done. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
