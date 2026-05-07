#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


LATIN_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
THAI_SEQ_RE = re.compile(r"[\u0e00-\u0e7f]+")
PREVIEW_N = 50
PROJECT_LANGS = ("eng_Latn", "zho_Hans", "spa_Latn", "ind_Latn", "vie_Latn", "tha_Thai", "tgl_Latn")

STOPWORDS: dict[str, set[str]] = {
    "eng_Latn": {
        "the", "and", "that", "have", "with", "this", "from", "they", "were", "been", "their", "there",
        "would", "could", "should", "about", "after", "before", "again", "often", "common", "words",
        "once", "only", "small", "note", "comment", "claim",
    },
    "spa_Latn": {
        "para", "como", "pero", "esta", "este", "estos", "estas", "porque", "sobre", "entre", "desde",
        "cuando", "tambien", "también", "donde", "mientras",
    },
    "ind_Latn": {
        "yang", "dan", "dari", "untuk", "dengan", "pada", "dalam", "atau", "karena", "sebagai", "akan",
        "tidak", "sudah", "adalah",
    },
    "vie_Latn": {
        "của", "cho", "với", "trong", "được", "không", "một", "những", "người", "này", "các", "vào",
        "trên", "khi", "đến",
    },
    "tgl_Latn": {
        "ang", "mga", "para", "nang", "kung", "hindi", "isang", "mula", "dahil", "bilang", "kaniyang",
        "kanilang", "tungkol",
    },
    "zho_Hans": {
        "一个", "一种", "这个", "这些", "他们", "我们", "你们", "因为", "所以", "但是", "如果", "可以", "进行",
        "没有", "不是", "已经",
    },
    "tha_Thai": {
        "และ", "หรือ", "เป็น", "อยู่", "ของ", "การ", "ความ", "ที่", "ใน", "จาก", "ด้วย", "สำหรับ",
    },
}


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_text(row: dict[str, Any], text_keys: list[str], min_chars: int) -> str | None:
    for key in text_keys:
        value = row.get(key)
        if isinstance(value, str) and len(value.strip()) >= min_chars:
            return value.strip()
    fallback = ""
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) > len(fallback):
            fallback = value.strip()
    return fallback if len(fallback) >= min_chars else None


def default_min_token_len(lang: str) -> int:
    if lang == "zho_Hans":
        return 2
    if lang == "tha_Thai":
        return 2
    return 4


def default_max_token_len(lang: str) -> int:
    if lang == "zho_Hans":
        return 4
    if lang == "tha_Thai":
        return 30
    return 40


def normalize_token(token: str, lang: str) -> str:
    token = unicodedata.normalize("NFKC", token.strip()).casefold()
    if lang == "zho_Hans":
        token = "".join(CJK_CHAR_RE.findall(token))
    return token.strip("-'’‘\"“”.,;:!?()[]{}<>")


def cjk_ngrams(text: str, *, min_token_len: int, max_token_len: int) -> list[str]:
    chars = CJK_CHAR_RE.findall(text)
    toks: list[str] = []
    upper = max(min(max_token_len, 6), min_token_len)
    for n in range(min_token_len, upper + 1):
        for i in range(0, max(0, len(chars) - n + 1)):
            toks.append("".join(chars[i : i + n]))
    return toks


def thai_tokens(text: str) -> list[str]:
    try:
        from pythainlp.tokenize import word_tokenize

        return [tok.strip() for tok in word_tokenize(text, engine="newmm", keep_whitespace=False) if tok.strip()]
    except Exception:
        return THAI_SEQ_RE.findall(text)


def is_noise_token(token: str, *, lang: str, min_token_len: int, max_token_len: int) -> bool:
    if not token or len(token) < min_token_len or len(token) > max_token_len:
        return True
    if token in STOPWORDS.get(lang, set()):
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if "http" in token or "www" in token or "@" in token:
        return True
    if len(set(token)) <= 1:
        return True
    if lang != "zho_Hans" and lang != "tha_Thai":
        alpha = sum(1 for ch in token if ch.isalpha())
        if alpha / max(1, len(token)) < 0.75:
            return True
    if lang == "zho_Hans" and not CJK_CHAR_RE.search(token):
        return True
    return False


def tokenize(text: str, *, lang: str, min_token_len: int, max_token_len: int) -> list[str]:
    toks: list[str] = []
    if lang == "zho_Hans":
        raw_tokens = cjk_ngrams(text, min_token_len=min_token_len, max_token_len=max_token_len)
    elif lang == "tha_Thai":
        raw_tokens = thai_tokens(text)
    else:
        raw_tokens = [match.group(0) for match in LATIN_TOKEN_RE.finditer(text)]

    for raw in raw_tokens:
        tok = normalize_token(raw, lang)
        if not is_noise_token(tok, lang=lang, min_token_len=min_token_len, max_token_len=max_token_len):
            toks.append(tok)
    return toks


def context_for_term(text: str, term: str, window: int, *, lang: str) -> str | None:
    if lang == "zho_Hans" or lang == "tha_Thai":
        match = re.search(re.escape(term), text, flags=re.IGNORECASE)
    else:
        match = re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def build_prompt(*, term: str, lang: str, frequency: int, contexts: list[str]) -> list[dict[str, str]]:
    ctx = "\n".join(f"- {c}" for c in contexts)
    return [
        {
            "role": "system",
            "content": (
                "You explain rare corpus terms for a dictionary-like language learning tool. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language code: {lang}\n"
                f"Term: {term}\n"
                f"FineWeb frequency in sampled corpus: {frequency}\n"
                f"Contexts:\n{ctx}\n\n"
                "Return valid JSON with key `senses` only. `senses` must be an array of 1 to 4 sense objects. "
                "Each sense object must contain: definition, usage_note, example_sentence. "
                "Each distinct meaning or major usage should get its own sense. "
                "Each example_sentence must be one short natural sentence in the same language as the term "
                "and must clearly demonstrate that specific sense."
            ),
        },
    ]


def parse_llm_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM response is not a JSON object")
    return obj


def normalize_senses(parsed: dict[str, Any]) -> list[dict[str, str]]:
    raw_senses = parsed.get("senses")
    senses: list[dict[str, str]] = []
    if isinstance(raw_senses, list):
        for item in raw_senses:
            if not isinstance(item, dict):
                continue
            definition = str(item.get("definition", "")).strip()
            usage_note = str(item.get("usage_note", "")).strip()
            example_sentence = str(item.get("example_sentence", "")).strip()
            if definition or usage_note or example_sentence:
                senses.append(
                    {
                        "definition": definition,
                        "usage_note": usage_note,
                        "example_sentence": example_sentence,
                    }
                )

    if senses:
        return senses[:4]

    # Backward-compatible fallback for older model outputs.
    definition = str(parsed.get("definition", "")).strip()
    usage_note = str(parsed.get("usage_note", "")).strip()
    examples = parsed.get("example_sentences", [])
    example = ""
    if isinstance(examples, list):
        example = next((str(x).strip() for x in examples if isinstance(x, str) and x.strip()), "")
    if definition or usage_note or example:
        return [{"definition": definition, "usage_note": usage_note, "example_sentence": example}]
    return []


def explain_terms(
    rows: list[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    resume: bool,
) -> list[dict[str, Any]]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        if resume and row.get("definition"):
            enriched.append(row)
            continue
        req = {
            "model": model,
            "messages": build_prompt(
                term=str(row["term"]),
                lang=str(row["lang"]),
                frequency=int(row["frequency"]),
                contexts=[str(x) for x in row.get("contexts", [])],
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = client.chat.completions.create(**req, timeout=request_timeout)
            content = resp.choices[0].message.content or ""
            parsed = parse_llm_json(content)
            senses = normalize_senses(parsed)
            row = {
                **row,
                "senses": senses,
                "definition": senses[0]["definition"] if senses else "",
                "usage_note": senses[0]["usage_note"] if senses else "",
                "example_sentences": [s["example_sentence"] for s in senses if s.get("example_sentence")],
                "explanation_model": model,
            }
        except Exception as e:
            content = locals().get("content", "")
            row = {**row, "explanation_error": f"{type(e).__name__}: {e}", "raw_model_output": content}
        enriched.append(row)
    return enriched


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_for_lang(args: argparse.Namespace, lang: str, inputs: list[Path]) -> dict[str, Any]:
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        message = f"[{lang}] Missing input JSONL: {', '.join(missing)}"
        if args.skip_missing:
            print(f"WARNING: {message}; skipped.", file=sys.stderr)
            return {"lang": lang, "skipped": True, "missing": missing}
        raise SystemExit(message)

    text_keys = args.text_keys or ["text", "content", "raw_content"]
    min_token_len = args.min_token_len or default_min_token_len(lang)
    max_token_len = args.max_token_len or default_max_token_len(lang)
    term_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    docs_seen = 0
    for path in inputs:
        for row in read_jsonl(path):
            text = resolve_text(row, text_keys, args.min_chars)
            if not text:
                continue
            toks = tokenize(text, lang=lang, min_token_len=min_token_len, max_token_len=max_token_len)
            if not toks:
                continue
            term_counts.update(toks)
            doc_counts.update(set(toks))
            docs_seen += 1
            if args.max_docs > 0 and docs_seen >= args.max_docs:
                break
        if args.max_docs > 0 and docs_seen >= args.max_docs:
            break

    candidates = [
        term
        for term, count in term_counts.items()
        if args.min_count <= count <= args.max_count
    ]
    candidates.sort(key=lambda t: (term_counts[t], doc_counts[t], -len(t), t))
    selected = candidates[: args.top_n]
    selected_set = set(selected)

    contexts: dict[str, list[str]] = defaultdict(list)
    for path in inputs:
        for row in read_jsonl(path):
            text = resolve_text(row, text_keys, args.min_chars)
            if not text:
                continue
            present = set(tokenize(text, lang=lang, min_token_len=min_token_len, max_token_len=max_token_len)) & selected_set
            for term in sorted(present):
                if len(contexts[term]) >= args.contexts_per_term:
                    continue
                ctx = context_for_term(text, term, args.context_window, lang=lang)
                if ctx:
                    contexts[term].append(ctx)
            if all(len(contexts[t]) >= args.contexts_per_term for t in selected):
                break

    now = int(time.time())
    rows = [
        {
            "source": "fineweb_frequency",
            "lang": lang,
            "term": term,
            "frequency": int(term_counts[term]),
            "document_frequency": int(doc_counts[term]),
            "contexts": contexts.get(term, []),
            "senses": [],
            "definition": "",
            "usage_note": "",
            "example_sentences": [],
            "created_at": now,
        }
        for term in selected
    ]

    out_path = args.out_dir / f"rare_terms_{lang}.jsonl"
    if args.generate_explanations:
        if not args.api_key:
            raise SystemExit("--generate-explanations requires --api-key or RARE_TERM_API_KEY/QWEN_API_KEY")
        if args.resume and out_path.is_file():
            existing_by_term = {r.get("term"): r for r in read_jsonl(out_path)}
            rows = [existing_by_term.get(r["term"], r) for r in rows]
        rows = explain_terms(
            rows,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            resume=args.resume,
        )

    write_jsonl(out_path, rows)
    preview_path = args.out_dir / "previews" / f"rare_terms_{lang}.preview_{PREVIEW_N}.jsonl"
    write_jsonl(preview_path, rows[:PREVIEW_N])
    stats = {
        "lang": lang,
        "inputs": [str(p) for p in inputs],
        "docs_seen": docs_seen,
        "unique_terms": len(term_counts),
        "selected_terms": len(rows),
        "min_count": args.min_count,
        "max_count": args.max_count,
        "top_n": args.top_n,
        "min_token_len": min_token_len,
        "max_token_len": max_token_len,
        "output": str(out_path),
        "preview": str(preview_path),
    }
    stats_path = args.out_dir / f"rare_terms_{lang}.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"stats -> {stats_path}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FineWeb rare-term explanation data for dictionary_tool.")
    ap.add_argument(
        "--lang",
        action="append",
        dest="langs",
        help="Language code. Repeatable. Default: eng_Latn unless --all-langs is set.",
    )
    ap.add_argument("--all-langs", action="store_true", help=f"Process all project languages: {', '.join(PROJECT_LANGS)}")
    ap.add_argument(
        "--input",
        action="append",
        type=Path,
        dest="inputs",
        help="FineWeb monolingual JSONL path. Repeatable. Only valid with a single --lang.",
    )
    ap.add_argument("--text-key", action="append", dest="text_keys", default=None)
    ap.add_argument("--out-dir", type=Path, default=root() / "training" / "data" / "dictionaries" / "fineweb_rare_terms")
    ap.add_argument("--max-docs", type=int, default=0, help="0 means unlimited.")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--min-token-len", type=int, default=0, help="0 means language-specific default.")
    ap.add_argument("--max-token-len", type=int, default=0, help="0 means language-specific default.")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--max-count", type=int, default=2)
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--contexts-per-term", type=int, default=3)
    ap.add_argument("--context-window", type=int, default=160)
    ap.add_argument("--skip-missing", action="store_true", help="Skip missing language input files instead of failing.")
    ap.add_argument("--generate-explanations", action="store_true")
    ap.add_argument("--model", default=os.environ.get("RARE_TERM_MODEL", os.environ.get("QWEN_MODEL", "qwen3.6-max-preview")))
    ap.add_argument("--base-url", default=os.environ.get("RARE_TERM_API_BASE", os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")))
    ap.add_argument("--api-key", default=os.environ.get("RARE_TERM_API_KEY", os.environ.get("QWEN_API_KEY", "")))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--request-timeout", type=float, default=60.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.all_langs:
        langs = list(PROJECT_LANGS)
    else:
        langs = sorted(set(args.langs or ["eng_Latn"]))

    if args.inputs and len(langs) != 1:
        raise SystemExit("--input can only be used with exactly one language. Omit --input for multi-language defaults.")

    stats_rows: list[dict[str, Any]] = []
    for lang in langs:
        inputs = args.inputs or [root() / "training" / "data" / "monolingual" / f"fineweb2_pt_{lang}.jsonl"]
        stats_rows.append(build_for_lang(args, lang, inputs))

    manifest_path = args.out_dir / "rare_terms_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"languages": langs, "runs": stats_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
