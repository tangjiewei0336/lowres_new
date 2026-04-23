from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


LANG_EN = {
    "eng_Latn": "English",
    "zho_Hans": "Simplified Chinese",
    "zho_Hant": "Traditional Chinese",
    "spa_Latn": "Spanish",
    "ind_Latn": "Indonesian",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "tgl_Latn": "Tagalog",
}


def lang_name(code: str) -> str:
    code = (code or "").strip()
    return LANG_EN.get(code, code)


def translation_instruction(src_lang: str, tgt_lang: str) -> str:
    src = lang_name(src_lang)
    tgt = lang_name(tgt_lang)
    extra = " Do not use Traditional Chinese." if tgt_lang == "zho_Hans" else ""
    return f"Translate the following {src} text into {tgt}. Output only the translation.{extra}"


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    fam = (model_family or "").lower()
    if fam in ("qwen3", "qwen", "qwen3-4b", "qwen3-8b", "qwen3.5", "qwen3-5", "qwen35"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def build_translation_request(
    *,
    model: str,
    src_lang: str,
    tgt_lang: str,
    text: str,
    max_tokens: int,
    temperature: float,
    model_family: str,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional machine translation engine."},
            {"role": "user", "content": f"{translation_instruction(src_lang, tgt_lang)}\n\n{text}"},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    extra = build_extra_body(model_family)
    if extra:
        req["extra_body"] = extra
    return req


def create_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )


def translate_with_vllm(
    *,
    text: str,
    src_lang: str,
    tgt_lang: str,
    model: str,
    model_family: str = "qwen3",
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    client = create_client(base_url=base_url, api_key=api_key)
    req = build_translation_request(
        model=model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        text=text,
        max_tokens=max_tokens,
        temperature=temperature,
        model_family=model_family,
    )
    resp = client.chat.completions.create(**req)
    out = (resp.choices[0].message.content or "").strip()
    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model": model,
        "base_url": base_url or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
        "instruction": translation_instruction(src_lang, tgt_lang),
        "input": text,
        "output": out,
    }
