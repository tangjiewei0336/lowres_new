from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from core import get_index
from llm import translate_with_vllm


LEXICON_DIR = os.environ.get("DICTIONARY_TOOL_LEXICON_DIR")
mcp = FastMCP("lowres-dictionary-tool")


def _index():
    return get_index(LEXICON_DIR)


@mcp.tool()
def list_dictionary_pairs() -> list[dict[str, Any]]:
    """List available dictionary language pairs and entry counts."""
    return _index().list_pairs()


@mcp.tool()
def lookup_dictionary(
    src_lang: str,
    tgt_lang: str,
    term: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Lookup dictionary entries for one language pair."""
    return _index().lookup(src_lang=src_lang, tgt_lang=tgt_lang, term=term, top_k=top_k)


@mcp.tool()
def search_dictionary_pairs(
    term: str,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search a term across multiple dictionary pairs."""
    return _index().search_pairs(term=term, src_lang=src_lang, tgt_lang=tgt_lang, top_k=top_k)


@mcp.tool()
def llm_translate_via_vllm(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model: str = "qwen3-8b",
    model_family: str = "qwen3",
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call the previously deployed vLLM OpenAI-compatible endpoint for translation."""
    return translate_with_vllm(
        text=text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        model=model,
        model_family=model_family,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )


if __name__ == "__main__":
    mcp.run()
