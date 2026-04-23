from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from core import get_index


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


if __name__ == "__main__":
    mcp.run()
