from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from core import DICTIONARY_SRC_LANG, DICTIONARY_TGT_LANG, get_index


FINAL_TRANSLATION_TOOL_NAME = "final_translation"
_OMITTED_OUTPUT_KEYS = {"license_note", "source_url"}


def _index(lexicon_dir: Path):
    return get_index(str(lexicon_dir))


def list_dictionary_pairs(*, lexicon_dir: Path) -> list[dict[str, Any]]:
    return _index(lexicon_dir).list_pairs()


def strip_omitted_output_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_omitted_output_fields(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_omitted_output_fields(item)
            for key, item in value.items()
            if key not in _OMITTED_OUTPUT_KEYS
        }
    return value


def lookup_dictionary(
    *,
    lexicon_dir: Path,
    term: str,
    top_k: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """
    越南语 → 英语：在 lexicon（dict_terms_vie_Latn__eng_Latn.jsonl）中查词。

    按“包含关系（含 exact）”匹配 source_text（越南语词/短语），支持 offset 分页。
    """
    idx = _index(lexicon_dir)
    primary = idx.lookup(
        src_lang=DICTIONARY_SRC_LANG,
        tgt_lang=DICTIONARY_TGT_LANG,
        term=term,
        top_k=top_k,
        offset=offset,
    )
    return strip_omitted_output_fields(primary)


def build_final_translation_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": FINAL_TRANSLATION_TOOL_NAME,
            "description": (
                "Submit the final translation after using dictionary evidence when useful. "
                "Call this exactly once when the final answer is ready."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "translation": {
                        "type": "string",
                        "description": "The final translated text only, with no analysis or tool-call markup.",
                    },
                },
                "required": ["translation"],
            },
        },
    }


def build_final_translation_tools() -> list[dict[str, Any]]:
    return [build_final_translation_tool()]


def build_openai_tools(*, dictionary_hint: str | None = None) -> list[dict[str, Any]]:
    hint_suffix = f" {dictionary_hint}" if dictionary_hint else ""
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_dictionary",
                "description": (
                    "Vietnamese (vie_Latn) to English (eng_Latn) dictionary lookup only. "
                    "Matching is contains-based on Vietnamese headwords/phrases (including exact). "
                    "Use pagination via offset when many rows match."
                    f"{hint_suffix}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string", "description": "Vietnamese word or substring to search."},
                        "top_k": {"type": "integer", "default": 20},
                        "offset": {"type": "integer", "default": 0},
                    },
                    "required": ["term"],
                },
            },
        },
        build_final_translation_tool(),
    ]


def build_local_dispatcher(
    *,
    lexicon_dir: Path,
) -> dict[str, Callable[..., Any]]:
    return {
        "lookup_dictionary": lambda **kwargs: lookup_dictionary(
            lexicon_dir=lexicon_dir,
            term=str(kwargs["term"]),
            top_k=int(kwargs.get("top_k", 20)),
            offset=int(kwargs.get("offset", 0)),
        ),
        "list_dictionary_pairs": lambda **kwargs: list_dictionary_pairs(lexicon_dir=lexicon_dir),
    }


def env_lexicon_dir() -> Path:
    p = os.environ.get("DICTIONARY_TOOL_LEXICON_DIR")
    if p:
        return Path(p).resolve()
    return (Path(__file__).resolve().parents[1] / "training" / "data" / "dictionaries" / "moe_lexicon").resolve()
