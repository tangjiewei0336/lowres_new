from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from core import get_index
from rare_terms import get_rare_term_index


FINAL_TRANSLATION_TOOL_NAME = "final_translation"
_OMITTED_OUTPUT_KEYS = {"license_note", "source_url"}


def _index(lexicon_dir: Path):
    return get_index(str(lexicon_dir))


def list_dictionary_pairs(*, lexicon_dir: Path) -> list[dict[str, Any]]:
    return _index(lexicon_dir).list_pairs()


def _rare_index(rare_terms_dir: Path):
    return get_rare_term_index(str(rare_terms_dir))


def list_fineweb_rare_term_languages(*, rare_terms_dir: Path) -> list[dict[str, Any]]:
    return _rare_index(rare_terms_dir).list_languages()


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
    src_lang: str,
    tgt_lang: str,
    term: str,
    top_k: int = 20,
    offset: int = 0,
    fallback_top_k: int = 10,
) -> dict[str, Any]:
    """
    定点查询：在一个固定语向(src_lang -> tgt_lang)内查词。

    适用场景:
    - 已经明确翻译方向时，优先使用本函数获取高相关词条。
    - 本函数按“包含关系（含 exact）”检索，可配合 offset 分页拉取全部结果。
    - 若当前 src_lang->tgt_lang 无结果，会自动给出回退建议：
      1) 同 src_lang 的其它 tgt_lang 结果
      2) 其它 src_lang 的结果
    """
    idx = _index(lexicon_dir)
    primary = idx.lookup(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        term=term,
        top_k=top_k,
        offset=offset,
    )
    if int(primary.get("total_matches", 0)) > 0:
        primary["fallback_used"] = False
        return strip_omitted_output_fields(primary)

    # 先尝试同 src_lang（跨目标语）推荐
    same_src_rows = idx.search_pairs(term=term, src_lang=src_lang, top_k=max(fallback_top_k * 3, fallback_top_k))
    same_src_recs = [
        row
        for row in same_src_rows.get("results", [])
        if not (row.get("src_lang") == src_lang and row.get("tgt_lang") == tgt_lang)
    ][:fallback_top_k]
    if same_src_recs:
        primary["fallback_used"] = True
        primary["fallback_scope"] = "same_src_other_targets"
        primary["fallback_results"] = same_src_recs
        return strip_omitted_output_fields(primary)

    # 同 src_lang 也没命中时，回退到其他 src_lang
    other_rows = idx.search_pairs(term=term, top_k=max(fallback_top_k * 6, fallback_top_k))
    other_src_recs = [row for row in other_rows.get("results", []) if row.get("src_lang") != src_lang][:fallback_top_k]
    primary["fallback_used"] = bool(other_src_recs)
    primary["fallback_scope"] = "other_sources"
    primary["fallback_results"] = other_src_recs
    return strip_omitted_output_fields(primary)


def lookup_fineweb_rare_term(
    *,
    rare_terms_dir: Path,
    lang: str,
    term: str,
    top_k: int = 10,
    offset: int = 0,
) -> dict[str, Any]:
    return _rare_index(rare_terms_dir).lookup(
        lang=lang,
        term=term,
        top_k=top_k,
        offset=offset,
    )


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


def build_openai_tools(*, supported_pairs_hint: str | None = None) -> list[dict[str, Any]]:
    pair_suffix = f" Supported dictionary pairs: {supported_pairs_hint}" if supported_pairs_hint else ""
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_dictionary",
                "description": (
                    "Lookup entries in one fixed direction (src_lang -> tgt_lang). "
                    "Use when translation direction is already known. "
                    "Matching is contains-based (including exact), supports pagination with offset, "
                    "and auto-suggests fallback matches from other pairs when current direction has no hit."
                    f"{pair_suffix}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src_lang": {"type": "string"},
                        "tgt_lang": {"type": "string"},
                        "term": {"type": "string"},
                        "top_k": {"type": "integer", "default": 20},
                        "offset": {"type": "integer", "default": 0},
                        "fallback_top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["src_lang", "tgt_lang", "term"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_fineweb_rare_term",
                "description": (
                    "Lookup rare FineWeb corpus terms with frequency, definitions, example sentences, "
                    "and sampled contexts. Use this for uncommon words or phrases that may not be in "
                    "the bilingual dictionary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lang": {"type": "string"},
                        "term": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "offset": {"type": "integer", "default": 0},
                    },
                    "required": ["lang", "term"],
                },
            },
        },
        build_final_translation_tool(),
    ]


def build_local_dispatcher(
    *,
    lexicon_dir: Path,
    rare_terms_dir: Path | None = None,
) -> dict[str, Callable[..., Any]]:
    rare_dir = rare_terms_dir or env_rare_terms_dir()
    return {
        "lookup_dictionary": lambda **kwargs: lookup_dictionary(
            lexicon_dir=lexicon_dir,
            src_lang=str(kwargs["src_lang"]),
            tgt_lang=str(kwargs["tgt_lang"]),
            term=str(kwargs["term"]),
            top_k=int(kwargs.get("top_k", 20)),
            offset=int(kwargs.get("offset", 0)),
            fallback_top_k=int(kwargs.get("fallback_top_k", 10)),
        ),
        "list_dictionary_pairs": lambda **kwargs: list_dictionary_pairs(lexicon_dir=lexicon_dir),
        "lookup_fineweb_rare_term": lambda **kwargs: lookup_fineweb_rare_term(
            rare_terms_dir=rare_dir,
            lang=str(kwargs["lang"]),
            term=str(kwargs["term"]),
            top_k=int(kwargs.get("top_k", 10)),
            offset=int(kwargs.get("offset", 0)),
        ),
        "list_fineweb_rare_term_languages": lambda **kwargs: list_fineweb_rare_term_languages(
            rare_terms_dir=rare_dir,
        ),
    }


def env_lexicon_dir() -> Path:
    p = os.environ.get("DICTIONARY_TOOL_LEXICON_DIR")
    if p:
        return Path(p).resolve()
    return (Path(__file__).resolve().parents[1] / "training" / "data" / "dictionaries" / "moe_lexicon").resolve()


def env_rare_terms_dir() -> Path:
    p = os.environ.get("FINEWEB_RARE_TERMS_DIR")
    if p:
        return Path(p).resolve()
    return (Path(__file__).resolve().parents[1] / "training" / "data" / "dictionaries" / "fineweb_rare_terms").resolve()
