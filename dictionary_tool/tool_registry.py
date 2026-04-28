from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from core import get_index


def _index(lexicon_dir: Path):
    return get_index(str(lexicon_dir))


def list_dictionary_pairs(*, lexicon_dir: Path) -> list[dict[str, Any]]:
    return _index(lexicon_dir).list_pairs()


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
        return primary

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
        return primary

    # 同 src_lang 也没命中时，回退到其他 src_lang
    other_rows = idx.search_pairs(term=term, top_k=max(fallback_top_k * 6, fallback_top_k))
    other_src_recs = [row for row in other_rows.get("results", []) if row.get("src_lang") != src_lang][:fallback_top_k]
    primary["fallback_used"] = bool(other_src_recs)
    primary["fallback_scope"] = "other_sources"
    primary["fallback_results"] = other_src_recs
    return primary


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
    ]


def build_local_dispatcher(
    *,
    lexicon_dir: Path,
) -> dict[str, Callable[..., Any]]:
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
    }


def env_lexicon_dir() -> Path:
    p = os.environ.get("DICTIONARY_TOOL_LEXICON_DIR")
    if p:
        return Path(p).resolve()
    return (Path(__file__).resolve().parents[1] / "training" / "data" / "dictionaries" / "moe_lexicon").resolve()
