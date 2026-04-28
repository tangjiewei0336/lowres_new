from __future__ import annotations

from pathlib import Path

from tool_registry import lookup_dictionary


def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_lexicon"


def test_lookup_dictionary_with_fallback_from_other_sources() -> None:
    # "后" 在 sample 词典里主要出现在 zho_Hans 源语言下；
    # 对 eng_Latn->zho_Hans 查询应触发 fallback。
    out = lookup_dictionary(
        lexicon_dir=fixture_dir(),
        src_lang="eng_Latn",
        tgt_lang="zho_Hans",
        term="后",
        top_k=5,
        offset=0,
        fallback_top_k=3,
    )
    assert out["total_matches"] == 0
    assert out["fallback_used"] is True
    assert out["fallback_scope"] in {"same_src_other_targets", "other_sources"}
    assert out["fallback_results"]
