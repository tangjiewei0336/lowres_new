from __future__ import annotations

from pathlib import Path

from tool_registry import (
    FINAL_TRANSLATION_TOOL_NAME,
    build_final_translation_tools,
    build_local_dispatcher,
    build_openai_tools,
    list_fineweb_rare_term_languages,
    lookup_fineweb_rare_term,
    lookup_dictionary,
)


def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_lexicon"


def rare_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_rare_terms"


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


def test_lookup_dictionary_omits_source_url_and_license_note() -> None:
    out = lookup_dictionary(
        lexicon_dir=fixture_dir(),
        src_lang="eng_Latn",
        tgt_lang="zho_Hans",
        term="dictionary",
        top_k=5,
        offset=0,
        fallback_top_k=3,
    )
    assert out["results"]
    assert "source_url" not in out["results"][0]
    assert "license_note" not in out["results"][0]


def test_openai_tools_include_final_translation_tool() -> None:
    tools = build_openai_tools()
    names = [tool["function"]["name"] for tool in tools]
    assert "lookup_dictionary" in names
    assert "lookup_fineweb_rare_term" in names
    assert FINAL_TRANSLATION_TOOL_NAME in names


def test_final_translation_tools_only_include_final_tool() -> None:
    tools = build_final_translation_tools()
    assert [tool["function"]["name"] for tool in tools] == [FINAL_TRANSLATION_TOOL_NAME]


def test_lookup_fineweb_rare_term() -> None:
    out = lookup_fineweb_rare_term(
        rare_terms_dir=rare_fixture_dir(),
        lang="eng_Latn",
        term="quokkaesque",
        top_k=5,
        offset=0,
    )
    assert out["total_matches"] == 1
    assert out["results"][0]["definition"].startswith("Having a cheerful")


def test_list_fineweb_rare_term_languages() -> None:
    out = list_fineweb_rare_term_languages(rare_terms_dir=rare_fixture_dir())
    assert {"lang": "eng_Latn", "size": 2} in out


def test_dispatcher_includes_fineweb_rare_term_tool() -> None:
    dispatcher = build_local_dispatcher(lexicon_dir=fixture_dir(), rare_terms_dir=rare_fixture_dir())
    out = dispatcher["lookup_fineweb_rare_term"](lang="eng_Latn", term="ultracrepidarian")
    assert out["results"][0]["term"] == "ultracrepidarian"
