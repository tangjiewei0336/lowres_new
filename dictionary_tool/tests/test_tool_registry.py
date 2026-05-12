from __future__ import annotations

from pathlib import Path

from tool_registry import (
    FINAL_TRANSLATION_TOOL_NAME,
    build_final_translation_tools,
    build_local_dispatcher,
    build_openai_tools,
    lookup_dictionary,
)


def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_lexicon"


def test_lookup_dictionary_vietnamese_english() -> None:
    out = lookup_dictionary(
        lexicon_dir=fixture_dir(),
        term="nước",
        top_k=5,
        offset=0,
    )
    assert out["src_lang"] == "vie_Latn"
    assert out["tgt_lang"] == "eng_Latn"
    assert out["total_matches"] >= 1
    assert out["results"][0]["target_text"] == "water"


def test_lookup_dictionary_zero_matches() -> None:
    out = lookup_dictionary(
        lexicon_dir=fixture_dir(),
        term="__no_such_vietnamese_term__",
        top_k=5,
        offset=0,
    )
    assert out["total_matches"] == 0
    assert out["results"] == []


def test_lookup_dictionary_omits_source_url_and_license_note() -> None:
    out = lookup_dictionary(
        lexicon_dir=fixture_dir(),
        term="đi",
        top_k=5,
        offset=0,
    )
    assert out["results"]
    assert "source_url" not in out["results"][0]
    assert "license_note" not in out["results"][0]
    assert out["results"][0].get("examples")


def test_openai_tools_include_final_translation_tool() -> None:
    tools = build_openai_tools()
    names = [tool["function"]["name"] for tool in tools]
    assert names == ["lookup_dictionary", FINAL_TRANSLATION_TOOL_NAME]
    dict_tool = next(t for t in tools if t["function"]["name"] == "lookup_dictionary")
    params = dict_tool["function"]["parameters"]["properties"]
    assert "term" in params
    assert "src_lang" not in params


def test_final_translation_tools_only_include_final_tool() -> None:
    tools = build_final_translation_tools()
    assert [tool["function"]["name"] for tool in tools] == [FINAL_TRANSLATION_TOOL_NAME]


def test_dispatcher_lookup_dictionary() -> None:
    dispatcher = build_local_dispatcher(lexicon_dir=fixture_dir())
    out = dispatcher["lookup_dictionary"](term="nhà")
    assert out["results"][0]["source_text"] == "nhà"
