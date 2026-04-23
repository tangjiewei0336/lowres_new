from __future__ import annotations

from llm import build_translation_request, translation_instruction


def test_translation_instruction_english() -> None:
    inst = translation_instruction("spa_Latn", "zho_Hans")
    assert inst.startswith("Translate the following Spanish text into Simplified Chinese.")
    assert "Traditional Chinese" in inst


def test_build_translation_request_qwen_disables_thinking() -> None:
    req = build_translation_request(
        model="qwen3-8b",
        src_lang="eng_Latn",
        tgt_lang="vie_Latn",
        text="hello",
        max_tokens=128,
        temperature=0.0,
        model_family="qwen3",
    )
    assert req["model"] == "qwen3-8b"
    assert req["messages"][0]["role"] == "system"
    assert req["messages"][1]["content"].endswith("\n\nhello")
    assert req["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_build_translation_request_generic_has_no_extra_body() -> None:
    req = build_translation_request(
        model="generic-model",
        src_lang="eng_Latn",
        tgt_lang="spa_Latn",
        text="hello",
        max_tokens=64,
        temperature=0.0,
        model_family="generic",
    )
    assert "extra_body" not in req
