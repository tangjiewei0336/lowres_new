from agent_runtime import extract_final_translation_from_text, extract_text_tool_calls, looks_like_tool_call_text


def test_extract_final_translation_from_plain_tool_call_json() -> None:
    text = '{"name":"final_translation","arguments":{"translation":"周一，科学家宣布了新工具。"}}'
    assert extract_final_translation_from_text(text) == "周一，科学家宣布了新工具。"


def test_extract_final_translation_from_function_tool_call_json() -> None:
    text = (
        '<tool_call>{"function":{"name":"final_translation",'
        '"arguments":"{\\"translation\\": \\"idioma\\"}"}}</tool_call>'
    )
    assert extract_final_translation_from_text(text) == "idioma"


def test_lookup_dictionary_markup_is_not_final_text() -> None:
    text = '<tool_call>{"name":"lookup_dictionary","arguments":{"term":"language"}}</tool_call>'
    assert extract_final_translation_from_text(text) is None
    assert looks_like_tool_call_text(text) is True


def test_extract_final_translation_from_function_call_text() -> None:
    text = (
        "final_translation(translation=Participating countries hold art and education exhibitions "
        "in national pavilions, showcasing global issues.)"
    )
    assert extract_final_translation_from_text(text) == (
        "Participating countries hold art and education exhibitions in national pavilions, "
        "showcasing global issues."
    )


def test_extract_multiple_lookup_tool_call_blocks() -> None:
    text = (
        '<tool_call>\n{"name": "lookup_dictionary", "arguments": {"src_lang": "eng_Latn", '
        '"tgt_lang": "zho_Hans", "term": "success stories"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "lookup_dictionary", "arguments": {"src_lang": "eng_Latn", '
        '"tgt_lang": "zho_Hans", "term": "future"}}\n</tool_call>'
    )
    calls = extract_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["lookup_dictionary", "lookup_dictionary"]
    assert [c["arguments"]["term"] for c in calls] == ["success stories", "future"]


def test_regular_translation_is_not_tool_call_text() -> None:
    assert looks_like_tool_call_text("周一，科学家宣布了新工具。") is False
