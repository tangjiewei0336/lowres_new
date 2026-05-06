from agent_runtime import extract_final_translation_from_text, looks_like_tool_call_text


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


def test_regular_translation_is_not_tool_call_text() -> None:
    assert looks_like_tool_call_text("周一，科学家宣布了新工具。") is False
