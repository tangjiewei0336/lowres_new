from agent_prompt import build_translation_agent_prompt


def test_agent_prompt_mentions_dictionary_first_for_terms() -> None:
    prompt = build_translation_agent_prompt()
    assert "lookup_dictionary" in prompt
    assert "vie_Latn" in prompt
    assert "llm_translate_via_vllm" not in prompt
    assert "final_translation" in prompt
    assert "fallback_results" not in prompt
