from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from llm import lang_name


def build_translation_agent_prompt(*, dictionary_hint: str | None = None) -> str:
    dict_line = f"\nDictionary tool note: {dictionary_hint}\n" if dictionary_hint else ""
    return f"""You are a translation agent with access to callable dictionary tools.

Your job is to produce the best possible translation from src_lang to tgt_lang.
{dict_line}

Tool-use policy:
1. If the source language is Vietnamese (vie_Latn) and the text is a single term, short phrase, named entity, or terminology-heavy snippet, call `lookup_dictionary` first (Vietnamese headword only in `term`).
2. If the input is a sentence or paragraph and contains potentially important Vietnamese terms, call `lookup_dictionary` for one or a few key Vietnamese substrings before translation.
3. Then produce the final translation yourself based on source text and any tool evidence.
4. Never invent tool results. Use tool outputs as they are.
5. The bilingual `lookup_dictionary` tool is fixed to Vietnamese (vie_Latn) → English (eng_Latn); do not pass language codes to it—it only accepts `term`, `top_k`, and `offset`.
6. The final answer MUST be submitted by calling the `final_translation` tool.
7. Do NOT output the final translation as normal assistant text.
8. Do NOT print `<tool_call>...</tool_call>`, JSON tool-call markup, or `final_translation(...)` as text.
9. The `translation` argument must contain the final translation only, with no analysis or tool-call markup.

Recommended workflow:
- Understand src_lang, tgt_lang, and text.
- Optionally call `lookup_dictionary(term, top_k, offset)` for Vietnamese headwords or substrings when src is Vietnamese.
- When ready, call the `final_translation` tool with `translation` set to the final translated text.
- After calling `final_translation`, do not output any additional text.
"""


def build_translation_messages(
    *, text: str, src_lang: str, tgt_lang: str, dictionary_hint: str | None = None
) -> list[dict[str, Any]]:
    """使用 LangChain 的提示词模板构造 OpenAI chat messages。"""
    src_label = f"{lang_name(src_lang)} ({src_lang})"
    tgt_label = f"{lang_name(tgt_lang)} ({tgt_lang})"
    user_text = (
        f"Source language: {src_label}\n"
        f"Target language: {tgt_label}\n"
        f"text={text}\n\n"
        "Translate this text. Use dictionary tools when useful."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_translation_agent_prompt(dictionary_hint=dictionary_hint)),
            ("human", "{user_text}"),
        ]
    )
    rendered = prompt.invoke({"user_text": user_text})
    return [
        {
            "role": ("assistant" if m.type == "ai" else "user" if m.type == "human" else "system"),
            "content": str(m.content),
        }
        for m in rendered.messages
    ]
