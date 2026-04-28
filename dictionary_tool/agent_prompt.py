from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from llm import lang_name

def build_translation_agent_prompt(*, supported_pairs_hint: str | None = None) -> str:
    pairs_line = (
        f"\nSupported dictionary language pairs: {supported_pairs_hint}\n"
        if supported_pairs_hint
        else ""
    )
    return f"""You are a translation agent with access to callable dictionary tools.

Your job is to produce the best possible translation from src_lang to tgt_lang.
{pairs_line}

Tool-use policy:
1. If the input is a single term, short phrase, named entity, or terminology-heavy snippet, call `lookup_dictionary` first.
2. If the input is a sentence or paragraph and contains potentially important terminology, call `lookup_dictionary` for a few key terms before translation.
3. Then produce the final translation yourself based on source text and any dictionary evidence.
4. Never invent dictionary results. Use tool outputs as they are.
5. Prefer the exact src_lang and tgt_lang provided by the user when calling tools.
6. Return the final translation only unless the user explicitly asks for analysis.

Recommended workflow:
- Understand src_lang, tgt_lang, and text.
- Optionally call `lookup_dictionary(src_lang, tgt_lang, term, top_k, offset)` for key terms.
- If lookup has no direct hit, check its fallback_results (auto suggestions from other pairs/languages).
- Produce the translation directly in your final answer.
- Return only the final translation.
"""


def build_translation_messages(
    *, text: str, src_lang: str, tgt_lang: str, supported_pairs_hint: str | None = None
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
            ("system", build_translation_agent_prompt(supported_pairs_hint=supported_pairs_hint)),
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
