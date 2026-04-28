from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agent_prompt import build_translation_messages
from llm import build_extra_body, create_client
from tool_registry import build_local_dispatcher, build_openai_tools


class LangGraphDictionaryAgentRuntime:
    def __init__(
        self,
        *,
        model: str,
        model_family: str,
        base_url: str,
        api_key: str,
        lexicon_dir: Path,
        max_tokens: int = 512,
        temperature: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.model_family = model_family
        self.base_url = base_url
        self.api_key = api_key
        self.lexicon_dir = lexicon_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.debug = debug
        self._client = None
        self._tools = build_openai_tools()
        self._supported_pairs_hint: str | None = None
        self._dispatcher = None

    async def __aenter__(self) -> "LangGraphDictionaryAgentRuntime":
        self._client = create_client(base_url=self.base_url, api_key=self.api_key)
        self._dispatcher = build_local_dispatcher(
            lexicon_dir=self.lexicon_dir.resolve(),
        )
        pairs = self._dispatcher["list_dictionary_pairs"]()
        self._supported_pairs_hint = self._format_supported_pairs_hint(pairs)
        self._tools = build_openai_tools(supported_pairs_hint=self._supported_pairs_hint)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._client = None
        self._dispatcher = None
        self._supported_pairs_hint = None

    @staticmethod
    def _format_supported_pairs_hint(pairs: list[dict[str, Any]], max_items: int = 30) -> str:
        formatted = [
            f"{p.get('src_lang', '?')}->{p.get('tgt_lang', '?')}"
            for p in pairs
            if p.get("src_lang") and p.get("tgt_lang")
        ]
        if not formatted:
            return "none"
        head = formatted[:max_items]
        if len(formatted) > max_items:
            head.append(f"... (+{len(formatted) - max_items} more)")
        return ", ".join(head)

    def _build_messages(self, *, text: str, src_lang: str, tgt_lang: str) -> list[dict[str, Any]]:
        messages = build_translation_messages(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            supported_pairs_hint=self._supported_pairs_hint,
        )
        if self.debug:
            prompt_dump = "\n\n".join(
                f"[{m.get('role', 'unknown')}]\n{m.get('content', '')}" for m in messages
            )
            print("=== DEBUG PROMPT START ===", file=sys.stderr)
            print(prompt_dump, file=sys.stderr)
            print("=== DEBUG PROMPT END ===", file=sys.stderr)
        return messages

    async def translate(self, *, text: str, src_lang: str, tgt_lang: str) -> str:
        if self._client is None or self._dispatcher is None:
            raise RuntimeError("Agent runtime is not initialized.")
        messages = self._build_messages(text=text, src_lang=src_lang, tgt_lang=tgt_lang)
        extra = build_extra_body(self.model_family)
        for _ in range(8):
            req: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": self._tools,
                "tool_choice": "auto",
                "temperature": float(self.temperature),
                "max_tokens": int(self.max_tokens),
            }
            if extra:
                req["extra_body"] = extra
            response = self._client.chat.completions.create(**req)
            msg = response.choices[0].message

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_message)

            if not msg.tool_calls:
                return (msg.content or "").strip()

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if tool_name not in self._dispatcher:
                    tool_output: Any = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_output = self._dispatcher[tool_name](**args)
                    except Exception as e:
                        tool_output = {"error": str(e)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    }
                )
        raise RuntimeError("Tool-calling loop exceeded max rounds without final answer.")
