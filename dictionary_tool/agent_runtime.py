from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from agent_prompt import build_translation_messages
from llm import build_extra_body, create_client
from tool_registry import (
    FINAL_TRANSLATION_TOOL_NAME,
    build_final_translation_tools,
    build_local_dispatcher,
    build_openai_tools,
)


MAX_TOOL_CALLING_ROUNDS = 5
_ANSI_BLUE = "\033[34m"
_ANSI_GREEN = "\033[32m"
_ANSI_RESET = "\033[0m"
_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL | re.IGNORECASE)
_FINAL_TRANSLATION_CALL_RE = re.compile(
    r"final_translation\s*\(\s*translation\s*=\s*(.*?)\s*\)\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


def _parse_json_maybe(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_final_translation_from_obj(obj: Any) -> str | None:
    if isinstance(obj, list):
        for item in obj:
            found = _extract_final_translation_from_obj(item)
            if found:
                return found
        return None
    if not isinstance(obj, dict):
        return None

    name = obj.get("name")
    args = obj.get("arguments")
    function = obj.get("function")
    if isinstance(function, dict):
        name = function.get("name", name)
        args = function.get("arguments", args)

    if isinstance(args, str):
        parsed_args = _parse_json_maybe(args)
        if parsed_args is not None:
            args = parsed_args

    if name == FINAL_TRANSLATION_TOOL_NAME and isinstance(args, dict):
        translation = str(args.get("translation", "")).strip()
        return translation or None

    for value in obj.values():
        found = _extract_final_translation_from_obj(value)
        if found:
            return found
    return None


def _strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def _tool_call_from_obj(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        calls: list[dict[str, Any]] = []
        for item in obj:
            calls.extend(_tool_call_from_obj(item))
        return calls
    if not isinstance(obj, dict):
        return []

    name = obj.get("name")
    args = obj.get("arguments")
    function = obj.get("function")
    if isinstance(function, dict):
        name = function.get("name", name)
        args = function.get("arguments", args)

    if isinstance(args, str):
        parsed_args = _parse_json_maybe(args)
        if parsed_args is not None:
            args = parsed_args

    if isinstance(name, str):
        return [{"name": name, "arguments": args if isinstance(args, dict) else {}}]

    calls: list[dict[str, Any]] = []
    for value in obj.values():
        calls.extend(_tool_call_from_obj(value))
    return calls


def extract_text_tool_calls(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_BLOCK_RE.findall(text):
        obj = _parse_json_maybe(block)
        if obj is not None:
            calls.extend(_tool_call_from_obj(obj))

    if not calls:
        obj = _parse_json_maybe(text)
        if obj is not None:
            calls.extend(_tool_call_from_obj(obj))

    final_match = _FINAL_TRANSLATION_CALL_RE.search(text)
    if final_match:
        calls.append(
            {
                "name": FINAL_TRANSLATION_TOOL_NAME,
                "arguments": {"translation": _strip_wrapping_quotes(final_match.group(1))},
            }
        )
    return calls


def extract_final_translation_from_text(text: str) -> str | None:
    for call in extract_text_tool_calls(text):
        if call.get("name") == FINAL_TRANSLATION_TOOL_NAME:
            translation = str(call.get("arguments", {}).get("translation", "")).strip()
            if translation:
                return translation

    obj = _parse_json_maybe(text)
    if obj is not None:
        found = _extract_final_translation_from_obj(obj)
        if found:
            return found
    return None


def looks_like_tool_call_text(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "<tool_call",
            "</tool_call",
            '"tool_calls"',
            '"function"',
            '"arguments"',
            '"lookup_dictionary"',
            f'"{FINAL_TRANSLATION_TOOL_NAME}"',
            FINAL_TRANSLATION_TOOL_NAME,
        )
    )


def _colored(text: str, color: str) -> str:
    return f"{color}{text}{_ANSI_RESET}"


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
        log_llm_output: bool = False,
    ) -> None:
        self.model = model
        self.model_family = model_family
        self.base_url = base_url
        self.api_key = api_key
        self.lexicon_dir = lexicon_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.debug = debug
        self.log_llm_output = log_llm_output
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

    def _log_llm_message(self, *, stage: str, round_index: int, message: Any, finish_reason: str | None) -> None:
        if not self.log_llm_output:
            return
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(message, "tool_calls", None) or []:
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
            )
        payload = {
            "stage": stage,
            "round": round_index,
            "finish_reason": finish_reason,
            "content": getattr(message, "content", None) or "",
            "tool_calls": tool_calls,
        }
        print(_colored("=== LLM OUTPUT START ===", _ANSI_BLUE), file=sys.stderr)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        print(_colored("=== LLM OUTPUT END ===", _ANSI_BLUE), file=sys.stderr)

    def _log_tool_result(
        self,
        *,
        stage: str,
        round_index: int,
        tool_call_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
    ) -> None:
        if not self.log_llm_output:
            return
        payload = {
            "stage": stage,
            "round": round_index,
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "arguments": arguments,
            "output": output,
        }
        print(_colored("=== TOOL RESULT START ===", _ANSI_GREEN), file=sys.stderr)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        print(_colored("=== TOOL RESULT END ===", _ANSI_GREEN), file=sys.stderr)

    def _run_text_tool_calls(
        self, calls: list[dict[str, Any]], *, stage: str, round_index: int
    ) -> tuple[str | None, list[dict[str, Any]]]:
        if self._dispatcher is None:
            raise RuntimeError("Agent runtime is not initialized.")
        tool_results: list[dict[str, Any]] = []
        for call in calls:
            tool_name = str(call.get("name", ""))
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            if tool_name == FINAL_TRANSLATION_TOOL_NAME:
                translation = str(args.get("translation", "")).strip()
                if translation:
                    return translation, tool_results
                tool_results.append(
                    {
                        "name": tool_name,
                        "arguments": args,
                        "output": {"error": "final_translation requires a non-empty translation argument."},
                    }
                )
                self._log_tool_result(
                    stage=stage,
                    round_index=round_index,
                    tool_call_id=None,
                    tool_name=tool_name,
                    arguments=args,
                    output=tool_results[-1]["output"],
                )
                continue
            if tool_name not in self._dispatcher:
                tool_output: Any = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_output = self._dispatcher[tool_name](**args)
                except Exception as e:
                    tool_output = {"error": str(e)}
            tool_results.append({"name": tool_name, "arguments": args, "output": tool_output})
            self._log_tool_result(
                stage=stage,
                round_index=round_index,
                tool_call_id=None,
                tool_name=tool_name,
                arguments=args,
                output=tool_output,
            )
        return None, tool_results

    async def translate(self, *, text: str, src_lang: str, tgt_lang: str) -> str:
        if self._client is None or self._dispatcher is None:
            raise RuntimeError("Agent runtime is not initialized.")
        messages = self._build_messages(text=text, src_lang=src_lang, tgt_lang=tgt_lang)
        extra = build_extra_body(self.model_family)
        for round_index in range(1, MAX_TOOL_CALLING_ROUNDS + 1):
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
            choice = response.choices[0]
            msg = choice.message
            self._log_llm_message(
                stage="tool_loop",
                round_index=round_index,
                message=msg,
                finish_reason=getattr(choice, "finish_reason", None),
            )

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
                content = (msg.content or "").strip()
                final_translation = extract_final_translation_from_text(content)
                if final_translation:
                    return final_translation
                text_tool_calls = extract_text_tool_calls(content)
                if text_tool_calls:
                    text_final, text_tool_results = self._run_text_tool_calls(
                        text_tool_calls,
                        stage="text_tool_call",
                        round_index=round_index,
                    )
                    if text_final:
                        return text_final
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous message contained tool calls as text instead of structured "
                                "tool_calls. I executed those local tools. Tool results:\n"
                                f"{json.dumps(text_tool_results, ensure_ascii=False)}\n\n"
                                "Continue from these tool results. The final answer must be submitted by "
                                "calling the final_translation tool. Do not output normal assistant text or "
                                "tool-call markup as text."
                            ),
                        }
                    )
                    continue
                if content and not looks_like_tool_call_text(content):
                    return content
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous message contained tool-call markup instead of the final translation. "
                            "Do not output tool-call markup or normal assistant text. Submit the final answer "
                            "by calling the final_translation tool with the final translated text only."
                        ),
                    }
                )
                break

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if tool_name == FINAL_TRANSLATION_TOOL_NAME:
                    translation = str(args.get("translation", "")).strip()
                    if translation:
                        self._log_tool_result(
                            stage="tool_loop",
                            round_index=round_index,
                            tool_call_id=tc.id,
                            tool_name=tool_name,
                            arguments=args,
                            output={"translation": translation},
                        )
                        return translation
                    tool_output = {"error": "final_translation requires a non-empty translation argument."}
                    self._log_tool_result(
                        stage="tool_loop",
                        round_index=round_index,
                        tool_call_id=tc.id,
                        tool_name=tool_name,
                        arguments=args,
                        output=tool_output,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": json.dumps(tool_output, ensure_ascii=False),
                        }
                    )
                    continue
                if tool_name not in self._dispatcher:
                    tool_output: Any = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_output = self._dispatcher[tool_name](**args)
                    except Exception as e:
                        tool_output = {"error": str(e)}
                self._log_tool_result(
                    stage="tool_loop",
                    round_index=round_index,
                    tool_call_id=tc.id,
                    tool_name=tool_name,
                    arguments=args,
                    output=tool_output,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": (
                    "Dictionary lookup round limit reached. Do not call lookup_dictionary again. "
                    "You must now provide the best final translation based on the source text and "
                    "the dictionary evidence already available. Submit the final answer by calling the "
                    "final_translation tool. Do not output normal assistant text or tool-call markup as text."
                ),
            }
        )
        final_req: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": build_final_translation_tools(),
            "tool_choice": "auto",
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
        }
        if extra:
            final_req["extra_body"] = extra
        response = self._client.chat.completions.create(**final_req)
        choice = response.choices[0]
        msg = choice.message
        self._log_llm_message(
            stage="final_answer",
            round_index=MAX_TOOL_CALLING_ROUNDS + 1,
            message=msg,
            finish_reason=getattr(choice, "finish_reason", None),
        )
        if not msg.tool_calls:
            final_text = (msg.content or "").strip()
            parsed_final = extract_final_translation_from_text(final_text)
            if parsed_final:
                return parsed_final
            if final_text:
                if looks_like_tool_call_text(final_text):
                    raise RuntimeError("Final-answer round returned tool-call markup instead of final text.")
                return final_text
            raise RuntimeError("Final-answer round returned no content.")

        for tc in msg.tool_calls:
            if tc.function.name != FINAL_TRANSLATION_TOOL_NAME:
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            translation = str(args.get("translation", "")).strip()
            if translation:
                self._log_tool_result(
                    stage="final_answer",
                    round_index=MAX_TOOL_CALLING_ROUNDS + 1,
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=args,
                    output={"translation": translation},
                )
                return translation
        raise RuntimeError("Final-answer round did not provide final_translation.")
