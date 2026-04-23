from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from agent_prompt import build_translation_agent_prompt


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
    ) -> None:
        self.model = model
        self.model_family = model_family
        self.base_url = base_url
        self.api_key = api_key
        self.lexicon_dir = lexicon_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._stdio_cm = None
        self._session_cm = None
        self._session = None
        self._agent = None

    async def __aenter__(self) -> "LangGraphDictionaryAgentRuntime":
        from langchain_openai import ChatOpenAI
        from langchain_mcp_adapters.tools import load_mcp_tools
        from langgraph.prebuilt import create_react_agent
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env["DICTIONARY_TOOL_LEXICON_DIR"] = str(self.lexicon_dir.resolve())
        env["OPENAI_API_BASE"] = self.base_url
        env["OPENAI_API_KEY"] = self.api_key

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["server.py"],
            env=env,
            cwd=str(Path(__file__).resolve().parent),
        )

        model_kwargs: dict[str, Any] = {}
        fam = (self.model_family or "").lower()
        if fam in ("qwen3", "qwen", "qwen3-4b", "qwen3-8b", "qwen3.5", "qwen3-5", "qwen35"):
            model_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        model = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=float(self.temperature),
            max_tokens=int(self.max_tokens),
            model_kwargs=model_kwargs,
        )

        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        tools = await load_mcp_tools(self._session)
        self._agent = create_react_agent(model, tools, prompt=build_translation_agent_prompt())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc, tb)
        self._agent = None
        self._session = None
        self._session_cm = None
        self._stdio_cm = None

    async def translate(self, *, text: str, src_lang: str, tgt_lang: str) -> str:
        if self._agent is None:
            raise RuntimeError("Agent runtime is not initialized.")
        user_text = (
            f"src_lang={src_lang}\n"
            f"tgt_lang={tgt_lang}\n"
            f"text={text}\n\n"
            "Translate this text. Use dictionary tools when useful."
        )
        result = await self._agent.ainvoke({"messages": [{"role": "user", "content": user_text}]})
        messages = result.get("messages", [])
        if not messages:
            raise RuntimeError("Agent returned no messages.")
        final = messages[-1]
        content = getattr(final, "content", None)
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
