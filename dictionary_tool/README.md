# Dictionary Tool

Standalone callable dictionary tool built on top of **FastMCP**.

It exposes the project dictionary lexicons under `training/data/dictionaries/moe_lexicon/`
as MCP tools, so downstream agents can call dictionary lookup directly instead of
parsing JSONL by hand.

## Features

- `lookup_dictionary`
  - exact-match lookup by `src_lang`, `tgt_lang`, and `term`
  - returns ranked dictionary entries and candidate translations
- `list_dictionary_pairs`
  - list available language pairs
- `search_dictionary_pairs`
  - search a term across multiple pairs
- `llm_translate_via_vllm`
  - call the previously deployed vLLM OpenAI-compatible endpoint
  - explicit `src_lang` / `tgt_lang` parameters
  - default model family disables thinking for Qwen-style models

## Install

```bash
cd dictionary_tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default lexicon directory:

```text
../training/data/dictionaries/moe_lexicon
```

You can override it with `DICTIONARY_TOOL_LEXICON_DIR`.

Start the MCP server over stdio:

```bash
cd dictionary_tool
python server.py
```

## Direct MCP client demo

This demo starts `server.py` as a subprocess and talks to it over MCP stdio
without using any extra client SDK.

```bash
cd dictionary_tool
python demo_mcp_client.py --src-lang eng_Latn --tgt-lang zho_Hans --term dictionary
```

The LLM tool uses the same server and defaults to:

```text
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
```

So if your vLLM service is already running, MCP callers can invoke:

- `lookup_dictionary(src_lang, tgt_lang, term, top_k)`
- `llm_translate_via_vllm(text, src_lang, tgt_lang, model, ...)`

## LangGraph automatic tool-calling demo

If you want an **LLM agent** to automatically call the MCP dictionary server,
use the LangGraph demo:

```bash
cd dictionary_tool
python langgraph_mcp_agent_demo.py \
  --src-lang eng_Latn \
  --tgt-lang zho_Hans \
  --text "dictionary"
```

This demo:

1. starts the local MCP server over stdio
2. loads MCP tools into LangChain via `langchain-mcp-adapters`
3. creates a LangGraph ReAct agent
4. lets the LLM decide when to call:
   - `lookup_dictionary`
   - `llm_translate_via_vllm`

To inspect tool calls:

```bash
python langgraph_mcp_agent_demo.py \
  --src-lang eng_Latn \
  --tgt-lang zho_Hans \
  --text "dictionary" \
  --show-tool-trace
```

## Evaluate the agent

You can evaluate the LangGraph+MCP agent on FLORES and reuse the same BLEU/COMET
pipeline used elsewhere in the repo:

```bash
cd dictionary_tool
python eval_langgraph_agent.py \
  --model qwen3-8b \
  --model-family qwen3 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --corpus flores
```

For resume:

```bash
python eval_langgraph_agent.py \
  --output-run-dir ../eval_multilingual/langgraph_agent_resume \
  --resume
```

Shell wrapper:

```bash
cd dictionary_tool
bash run_eval_langgraph_agent.sh
```

With resume:

```bash
bash run_eval_langgraph_agent.sh \
  --output-run-dir ../eval_multilingual/langgraph_agent_resume \
  --resume
```

## Tests

```bash
cd dictionary_tool
source .venv/bin/activate
pytest
```
