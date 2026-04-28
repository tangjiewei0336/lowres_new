# Dictionary Tool

Standalone callable dictionary tool with **local method dispatch** (no FastMCP).

It exposes the project lexicons under `training/data/dictionaries/moe_lexicon/`
as in-process tools, so downstream agents can call dictionary lookup directly.

## Features

- `lookup_dictionary`
  - contains-based lookup (including exact) by `src_lang`, `tgt_lang`, and `term`
  - returns ranked dictionary entries and candidate translations
  - if current `src_lang->tgt_lang` has no hit, auto returns fallback suggestions from other pairs/languages
- `list_dictionary_pairs`
  - list available language pairs

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

Run the local tool CLI:

```bash
cd dictionary_tool
python server.py
```

Translation generation is done by the agent model itself (using your configured vLLM endpoint in the agent runtime).
Typical defaults:

```text
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
```

So if your vLLM service is already running, the runtime can invoke:

- `lookup_dictionary(src_lang, tgt_lang, term, top_k, offset, fallback_top_k)`

## Automatic tool-calling demo

If you want an LLM agent to automatically call dictionary methods, use:

```bash
cd dictionary_tool
python langgraph_mcp_agent_demo.py \
  --src-lang eng_Latn \
  --tgt-lang zho_Hans \
  --text "dictionary"
```

## Evaluate the agent

You can evaluate the dictionary tool-calling agent on FLORES and reuse the same BLEU/COMET
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
