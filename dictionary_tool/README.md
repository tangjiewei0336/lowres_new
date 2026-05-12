# Dictionary Tool

Standalone callable dictionary tool with **local method dispatch** (no FastMCP).

It exposes the project lexicons under `training/data/dictionaries/moe_lexicon/`
as in-process tools, so downstream agents can call dictionary lookup directly.

## Features

- `lookup_dictionary`
  - **Vietnamese (vie_Latn) → English (eng_Latn) only**; loads `dict_terms_vie_Latn__eng_Latn.jsonl` under the lexicon directory
  - contains-based lookup (including exact) on Vietnamese `source_text` via `term` (no language parameters)
  - returns ranked entries, English glosses / `target_candidates`, and optional `examples` when present in the lexicon
- `list_dictionary_pairs`
  - list available language pairs (currently Vietnamese → English when lexicon is present)

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
python server.py --tool lookup_dictionary --term nước --top-k 5
```

Translation generation is done by the agent model itself (using your configured vLLM endpoint in the agent runtime).
Typical defaults:

```text
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
```

So if your vLLM service is already running, the runtime can invoke:

- `lookup_dictionary(term, top_k, offset)`

## Build Vietnamese ↔ English lexicon (Kaikki / Wiktextract)

Download [Kaikki Vietnamese JSONL](https://kaikki.org/dictionary/Vietnamese/kaikki.org-dictionary-Vietnamese.jsonl), then:

```bash
python scripts/dictionary/prepare_kaikki_vietnamese_dictionary.py \
  --input /path/to/kaikki.org-dictionary-Vietnamese.jsonl \
  --output-dir training/data/dictionaries/moe_lexicon
```

This writes `dict_terms_vie_Latn__eng_Latn.jsonl` (one row per sense, optional example snippets).

## Automatic tool-calling demo

If you want an LLM agent to automatically call dictionary methods, use:

```bash
cd dictionary_tool
python langgraph_mcp_agent_demo.py \
  --src-lang vie_Latn \
  --tgt-lang eng_Latn \
  --text "nước"
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
