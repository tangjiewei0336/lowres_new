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

## Tests

```bash
cd dictionary_tool
source .venv/bin/activate
pytest
```
