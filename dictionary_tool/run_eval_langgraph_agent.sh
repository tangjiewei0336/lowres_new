#!/usr/bin/env bash
# LangGraph + MCP dictionary agent evaluation on FLORES.
# Run inside the dictionary_tool environment after dependencies are installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export AGENT_MODEL="${AGENT_MODEL:-qwen3-8b}"
export AGENT_MODEL_FAMILY="${AGENT_MODEL_FAMILY:-qwen3}"
export EVAL_MODEL_TAG="${EVAL_MODEL_TAG:-langgraph_mcp_agent}"
export DICTIONARY_TOOL_LEXICON_DIR="${DICTIONARY_TOOL_LEXICON_DIR:-${ROOT}/training/data/dictionaries/moe_lexicon}"

exec python eval_langgraph_agent.py \
  --model "${AGENT_MODEL}" \
  --model-family "${AGENT_MODEL_FAMILY}" \
  --base-url "${OPENAI_API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --lexicon-dir "${DICTIONARY_TOOL_LEXICON_DIR}" \
  --model-tag "${EVAL_MODEL_TAG}" \
  "$@"
