#!/usr/bin/env bash
# Semantic-reward translation fine-tuning launcher.
#
# Prerequisites:
#   uv sync 或 pip install -r requirements.txt
#   Prepare data, e.g.:
#     python scripts/prepare/prepare_nllb_for_llamafactory.py \\
#       --export-from-config --pairs-config training/ccmatrix_pair_limits.json
#
# Example (zh->en, cross-lingual alignment + fluency + SFT anchor):
#   bash scripts/run/run_train_semantic_rl.sh
#
# Override via env vars:
#   MODEL_PATH=... DATA=... SRC_LANG=... TGT_LANG=... bash scripts/run/run_train_semantic_rl.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-8B_latest}"
DATA="${DATA:-$ROOT/training/data/multilingual/nllb/nllb_mt_zho_Hans__eng_Latn.jsonl}"
SRC_LANG="${SRC_LANG:-zho_Hans}"
TGT_LANG="${TGT_LANG:-eng_Latn}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/models/Qwen3-8B_${SRC_LANG}_to_${TGT_LANG}_semantic_rl_lora}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat/multilingual-e5-base}"
REWARD_MODE="${REWARD_MODE:-align}"
FLUENCY_MODEL="${FLUENCY_MODEL:-$MODEL_PATH}"
FLUENCY_WEIGHT="${FLUENCY_WEIGHT:-0.1}"
SFT_WEIGHT="${SFT_WEIGHT:-0.5}"
LIMIT="${LIMIT:-0}"

python scripts/train/semantic_rl_train.py \
  --model-path "$MODEL_PATH" \
  --data "$DATA" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --output-dir "$OUTPUT_DIR" \
  --reward-mode "$REWARD_MODE" \
  --embedding-model "$EMBEDDING_MODEL" \
  --fluency-model "$FLUENCY_MODEL" \
  --fluency-weight "$FLUENCY_WEIGHT" \
  --sft-weight "$SFT_WEIGHT" \
  --limit "$LIMIT" \
  "$@"
