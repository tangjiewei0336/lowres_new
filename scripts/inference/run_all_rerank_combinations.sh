#!/usr/bin/env bash
set -euo pipefail

# Run the four inference-time reranking combinations:
#   sample + llm
#   sample + comet-qe
#   rag    + llm
#   rag    + comet-qe
#
# Example:
#   OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY \
#   bash scripts/inference/run_all_rerank_combinations.sh qwen3-8b qwen qwen3_8b_rerank
#
# Build the RAG FAISS index once before running rag combinations:
#   conda run -n lowres python scripts/inference/build_faiss_rag_index.py \
#     --aug-data-dir training/data/multilingual/fineweb2_synth \
#     --out-dir indexes/faiss_aug_fineweb

MODEL="${1:?usage: $0 MODEL MODEL_FAMILY MODEL_TAG_PREFIX [extra args...]}"
MODEL_FAMILY="${2:?usage: $0 MODEL MODEL_FAMILY MODEL_TAG_PREFIX [extra args...]}"
TAG_PREFIX="${3:?usage: $0 MODEL MODEL_FAMILY MODEL_TAG_PREFIX [extra args...]}"
shift 3

COMMON_ARGS=(
  --model "${MODEL}"
  --model-family "${MODEL_FAMILY}"
  --base-url "${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
  --api-key "${OPENAI_API_KEY:-EMPTY}"
  "$@"
)

for candidate_mode in sample rag; do
  for reranker in llm comet-qe; do
    tag="${TAG_PREFIX}_${candidate_mode}_${reranker}"
    echo "=== ${candidate_mode} + ${reranker} -> ${tag} ==="
    conda run -n lowres python scripts/inference/generate_reranked_hypotheses.py \
      "${COMMON_ARGS[@]}" \
      --candidate-mode "${candidate_mode}" \
      --reranker "${reranker}" \
      --model-tag "${tag}"
  done
done
