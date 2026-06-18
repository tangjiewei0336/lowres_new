#!/usr/bin/env bash
# Continue pretraining the 300k-step HF checkpoint on all ingredient2 shards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/share/airesearch/slm/checkpoints/slm_msxf_1b/checkpoints/ckp-2026-06-11-fullset-v1.5-13lang-nosynth-gbs9M-cpt-hf/300000}"
DATA_ROOT="${DATA_ROOT:-/share/aidev/data/slm_dataset_tokenized_docshuffled_qwen3/dolma3_dolmino_mix-100B-1125-tokQwen3/ingredient2}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/models/slm_msxf_1b_ingredient2_cpt_4x5090}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT}/training/deepspeed_zero2_cpt_4x5090.json}"

SEQ_LEN="${SEQ_LEN:-4096}"
MAX_STEPS="${MAX_STEPS:-30000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

mkdir -p "${OUTPUT_DIR}"

torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/train/continue_pretrain_ds_indexed.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --seq_len "${SEQ_LEN}" \
  --max_steps "${MAX_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --deepspeed "${DEEPSPEED_CONFIG}"
