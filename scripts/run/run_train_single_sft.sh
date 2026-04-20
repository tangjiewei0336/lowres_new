#!/usr/bin/env bash
# Train one Qwen3-8B LoRA adapter on all directions merged together.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-${ROOT}/training/llamafactory_qwen3_8b_all_directions_sft_lora.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi

exec llamafactory-cli train "${CONFIG}"
