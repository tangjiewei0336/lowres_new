#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/training/moe_experts}"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli not found. Activate the training environment first." >&2
  exit 127
fi

shopt -s nullglob
configs=("$CONFIG_DIR"/llamafactory_qwen3_8b_moe_*.yaml)
if [ "${#configs[@]}" -eq 0 ]; then
  echo "No expert configs found in $CONFIG_DIR. Run scripts/moe/generate_pair_expert_assets.py first." >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  echo "=== Training MoE expert: $cfg"
  llamafactory-cli train "$cfg"
done
