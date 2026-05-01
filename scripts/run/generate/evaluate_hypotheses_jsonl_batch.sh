#!/usr/bin/env bash
# 对已生成的 hypotheses.jsonl 批量做 BLEU/COMET 评测。
# 评测调用 run_eval.py --hypotheses-jsonl，不会重复生成 hypothesis。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

ALL_PROVIDERS=(zhipu minimax qwen deepseek)

usage() {
  cat <<EOF
用法:
  bash scripts/run/generate/evaluate_hypotheses_jsonl_batch.sh <batch_dir> [provider... -- extra args]

示例:
  bash scripts/run/generate/evaluate_hypotheses_jsonl_batch.sh logs/hypotheses_generate_parallel/20260428_120000
  bash scripts/run/generate/evaluate_hypotheses_jsonl_batch.sh logs/hypotheses_generate_parallel/20260428_120000 zhipu qwen
  bash scripts/run/generate/evaluate_hypotheses_jsonl_batch.sh logs/hypotheses_generate_parallel/20260428_120000 -- --comet-model none

说明:
  - batch_dir 应包含 <provider>/hypotheses.jsonl
  - 默认依次评测: ${ALL_PROVIDERS[*]}
  - "--" 之后的参数会透传给 scripts/run/run_eval.py
EOF
}

if [[ "${#}" -lt 1 ]]; then
  usage >&2
  exit 1
fi

if [[ "${1}" == "-h" || "${1}" == "--help" ]]; then
  usage
  exit 0
fi

BATCH_DIR="$1"
shift

if [[ ! -d "${BATCH_DIR}" ]]; then
  echo "batch_dir 不存在: ${BATCH_DIR}" >&2
  exit 1
fi

selected_providers=()
forward_args=()
parsing_providers=1

for arg in "$@"; do
  if [[ "${arg}" == "--" ]]; then
    parsing_providers=0
    continue
  fi
  if [[ "${parsing_providers}" -eq 1 ]]; then
    case "${arg}" in
      zhipu|minimax|qwen|deepseek)
        selected_providers+=("${arg}")
        ;;
      *)
        echo "未知 provider: ${arg}" >&2
        usage >&2
        exit 1
        ;;
    esac
  else
    forward_args+=("${arg}")
  fi
done

if [[ "${#selected_providers[@]}" -eq 0 ]]; then
  selected_providers=("${ALL_PROVIDERS[@]}")
fi

pass_count=0
fail_count=0
skip_count=0

echo "== 开始评测已有 hypotheses =="
echo "batch_dir: ${BATCH_DIR}"
echo "providers: ${selected_providers[*]}"

for provider in "${selected_providers[@]}"; do
  provider_dir="${BATCH_DIR}/${provider}"
  hyp_path="${provider_dir}/hypotheses.jsonl"
  if [[ ! -f "${hyp_path}" ]]; then
    echo "[SKIP] ${provider}: 未找到 ${hyp_path}"
    skip_count=$((skip_count + 1))
    continue
  fi

  echo "[RUNNING] ${provider}: ${hyp_path}"
  if python scripts/run/run_eval.py \
    --hypotheses-jsonl "${hyp_path}" \
    --output-run-dir "${provider_dir}" \
    "${forward_args[@]}"; then
    echo "[PASS] ${provider}: ${provider_dir}/metrics.json"
    pass_count=$((pass_count + 1))
  else
    status=$?
    echo "[FAIL] ${provider}: exit=${status} dir=${provider_dir}"
    fail_count=$((fail_count + 1))
  fi
done

echo
echo "== 汇总 =="
echo "PASS: ${pass_count}"
echo "FAIL: ${fail_count}"
echo "SKIP: ${skip_count}"
echo "BATCH_DIR: ${BATCH_DIR}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 2
fi

exit 0
