#!/usr/bin/env bash
# 并发运行多家 OpenAI 兼容 API 的 hypotheses 生成脚本，并汇总结果。
# 本脚本只负责生成 hypotheses.jsonl，不做 BLEU/COMET 评测。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${OUT_DIR:-${ROOT}/logs/hypotheses_generate_parallel/${RUN_TS}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

ALL_PROVIDERS=(zhipu minimax qwen deepseek)

usage() {
  cat <<EOF
用法:
  bash scripts/run/generate/run_all_hypotheses_evals_parallel.sh [provider... -- extra args]

示例:
  bash scripts/run/generate/run_all_hypotheses_evals_parallel.sh
  bash scripts/run/generate/run_all_hypotheses_evals_parallel.sh zhipu qwen -- --max-workers 4

说明:
  - 默认并发运行: ${ALL_PROVIDERS[*]}
  - provider 可选: ${ALL_PROVIDERS[*]}
  - "--" 之后的参数会透传给各 generate_hypotheses_*.sh 脚本
  - 生成结果目录: ${OUT_DIR}/<provider>/hypotheses.jsonl
  - 日志目录: ${LOG_DIR}
EOF
}

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
      -h|--help)
        usage
        exit 0
        ;;
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

script_for_provider() {
  case "$1" in
    zhipu) echo "scripts/run/generate/generate_hypotheses_zhipu_glm5_1.sh" ;;
    minimax) echo "scripts/run/generate/generate_hypotheses_minimax.sh" ;;
    qwen) echo "scripts/run/generate/generate_hypotheses_qwen.sh" ;;
    deepseek) echo "scripts/run/generate/generate_hypotheses_deepseek.sh" ;;
    *)
      echo "unknown provider: $1" >&2
      return 1
      ;;
  esac
}

pids=()
logs=()

echo "== 并发启动 hypotheses 生成 =="
echo "providers: ${selected_providers[*]}"
echo "output: ${OUT_DIR}"
echo "logs: ${LOG_DIR}"

for provider in "${selected_providers[@]}"; do
  script_path="$(script_for_provider "${provider}")"
  log_path="${LOG_DIR}/${provider}.log"
  provider_out_dir="${OUT_DIR}/${provider}"
  mkdir -p "${provider_out_dir}"
  (
    echo "[$(date '+%F %T')] START provider=${provider} script=${script_path} output_run_dir=${provider_out_dir}"
    bash "${script_path}" --output-run-dir "${provider_out_dir}" "${forward_args[@]}"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  logs+=("${log_path}")
  echo "[RUNNING] ${provider}: pid=${pids[${#pids[@]}-1]} log=${log_path}"
done

pass_count=0
fail_count=0

echo
echo "== 等待任务完成 =="
for idx in "${!selected_providers[@]}"; do
  provider="${selected_providers[$idx]}"
  pid="${pids[$idx]}"
  log_path="${logs[$idx]}"
  if wait "${pid}"; then
    echo "[PASS] ${provider}: ${log_path}"
    pass_count=$((pass_count + 1))
  else
    status=$?
    echo "[FAIL] ${provider}: exit=${status} log=${log_path}"
    fail_count=$((fail_count + 1))
  fi
done

echo
echo "== 汇总 =="
echo "PASS: ${pass_count}"
echo "FAIL: ${fail_count}"
echo "OUT_DIR: ${OUT_DIR}"
echo "LOG_DIR: ${LOG_DIR}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 2
fi

exit 0
