#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/training/moe_experts}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/moe_train/$RUN_TS}"
FAIL_DIR="${LOG_DIR}/failed"
STATUS_DIR="${LOG_DIR}/.status"
SUMMARY="${LOG_DIR}/summary.tsv"

detect_gpus() {
  if [[ -n "${GPUS:-}" ]]; then
    echo "${GPUS}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -
  fi
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

gpu_chunk_for_slot() {
  local slot="$1"
  local gpu_count="$2"
  local per_job="$3"
  local start=$((slot * per_job))
  local ids=()
  local idx

  for ((idx = 0; idx < per_job; idx++)); do
    ids+=("${GPU_IDS[$(((start + idx) % gpu_count))]}")
  done

  join_by_comma "${ids[@]}"
}

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

mkdir -p "${LOG_DIR}" "${FAIL_DIR}" "${STATUS_DIR}"
: >"${SUMMARY}"

GPU_LIST="$(detect_gpus | tr -d '[:space:]')"
IFS=, read -r -a GPU_IDS <<<"${GPU_LIST}"
if [[ -z "${GPU_LIST}" ]]; then
  GPU_IDS=()
fi

GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
if ! is_positive_int "${GPUS_PER_JOB}"; then
  echo "GPUS_PER_JOB must be a positive integer, got: ${GPUS_PER_JOB}" >&2
  exit 1
fi

if [[ -n "${MAX_JOBS:-}" ]]; then
  CONCURRENCY="${MAX_JOBS}"
elif [[ "${#GPU_IDS[@]}" -gt 0 ]]; then
  CONCURRENCY=$(((${#GPU_IDS[@]} + GPUS_PER_JOB - 1) / GPUS_PER_JOB))
else
  CONCURRENCY=1
fi

if ! is_positive_int "${CONCURRENCY}"; then
  echo "MAX_JOBS must be a positive integer, got: ${CONCURRENCY}" >&2
  exit 1
fi

echo "MoE expert configs: ${#configs[@]}"
echo "Parallel jobs: ${CONCURRENCY}"
if [[ "${#GPU_IDS[@]}" -gt 0 ]]; then
  echo "GPUs: ${GPU_LIST} (${GPUS_PER_JOB} per job)"
else
  echo "GPUs: not pinned; jobs inherit current CUDA_VISIBLE_DEVICES"
fi
echo "Logs: ${LOG_DIR}"
echo -e "status\texpert\tlog" >"${SUMMARY}"

running=0
pids=()
free_slots=()
for ((slot = 0; slot < CONCURRENCY; slot++)); do
  free_slots+=("${slot}")
done

run_one() {
  local cfg="$1"
  local slot="$2"
  local name
  local log_path
  local gpu_chunk=""
  local status

  name="$(basename "${cfg}" .yaml)"
  log_path="${LOG_DIR}/${name}.log"

  if [[ "${#GPU_IDS[@]}" -gt 0 ]]; then
    gpu_chunk="$(gpu_chunk_for_slot "${slot}" "${#GPU_IDS[@]}" "${GPUS_PER_JOB}")"
    (
      set -o pipefail
      echo "[$(date '+%F %T')] START ${name}"
      echo "config=${cfg}"
      echo "CUDA_VISIBLE_DEVICES=${gpu_chunk}"
      CUDA_VISIBLE_DEVICES="${gpu_chunk}" llamafactory-cli train "${cfg}"
      status="$?"
      if [[ "${status}" -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE ${name}"
      else
        echo "[$(date '+%F %T')] FAIL ${name}: exit=${status}"
      fi
      exit "${status}"
    ) >"${log_path}" 2>&1
  else
    (
      set -o pipefail
      echo "[$(date '+%F %T')] START ${name}"
      echo "config=${cfg}"
      echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
      llamafactory-cli train "${cfg}"
      status="$?"
      if [[ "${status}" -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE ${name}"
      else
        echo "[$(date '+%F %T')] FAIL ${name}: exit=${status}"
      fi
      exit "${status}"
    ) >"${log_path}" 2>&1
  fi
}

wait_for_one() {
  local pid status name log_path status_path

  while true; do
    for idx in "${!pids[@]}"; do
      pid="${pids[$idx]}"
      status_path="${pid_statuses[$idx]}"
      if [[ -f "${status_path}" ]]; then
        status="$(cat "${status_path}")"
        set +e
        wait "${pid}"
        set -e

        name="${pid_names[$idx]}"
        log_path="${pid_logs[$idx]}"
        FREED_SLOT="${pid_slots[$idx]}"
        if [[ "${status}" -eq 0 ]]; then
          echo "[PASS] ${name}: ${log_path}"
          echo -e "PASS\t${name}\t${log_path}" >>"${SUMMARY}"
        else
          echo "[FAIL] ${name}: exit=${status} log=${log_path}" >&2
          echo -e "FAIL(${status})\t${name}\t${log_path}" >>"${SUMMARY}"
          touch "${FAIL_DIR}/${name}.exit_${status}"
        fi

        unset 'pids[idx]'
        unset 'pid_names[idx]'
        unset 'pid_logs[idx]'
        unset 'pid_statuses[idx]'
        unset 'pid_slots[idx]'
        set +u
        pids=("${pids[@]}")
        pid_names=("${pid_names[@]}")
        pid_logs=("${pid_logs[@]}")
        pid_statuses=("${pid_statuses[@]}")
        pid_slots=("${pid_slots[@]}")
        set -u
        return "${status}"
      fi
    done
    sleep 5
  done
}

overall_status=0
pid_names=()
pid_logs=()
pid_statuses=()
pid_slots=()
FREED_SLOT=""

for cfg in "${configs[@]}"; do
  while [[ "${running}" -ge "${CONCURRENCY}" ]]; do
    if ! wait_for_one; then
      overall_status=1
    fi
    running=$((running - 1))
    free_slots+=("${FREED_SLOT}")
  done

  name="$(basename "${cfg}" .yaml)"
  log_path="${LOG_DIR}/${name}.log"
  slot="${free_slots[0]}"
  unset 'free_slots[0]'
  set +u
  free_slots=("${free_slots[@]}")
  set -u
  status_path="${STATUS_DIR}/${name}.${slot}.$RANDOM.status"

  if [[ "${#GPU_IDS[@]}" -gt 0 ]]; then
    echo "[START] ${name}: gpu=$(gpu_chunk_for_slot "${slot}" "${#GPU_IDS[@]}" "${GPUS_PER_JOB}") log=${log_path}"
  else
    echo "[START] ${name}: log=${log_path}"
  fi

  (
    set +e
    run_one "${cfg}" "${slot}"
    status="$?"
    echo "${status}" >"${status_path}"
    exit "${status}"
  ) &
  pid=$!
  pids+=("${pid}")
  pid_names+=("${name}")
  pid_logs+=("${log_path}")
  pid_statuses+=("${status_path}")
  pid_slots+=("${slot}")
  running=$((running + 1))
done

while [[ "${running}" -gt 0 ]]; do
  if ! wait_for_one; then
    overall_status=1
  fi
  running=$((running - 1))
  free_slots+=("${FREED_SLOT}")
done

if [[ "${overall_status}" -eq 0 ]]; then
  echo "All MoE expert trainings finished."
else
  echo "Some MoE expert trainings failed. See: ${SUMMARY}" >&2
fi
echo "Summary: ${SUMMARY}"
exit "${overall_status}"
