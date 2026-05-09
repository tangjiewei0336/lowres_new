#!/usr/bin/env bash
# 批量评估 MoE checkpoints：
# 对每个 step 自动生成 manifest（adapter_path -> checkpoint-STEP），
# 然后串行执行：启动 vLLM -> 跑 run_eval_moe_router -> 关闭 vLLM。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../activate_conda_lowres.sh
source "${SCRIPT_DIR}/../activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

BASE_MANIFEST="${BASE_MANIFEST:-${ROOT}/training/moe_router_manifest.json}"
STEP_START="${STEP_START:-1000}"
STEP_END="${STEP_END:-6250}"
STEP_INTERVAL="${STEP_INTERVAL:-1000}"
INCLUDE_END_STEP="${INCLUDE_END_STEP:-0}"
SERVE_SCRIPT="${SERVE_SCRIPT:-${ROOT}/scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-moe_router_ckpt_scan}"
API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-300}"
LOG_RUN_TS="${LOG_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/moe_eval_ckpt/${LOG_RUN_TS}}"
VERBOSE_LOG="${VERBOSE_LOG:-1}"

usage() {
  cat <<'EOF'
用法:
  bash scripts/run/run_eval_moe_checkpoints.sh [run_eval_moe_router.py 参数...]

默认会评估 step=1000,2000,...,6000（STEP_END=6250 且 INCLUDE_END_STEP=0）。
如需也评估 6250，请设置 INCLUDE_END_STEP=1。

常用环境变量:
  BASE_MANIFEST      基础 manifest（默认 training/moe_router_manifest.json）
  STEP_START         起始 step（默认 1000）
  STEP_END           结束 step（默认 6250）
  STEP_INTERVAL      间隔（默认 1000）
  INCLUDE_END_STEP   0/1，末尾不整除时是否附加 STEP_END（默认 0）
  SERVE_SCRIPT       serve 脚本路径
  EVAL_TAG_PREFIX    评测标签前缀（默认 moe_router_ckpt_scan）
  OPENAI_API_BASE    API 地址（默认 http://127.0.0.1:8000/v1）
  READY_TIMEOUT_SEC  等待服务就绪超时（默认 300）
  LOG_DIR            日志目录（默认 logs/moe_eval_ckpt/<timestamp>）
  VERBOSE_LOG        0/1，是否在终端打印更详细日志（默认 1）

示例:
  STEP_START=1000 STEP_END=6250 STEP_INTERVAL=1000 INCLUDE_END_STEP=1 \
  bash scripts/run/run_eval_moe_checkpoints.sh --comet-model none
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${BASE_MANIFEST}" ]]; then
  echo "未找到 BASE_MANIFEST: ${BASE_MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${SERVE_SCRIPT}" ]]; then
  echo "未找到 SERVE_SCRIPT: ${SERVE_SCRIPT}" >&2
  exit 1
fi
if ! [[ "${STEP_START}" =~ ^[0-9]+$ && "${STEP_END}" =~ ^[0-9]+$ && "${STEP_INTERVAL}" =~ ^[0-9]+$ ]]; then
  echo "STEP_START/STEP_END/STEP_INTERVAL 必须是非负整数。" >&2
  exit 1
fi
if [[ "${STEP_INTERVAL}" -le 0 ]]; then
  echo "STEP_INTERVAL 必须 > 0。" >&2
  exit 1
fi
if [[ "${STEP_END}" -lt "${STEP_START}" ]]; then
  echo "STEP_END 必须 >= STEP_START。" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
TMP_DIR="$(mktemp -d "${LOG_DIR}/tmp_manifest.XXXXXX")"
SUMMARY_TSV="${LOG_DIR}/summary.tsv"
echo -e "step\tstatus\trun_tag\tmanifest\tserve_log\terror" > "${SUMMARY_TSV}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT

build_step_manifest() {
  local step="$1"
  local out_manifest="$2"
  python - "${BASE_MANIFEST}" "${out_manifest}" "${step}" <<'PY'
import json
import re
import sys
from pathlib import Path

base_manifest = Path(sys.argv[1])
out_manifest = Path(sys.argv[2])
step = int(sys.argv[3])
data = json.loads(base_manifest.read_text(encoding="utf-8"))
experts = data.get("experts") or []
if not experts:
    raise SystemExit(f"No experts found in {base_manifest}")

step_re = re.compile(r"^checkpoint[-_]\d+$")
for item in experts:
    path = str(item.get("adapter_path") or "").strip()
    if not path:
        raise SystemExit(f"Bad adapter_path in expert: {item!r}")
    p = Path(path)
    if step_re.match(p.name):
        p = p.parent
    item["adapter_root"] = str(p)
    item["adapter_path"] = str(p / f"checkpoint-{step}")

out_manifest.parent.mkdir(parents=True, exist_ok=True)
out_manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

wait_server_ready() {
  local timeout_sec="$1"
  python - "${API_BASE}" "${timeout_sec}" "${VERBOSE_LOG}" <<'PY'
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/")
timeout_sec = int(sys.argv[2])
verbose = sys.argv[3] == "1"
url = f"{base}/models"
deadline = time.time() + timeout_sec
last_error = ""
attempt = 0

while time.time() < deadline:
    attempt += 1
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if 200 <= resp.status < 300:
                if verbose:
                    print(f"[wait_server_ready] ready after {attempt} checks: {url}", flush=True)
                print("ready")
                raise SystemExit(0)
            last_error = f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        last_error = f"HTTPError {e.code}"
    except Exception as e:
        last_error = str(e)
    if verbose:
        print(f"[wait_server_ready] attempt={attempt} not ready: {last_error}", flush=True)
    time.sleep(2)

print(f"timeout waiting {url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

steps=()
for ((s = STEP_START; s <= STEP_END; s += STEP_INTERVAL)); do
  steps+=("${s}")
done
if [[ "${INCLUDE_END_STEP}" == "1" ]]; then
  last_idx=$((${#steps[@]} - 1))
  if [[ "${#steps[@]}" -eq 0 || "${steps[$last_idx]}" -ne "${STEP_END}" ]]; then
    steps+=("${STEP_END}")
  fi
fi

if [[ "${#steps[@]}" -eq 0 ]]; then
  echo "没有可评估的 steps。" >&2
  exit 1
fi

echo "Batch evaluate steps: ${steps[*]}"
echo "Logs: ${LOG_DIR}"
echo "Summary: ${SUMMARY_TSV}"
echo "Verbose log: ${VERBOSE_LOG}"

for step in "${steps[@]}"; do
  run_tag="${EVAL_TAG_PREFIX}_checkpoint-${step}"
  manifest_step="${TMP_DIR}/moe_router_manifest.checkpoint-${step}.json"
  serve_log="${LOG_DIR}/serve_checkpoint-${step}.log"
  eval_log="${LOG_DIR}/eval_checkpoint-${step}.log"

  echo "========== step ${step} =========="
  echo "[step ${step}] run_tag=${run_tag}"
  echo "[step ${step}] manifest=${manifest_step}"
  build_step_manifest "${step}" "${manifest_step}"

  cleanup
  echo "[step ${step}] start server... (log: ${serve_log})"
  MANIFEST="${manifest_step}" bash "${SERVE_SCRIPT}" > "${serve_log}" 2>&1 &
  SERVER_PID="$!"
  echo "[step ${step}] server pid=${SERVER_PID}"

  if ! wait_server_ready "${READY_TIMEOUT_SEC}"; then
    echo "[step ${step}] server not ready, see ${serve_log}" >&2
    echo -e "${step}\tFAIL\t${run_tag}\t${manifest_step}\t${serve_log}\tserver_not_ready" >> "${SUMMARY_TSV}"
    exit 1
  fi

  echo "[step ${step}] run eval... (log: ${eval_log})"
  if [[ "${VERBOSE_LOG}" == "1" ]]; then
    if MOE_ROUTER_MANIFEST="${manifest_step}" EVAL_MODEL_TAG="${run_tag}" bash "${SCRIPT_DIR}/run_eval_moe_router.sh" "$@" 2>&1 | tee "${eval_log}"; then
      echo -e "${step}\tPASS\t${run_tag}\t${manifest_step}\t${serve_log}\t" >> "${SUMMARY_TSV}"
      echo "[step ${step}] eval PASS"
    else
      echo "[step ${step}] eval failed, see ${eval_log}" >&2
      echo -e "${step}\tFAIL\t${run_tag}\t${manifest_step}\t${serve_log}\teval_failed" >> "${SUMMARY_TSV}"
      exit 1
    fi
  elif MOE_ROUTER_MANIFEST="${manifest_step}" EVAL_MODEL_TAG="${run_tag}" bash "${SCRIPT_DIR}/run_eval_moe_router.sh" "$@" > "${eval_log}" 2>&1; then
    echo -e "${step}\tPASS\t${run_tag}\t${manifest_step}\t${serve_log}\t" >> "${SUMMARY_TSV}"
    echo "[step ${step}] eval PASS"
  else
    echo "[step ${step}] eval failed, see ${eval_log}" >&2
    echo -e "${step}\tFAIL\t${run_tag}\t${manifest_step}\t${serve_log}\teval_failed" >> "${SUMMARY_TSV}"
    exit 1
  fi

  cleanup
done

echo "全部 checkpoint 评估完成。"
echo "Summary: ${SUMMARY_TSV}"
