#!/usr/bin/env bash
# 在仓库根目录：bash scripts/check_all.sh
# 请先安装依赖: uv sync 或 pip install -r requirements.txt
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"
python scripts/check_syntax_and_imports.py
python scripts/check_modelscope_download.py
if [[ "${RUN_PREPARE_DATASETS:-0}" == "1" ]]; then
  python scripts/prepare/prepare_datasets.py
  python scripts/expand_language_pairs.py
fi
echo "check_all 完成。"
