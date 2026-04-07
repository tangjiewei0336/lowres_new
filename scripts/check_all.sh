#!/usr/bin/env bash
# 在仓库根目录：bash scripts/check_all.sh
# 使用 conda 环境 lowres（请先: conda create -n lowres python=3.12 && conda activate lowres && pip install -r requirements.txt）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=activate_conda_lowres.sh
source "${SCRIPT_DIR}/activate_conda_lowres.sh"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/datasets/cache/modelscope}"
python scripts/check_syntax_and_imports.py
python scripts/check_modelscope_download.py
if [[ "${RUN_PREPARE_DATASETS:-0}" == "1" ]]; then
  python scripts/prepare_datasets.py
  python scripts/expand_language_pairs.py
fi
echo "check_all 完成。"
