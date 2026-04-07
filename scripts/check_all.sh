#!/usr/bin/env bash
# 在仓库根目录：bash scripts/check_all.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=activate_venv.sh
source "${SCRIPT_DIR}/activate_venv.sh"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
python scripts/check_syntax_and_imports.py
python scripts/check_modelscope_download.py
echo "check_all 完成。"
