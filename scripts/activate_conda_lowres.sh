#!/usr/bin/env bash
# 在其它脚本中 source 本文件以激活 conda 环境，并修复运行时库加载顺序。
# 默认环境名：lowres；可用 CONDA_ENV_NAME 覆盖（例如 serve 用 lowres-serve）。
# 用法:
#   source scripts/activate_conda_lowres.sh
#   CONDA_ENV_NAME=lowres-serve source scripts/activate_conda_lowres.sh

if [[ -z "${CONDA_EXE:-}" ]]; then
  if [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
    CONDA_EXE="${HOME}/miniforge3/bin/conda"
  elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
    CONDA_EXE="${HOME}/anaconda3/bin/conda"
  elif [[ -x "/opt/conda/bin/conda" ]]; then
    CONDA_EXE="/opt/conda/bin/conda"
  else
    CONDA_EXE="$(command -v conda || true)"
  fi
fi

if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
  echo "未找到 conda，请先安装 Miniforge/Anaconda 并创建环境: conda create -n lowres python=3.12" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
ENV_NAME="${CONDA_ENV_NAME:-lowres}"
conda activate "${ENV_NAME}"

# 关键：确保优先使用 conda 的运行时库（避免误用系统 /usr/lib 的旧 libstdc++.so.6）
# 否则可能出现：CXXABI_1.3.15 not found（vLLM/import sqlite3 触发）
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
  else
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
  fi
fi
