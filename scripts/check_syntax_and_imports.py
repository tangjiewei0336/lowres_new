#!/usr/bin/env python3
"""编译检查 scripts/*.py 并尝试导入关键依赖（需在已激活的 venv 中运行，或使用 .venv/bin/python）。"""
from __future__ import annotations

import compileall
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    ok = compileall.compile_dir(str(scripts_dir), quiet=1)
    if not ok:
        print("compileall: 存在语法错误", file=sys.stderr)
        return 1

    modules = [
        "modelscope",
        "openai",
        "sacrebleu",
        "comet",
        "torch",
        "tqdm",
        "yaml",
    ]
    failed: list[str] = []
    for m in modules:
        try:
            __import__(m)
        except Exception as e:
            failed.append(f"{m}: {e}")

    if failed:
        print("以下依赖导入失败（请确认已: source .venv/bin/activate && pip install -r requirements.txt）:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        return 1

    print("语法与依赖检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
