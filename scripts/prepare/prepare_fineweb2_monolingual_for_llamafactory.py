#!/usr/bin/env python3
"""
从 Hugging Face 数据集 `HuggingFaceFW/fineweb-2` 下载指定语言子集，
并转换为 LLaMAFactory 可用的单语预训练 JSONL（每行 `{"text": "..."}`）。

默认会读取仓库根目录 `evaluation_config.json`，校验目标语言是否在评测配置中出现。

用法示例：
  # 单语言
  conda run -n lowres python scripts/prepare/prepare_fineweb2_monolingual_for_llamafactory.py \
    --lang spa_Latn --limit 200000

  # 不传 --lang：自动从 evaluation_config.json 的 language_pair_groups 抽取语言并逐个导出
  conda run -n lowres python scripts/prepare/prepare_fineweb2_monolingual_for_llamafactory.py \
    --limit 200000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PREVIEW_N = 50
DEFAULT_LANG_ALIAS: dict[str, str] = {
    # NLLB/FLORES 常用中文标签在 FineWeb-2 中通常对应 cmn_Hani
    "zho_Hans": "cmn_Hani"
}


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_allowed_langs_from_eval_config(path: Path) -> set[str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    groups = cfg.get("language_pair_groups")
    if not isinstance(groups, list):
        return set()
    langs: set[str] = set()
    for group in groups:
        if isinstance(group, list):
            for item in group:
                if isinstance(item, str) and item.strip():
                    langs.add(item.strip())
    return langs


def resolve_text(row: dict[str, Any], keys: list[str], min_chars: int) -> str | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str):
            text = v.strip()
            if len(text) >= min_chars:
                return text

    # 兜底：选择该行里最长的字符串字段
    fallback = ""
    for v in row.values():
        if isinstance(v, str):
            s = v.strip()
            if len(s) > len(fallback):
                fallback = s
    if len(fallback) >= min_chars:
        return fallback
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="FineWeb-2 指定语言 -> LLaMAFactory 单语 jsonl")
    ap.add_argument("--repo-id", default="HuggingFaceFW/fineweb-2", help="Hugging Face 数据集 ID")
    ap.add_argument(
        "--lang",
        action="append",
        dest="langs",
        metavar="CODE",
        help="语言码（如 spa_Latn / zho_Hans），可重复传；不传则从 evaluation_config 读取",
    )
    ap.add_argument("--split", default="train", help="读取 split，默认 train")
    ap.add_argument("--limit", type=int, default=200_000, help="最多写入条数，0 表示不限制")
    ap.add_argument("--min-chars", type=int, default=20, help="最小文本长度（字符数）")
    ap.add_argument(
        "--text-key",
        action="append",
        dest="text_keys",
        metavar="NAME",
        help="文本字段名，可重复。默认尝试：text content raw_content",
    )
    ap.add_argument(
        "--evaluation-config",
        type=Path,
        default=root() / "evaluation_config.json",
        help="评测配置文件路径（用于校验语言是否在配置中）",
    )
    ap.add_argument(
        "--skip-eval-config-check",
        action="store_true",
        help="跳过 evaluation_config.json 校验",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root() / "training" / "data" / "monolingual",
        help="输出目录",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="可选：显式 Hugging Face token（默认读取 HF_TOKEN/HUGGING_FACE_HUB_TOKEN）",
    )
    ap.add_argument(
        "--lang-alias",
        action="append",
        dest="lang_aliases",
        metavar="SRC=DST",
        help="语言别名映射，可重复传。例如 zho_Hans=cmn_Hani",
    )
    ap.add_argument(
        "--strict-missing",
        action="store_true",
        help="严格模式：若映射后语言不在 FineWeb-2 中则报错退出（默认会跳过）",
    )
    ap.add_argument(
        "--no-hard-exit",
        action="store_true",
        help="默认启用硬退出规避 datasets 在部分环境的退出崩溃；传此参数可关闭该行为",
    )
    args = ap.parse_args()

    if not args.evaluation_config.exists():
        raise SystemExit(f"未找到 evaluation_config 文件：{args.evaluation_config}")

    eval_langs = load_allowed_langs_from_eval_config(args.evaluation_config)
    cli_langs = [x.strip() for x in (args.langs or []) if x and x.strip()]

    if cli_langs:
        langs = sorted(set(cli_langs))
        if not args.skip_eval_config_check and eval_langs:
            missing = [x for x in langs if x not in eval_langs]
            if missing:
                raise SystemExit(
                    f"以下语言不在 {args.evaluation_config} 的 language_pair_groups 中：{', '.join(missing)}"
                )
    else:
        if not eval_langs:
            raise SystemExit(
                "未提供 --lang，且 evaluation_config 中未解析到 language_pair_groups 语言。"
            )
        langs = sorted(eval_langs)

    alias_map = dict(DEFAULT_LANG_ALIAS)
    for item in args.lang_aliases or []:
        raw = item.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise SystemExit(f"--lang-alias 格式错误：{raw}（应为 SRC=DST）")
        src, dst = raw.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise SystemExit(f"--lang-alias 格式错误：{raw}（应为 SRC=DST）")
        alias_map[src] = dst

    text_keys = args.text_keys or ["text", "content", "raw_content"]
    limit = args.limit if args.limit and args.limit > 0 else None

    token: str | bool | None
    if args.hf_token:
        token = args.hf_token
    elif os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    else:
        token = None

    out_dir = args.out_dir
    prev_dir = out_dir / "previews"
    ensure_dir(out_dir)
    ensure_dir(prev_dir)

    from datasets import get_dataset_config_names, load_dataset

    configs = set(get_dataset_config_names(args.repo_id, token=token))
    lang_pairs: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for lang in langs:
        resolved = alias_map.get(lang, lang)
        if resolved in configs:
            lang_pairs.append((lang, resolved))
        else:
            skipped.append((lang, resolved))

    if skipped and args.strict_missing:
        sample = ", ".join(sorted(list(configs))[:20])
        detail = ", ".join(f"{src}->{dst}" for src, dst in skipped)
        raise SystemExit(
            f"{args.repo_id} 中找不到以下语言配置（映射后）：{detail}\n"
            f"示例配置：{sample}"
        )

    if skipped:
        detail = ", ".join(f"{src}->{dst}" for src, dst in skipped)
        print(f"警告：以下语言在 {args.repo_id} 中不存在，已跳过：{detail}")

    if not lang_pairs:
        print("警告：没有可导出的语言（全部被跳过）。本次不生成文件。")
        if not args.no_hard_exit:
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0

    for lang, repo_lang in lang_pairs:
        out_path = out_dir / f"fineweb2_pt_{lang}.jsonl"
        prev_path = prev_dir / f"fineweb2_pt_{lang}.preview_{PREVIEW_N}.jsonl"
        print(f"[{lang}] 开始导出（FineWeb 配置: {repo_lang}）-> {out_path}")

        ds = load_dataset(
            args.repo_id,
            repo_lang,
            split=args.split,
            streaming=True,
            token=token,
        )

        written = 0
        preview_lines: list[str] = []
        with open(out_path, "w", encoding="utf-8") as fo:
            for row in ds:
                if limit and written >= limit:
                    break
                if not isinstance(row, dict):
                    row = dict(row)

                text = resolve_text(row, text_keys, args.min_chars)
                if text is None:
                    continue

                rec = {"text": text}
                line = json.dumps(rec, ensure_ascii=False)
                fo.write(line + "\n")
                if written < PREVIEW_N:
                    preview_lines.append(line)
                written += 1
                if written % 10_000 == 0:
                    print(f"[{lang}] ... 已写入 {written} 条")

        prev_path.write_text(
            "\n".join(preview_lines) + ("\n" if preview_lines else ""),
            encoding="utf-8",
        )
        print(f"[{lang}] 完成：{out_path}，共 {written} 条")
        print(f"[{lang}] 预览：{prev_path}")

    # 经验兼容：当前 lowres 环境里，datasets streaming 读 FineWeb-2 后
    # 在解释器退出阶段可能触发 "terminate called without an active exception"。
    # 数据已落盘后采用硬退出可避免返回 134，便于脚本化调用。
    if not args.no_hard_exit:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
