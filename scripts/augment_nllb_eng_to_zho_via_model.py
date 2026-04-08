#!/usr/bin/env python3
"""
用模型把 NLLB 的「英语侧」翻译成中文（zho_Hans），生成新的中-多语言伪平行数据，双向都产出。

输入（已存在于本仓库的导出格式，Alpaca jsonl）：
  training/data/nllb_mt_eng_Latn__<tgt>.jsonl   # input=英语, output=<tgt>
  training/data/nllb_mt_<tgt>__eng_Latn.jsonl   # input=<tgt>, output=英语

做法：
  - 对 eng->tgt 文件：把 input(英语) 翻成中文，得到 zho_Hans->tgt
  - 对 tgt->eng 文件：把 output(英语) 翻成中文，得到 tgt->zho_Hans

输出（新增文件，不覆盖原始 NLLB 文件）：
  training/data/nllb_aug_zho_Hans__<tgt>.jsonl
  training/data/nllb_aug_<tgt>__zho_Hans.jsonl
并生成 previews/ 下的 preview_50.jsonl。

翻译方式：
  通过 vLLM 的 OpenAI 兼容接口调用“模型本身”，默认读取环境变量：
    OPENAI_API_BASE=http://127.0.0.1:8000/v1
    OPENAI_API_KEY=EMPTY
    SERVED_MODEL_NAME=<与你的 vLLM --served-model-name 一致>
  其中 Qwen3 可通过 --model-family qwen3 自动关闭 enable_thinking。

用法示例：
  conda activate lowres
  # 先启动 vLLM 服务（served-model-name 与下方一致）
  python scripts/augment_nllb_eng_to_zho_via_model.py --tgt-langs spa_Latn ind_Latn vie_Latn tha_Thai tgl_Latn jpn_Jpan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

PREVIEW_N = 50
ZHO = "zho_Hans"
ENG = "eng_Latn"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def mt_user_content(src_lang: str, tgt_lang: str, text: str) -> str:
    return f"Translate from {src_lang} to {tgt_lang}. Output only the translation, no explanations.\n\n{text}"


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    fam = (model_family or "").lower()
    if fam in ("qwen3", "qwen", "qwen3-4b"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def call_translate(
    client: OpenAI,
    model: str,
    src_lang: str,
    tgt_lang: str,
    text: str,
    *,
    max_tokens: int,
    model_family: str,
) -> str:
    extra = build_extra_body(model_family)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional machine translation engine."},
            {"role": "user", "content": mt_user_content(src_lang, tgt_lang, text)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra:
        kwargs["extra_body"] = extra
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_instruction(src: str, tgt: str) -> str:
    return f"请将以下 {src} 文本翻译为 {tgt}，只输出译文。"


def preview_path(out_path: Path) -> Path:
    prev_dir = out_path.parent / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    return prev_dir / f"{out_path.stem}.preview_{PREVIEW_N}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="用模型把 NLLB 英语侧翻译成中文，生成中-多语言伪平行数据")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=root() / "training" / "data",
        help="包含 nllb_mt_*.jsonl 的目录",
    )
    ap.add_argument(
        "--tgt-langs",
        nargs="+",
        required=True,
        help="需要增强的目标语言（如 spa_Latn ind_Latn ...）。会读取 eng<->tgt 两个方向的 nllb_mt 文件。",
    )
    ap.add_argument(
        "--served-model-name",
        type=str,
        default=os.environ.get("SERVED_MODEL_NAME", "base"),
        help="与 vLLM --served-model-name 一致",
    )
    ap.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
        help="vLLM OpenAI API base，含 /v1",
    )
    ap.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    ap.add_argument(
        "--model-family",
        type=str,
        default=os.environ.get("EVAL_MODEL_FAMILY", "generic"),
        help="qwen3 时关闭思考模式；其它模型可用 generic",
    )
    ap.add_argument("--max-workers", type=int, default=16, help="并发请求数")
    ap.add_argument("--max-tokens", type=int, default=512, help="单条翻译最大输出 tokens")
    ap.add_argument("--limit", type=int, default=0, help="每个方向最多处理多少条（0 不限制）")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="若输出文件已存在，则跳过已生成过的样本（按 input 文本去重）。",
    )
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    for tgt in args.tgt_langs:
        if tgt == ENG or tgt == ZHO:
            print(f"跳过不合理 tgt: {tgt}", file=sys.stderr)
            continue

        # 1) eng->tgt 生成 zho->tgt（翻译 input）
        in_a = data_dir / f"nllb_mt_{ENG}__{tgt}.jsonl"
        out_a = data_dir / f"nllb_aug_{ZHO}__{tgt}.jsonl"
        rows_a = read_jsonl(in_a)
        if not rows_a:
            print(f"[{tgt}] 缺少或为空: {in_a}，跳过该方向。", file=sys.stderr)
        else:
            existing_inputs: set[str] = set()
            out_rows: list[dict[str, Any]] = []
            if args.resume and out_a.is_file():
                for r in read_jsonl(out_a):
                    if isinstance(r.get("input"), str):
                        existing_inputs.add(r["input"])
                out_rows = read_jsonl(out_a)

            todo: list[tuple[int, dict[str, Any]]] = []
            for i, r in enumerate(rows_a):
                if args.limit and i >= args.limit:
                    break
                eng_text = str(r.get("input", "")).strip()
                tgt_text = str(r.get("output", "")).strip()
                if not eng_text or not tgt_text:
                    continue
                if eng_text in existing_inputs:
                    continue
                todo.append((i, r))

            print(f"[{tgt}] {ENG}->{tgt} 生成 {ZHO}->{tgt}：待翻译 {len(todo)} 条")

            def _one(pair: tuple[int, dict[str, Any]]) -> dict[str, Any]:
                _, r0 = pair
                eng_text = str(r0["input"]).strip()
                zh_text = call_translate(
                    client,
                    args.served_model_name,
                    ENG,
                    ZHO,
                    eng_text,
                    max_tokens=int(args.max_tokens),
                    model_family=str(args.model_family),
                )
                return {
                    "instruction": make_instruction(ZHO, tgt),
                    "input": zh_text,
                    "output": str(r0["output"]).strip(),
                    "meta": {
                        "source": "nllb_aug_eng_to_zho_via_model",
                        "from_file": in_a.name,
                        "orig_eng": eng_text,
                    },
                }

            new_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
                futs = [ex.submit(_one, x) for x in todo]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{ZHO}->{tgt}"):
                    new_rows.append(fut.result())

            # 稳定排序：按原索引的提交顺序不可得，这里按 orig_eng 排一下保证复现性
            new_rows.sort(key=lambda x: str(x.get("meta", {}).get("orig_eng", "")))
            out_rows.extend(new_rows)
            write_jsonl(out_a, out_rows)
            prev = preview_path(out_a)
            write_jsonl(prev, out_rows[:PREVIEW_N])
            print(f"[{tgt}] 写出: {out_a} 共 {len(out_rows)}；预览 {prev}")

        # 2) tgt->eng 生成 tgt->zho（翻译 output）
        in_b = data_dir / f"nllb_mt_{tgt}__{ENG}.jsonl"
        out_b = data_dir / f"nllb_aug_{tgt}__{ZHO}.jsonl"
        rows_b = read_jsonl(in_b)
        if not rows_b:
            print(f"[{tgt}] 缺少或为空: {in_b}，跳过该方向。", file=sys.stderr)
            continue

        existing_inputs_b: set[str] = set()
        out_rows_b: list[dict[str, Any]] = []
        if args.resume and out_b.is_file():
            for r in read_jsonl(out_b):
                if isinstance(r.get("input"), str):
                    existing_inputs_b.add(r["input"])
            out_rows_b = read_jsonl(out_b)

        todo_b: list[tuple[int, dict[str, Any]]] = []
        for i, r in enumerate(rows_b):
            if args.limit and i >= args.limit:
                break
            src_text = str(r.get("input", "")).strip()
            eng_text = str(r.get("output", "")).strip()
            if not src_text or not eng_text:
                continue
            if src_text in existing_inputs_b:
                continue
            todo_b.append((i, r))

        print(f"[{tgt}] {tgt}->{ENG} 生成 {tgt}->{ZHO}：待翻译 {len(todo_b)} 条")

        def _one_b(pair: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            _, r0 = pair
            src_text = str(r0["input"]).strip()
            eng_text = str(r0["output"]).strip()
            zh_text = call_translate(
                client,
                args.served_model_name,
                ENG,
                ZHO,
                eng_text,
                max_tokens=int(args.max_tokens),
                model_family=str(args.model_family),
            )
            return {
                "instruction": make_instruction(tgt, ZHO),
                "input": src_text,
                "output": zh_text,
                "meta": {
                    "source": "nllb_aug_eng_to_zho_via_model",
                    "from_file": in_b.name,
                    "orig_eng": eng_text,
                },
            }

        new_rows_b: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
            futs = [ex.submit(_one_b, x) for x in todo_b]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{tgt}->{ZHO}"):
                new_rows_b.append(fut.result())

        new_rows_b.sort(key=lambda x: str(x.get("meta", {}).get("orig_eng", "")))
        out_rows_b.extend(new_rows_b)
        write_jsonl(out_b, out_rows_b)
        prev_b = preview_path(out_b)
        write_jsonl(prev_b, out_rows_b[:PREVIEW_N])
        print(f"[{tgt}] 写出: {out_b} 共 {len(out_rows_b)}；预览 {prev_b}")

        # 轻微延迟，避免过快触发服务端限流（可按需删）
        time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

