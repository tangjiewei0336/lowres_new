#!/usr/bin/env python3
"""
从 NLLB 句级 Alpaca jsonl（prepare_nllb_for_llamafactory.py 导出）构造「draft → document refine」训练样本。

「两部分」数据说明（本脚本不会把两类数据塞进同一个文件）：
  1) 句级翻译数据：training/data/multilingual/nllb/nllb_mt_<src>__<tgt>.jsonl（仅作本脚本的输入）；
  2) 篇章 refine 数据：本脚本写出到 training/data/draft_refine/nllb/。
  两阶段训练时：句级 SFT 用 (1)，篇章 refine 用 (2)，在 dataset_info 里分别注册即可。

伪段落：连续 K 句（默认 K=4）拼成一段。
  - Source paragraph：源语句子按行拼接；
  - Initial translation（默认 --draft-source model）：对 K 句源语**分别**调用同一翻译模型各译一句，
    再按原顺序用换行拼成一段（句级独立翻译、无跨句上下文），模拟真实 draft；
  - output：并行语料中的参考译文段落（K 句参考译文按行拼接），作为改进目标。

离线备选：--draft-source shuffle_tgt（打乱参考译文句序）或 none（初稿=参考，仅调试）。

用法（conda env lowres，需先启动 vLLM，默认 draft-source=model）：
  export SERVED_MODEL_NAME=qwen3-4b
  python scripts/build_nllb_draft_refine_jsonl.py \\
    --in-dir training/data/multilingual/nllb --export-all-pair-files

离线（无需 API）：
  python scripts/build_nllb_draft_refine_jsonl.py --in-dir ... --export-all-pair-files \\
    --draft-source shuffle_tgt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

PREVIEW_N = 50

_DEFAULT_INSTRUCTION = "Improve the translation using document context."

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from augment_nllb_eng_to_zho_via_model import call_translate  # noqa: E402


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_alpaca_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        return None
    src = (o.get("input") or "").strip()
    tgt = (o.get("output") or "").strip()
    if not src or not tgt:
        return None
    return src, tgt


def paragraph_join(lines: list[str]) -> str:
    return "\n".join(lines)


def parse_pair_from_nllb_mt_path(path: Path) -> tuple[str, str]:
    """从 nllb_mt_<src>__<tgt>.jsonl 解析 FLORES 语言对（与句级文件命名一致）。"""
    name = path.name
    if not name.startswith("nllb_mt_") or not name.endswith(".jsonl"):
        raise ValueError(f"无法从文件名解析语言对（需 nllb_mt_*.jsonl）: {path}")
    stem = name[len("nllb_mt_") : -len(".jsonl")]
    if "__" not in stem:
        raise ValueError(f"文件名中缺少 __ 语言分隔: {path}")
    a, b = stem.split("__", 1)
    return a.strip(), b.strip()


def make_initial_shuffle(
    tgt_lines: list[str],
    noise: str,
    rng: random.Random,
) -> tuple[str, list[str]] | None:
    if noise == "none":
        return paragraph_join(tgt_lines), tgt_lines
    if noise == "shuffle_tgt":
        n = len(tgt_lines)
        if n < 2:
            return None
        idx = list(range(n))
        for _ in range(64):
            rng.shuffle(idx)
            if idx != list(range(n)):
                break
        if idx == list(range(n)):
            return None
        perm = [tgt_lines[i] for i in idx]
        return paragraph_join(perm), tgt_lines
    raise ValueError(f"unknown noise: {noise}")


def format_record(
    instruction: str,
    src_lines: list[str],
    initial_text: str,
    ordered_tgt_lines: list[str],
) -> dict[str, str]:
    src_block = paragraph_join(src_lines)
    user_input = (
        f"Source paragraph:\n{src_block}\n\n"
        f"Initial translation:\n{initial_text}\n"
    )
    return {
        "instruction": instruction,
        "input": user_input,
        "output": paragraph_join(ordered_tgt_lines),
    }


def translate_sentence_batch(
    client: OpenAI,
    model: str,
    src_lang: str,
    tgt_lang: str,
    sentences: list[str],
    *,
    max_tokens: int,
    model_family: str,
    max_workers: int,
) -> list[str] | None:
    """对多句源文各调用一次句级翻译，顺序与 sentences 一致。"""

    def one(s: str) -> str:
        return call_translate(
            client,
            model,
            src_lang,
            tgt_lang,
            s,
            max_tokens=max_tokens,
            model_family=model_family,
        )

    n = len(sentences)
    if n == 0:
        return None
    w = max(1, min(n, max_workers))
    try:
        if w == 1:
            return [one(s) for s in sentences]
        with ThreadPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(one, s) for s in sentences]
            return [f.result() for f in futs]
    except Exception as e:
        print(f"句译失败，跳过该段: {e}", file=sys.stderr)
        return None


def stream_process_file(
    in_path: Path,
    out_path: Path,
    prev_path: Path,
    sentences_per_doc: int,
    instruction: str,
    draft_source: str,
    noise: str,
    rng: random.Random,
    limit: int | None,
    *,
    client: OpenAI | None,
    model: str | None,
    src_lang: str | None,
    tgt_lang: str | None,
    max_tokens: int,
    model_family: str,
    translate_workers: int,
) -> int:
    preview_lines: list[str] = []
    written = 0
    buf: list[tuple[str, str]] = []

    ensure_dir(out_path.parent)
    ensure_dir(prev_path.parent)

    use_model = draft_source == "model"
    if use_model:
        if client is None or not model or not src_lang or not tgt_lang:
            raise SystemExit("draft-source=model 时需要有效的 OpenAI client、模型名与语言对")

    def process_batch(pairs: list[tuple[str, str]]) -> None:
        nonlocal written
        if limit is not None and written >= limit:
            return
        src_lines = [p[0] for p in pairs]
        tgt_lines = [p[1] for p in pairs]

        if use_model:
            drafts = translate_sentence_batch(
                client,  # type: ignore[arg-type]
                model,  # type: ignore[arg-type]
                src_lang,  # type: ignore[arg-type]
                tgt_lang,  # type: ignore[arg-type]
                src_lines,
                max_tokens=max_tokens,
                model_family=model_family,
                max_workers=translate_workers,
            )
            if drafts is None:
                return
            initial_text = paragraph_join(drafts)
        elif draft_source == "none":
            initial_text = paragraph_join(tgt_lines)
        else:
            init = make_initial_shuffle(tgt_lines, noise, rng)
            if init is None:
                return
            initial_text, ordered = init
            tgt_lines = ordered

        rec = format_record(instruction, src_lines, initial_text, tgt_lines)
        js = json.dumps(rec, ensure_ascii=False)
        fo.write(js + "\n")
        if written < PREVIEW_N:
            preview_lines.append(js)
        written += 1

    with open(in_path, encoding="utf-8") as fi, open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            if limit is not None and written >= limit:
                break
            pr = load_alpaca_line(line)
            if not pr:
                continue
            buf.append(pr)
            if len(buf) < sentences_per_doc:
                continue
            batch = buf[:sentences_per_doc]
            buf = buf[sentences_per_doc:]
            process_batch(batch)

        if buf and (limit is None or written < limit):
            process_batch(buf)

    prev_path.write_text("\n".join(preview_lines) + ("\n" if preview_lines else ""), encoding="utf-8")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="NLLB 句级 jsonl → draft/refine 篇章样本")
    ap.add_argument("--in-file", type=Path, default=None, help="单个句级 nllb_mt_*.jsonl")
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=root() / "training" / "data" / "multilingual" / "nllb",
        help="句级 jsonl 目录（与 --export-all-pair-files 联用）",
    )
    ap.add_argument(
        "--export-all-pair-files",
        action="store_true",
        help="处理 in-dir 下全部 nllb_mt_*.jsonl（排除 *_all.jsonl）",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root() / "training" / "data" / "draft_refine" / "nllb",
        help="refine 样本输出目录",
    )
    ap.add_argument("--sentences-per-doc", type=int, default=4, help="每个伪段落包含的句对数量")
    ap.add_argument(
        "--draft-source",
        choices=("model", "shuffle_tgt", "none"),
        default="model",
        help="初稿来源：model=每句源文独立机翻后拼接；shuffle_tgt/none=离线见文档",
    )
    ap.add_argument(
        "--noise",
        choices=("shuffle_tgt", "none"),
        default="shuffle_tgt",
        help="仅 draft-source=shuffle_tgt 时有效",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instruction", type=str, default=_DEFAULT_INSTRUCTION)
    ap.add_argument("--limit", type=int, default=None, help="每个输入文件最多写出多少条 refine 样本")
    ap.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
        help="vLLM OpenAI 兼容 API（draft-source=model）",
    )
    ap.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    ap.add_argument(
        "--served-model-name",
        type=str,
        default=os.environ.get("SERVED_MODEL_NAME"),
        help="与 vLLM --served-model-name 一致（draft-source=model 必填，可用环境变量）",
    )
    ap.add_argument(
        "--model-family",
        type=str,
        default=os.environ.get("EVAL_MODEL_FAMILY", "qwen3"),
        help="qwen3 时关闭 enable_thinking（与 augment_nllb 一致）",
    )
    ap.add_argument("--max-tokens", type=int, default=512, help="单句句译最大 tokens")
    ap.add_argument(
        "--translate-workers",
        type=int,
        default=4,
        help="同一伪段落内并发翻译句子的线程数（≤ 句数）",
    )
    args = ap.parse_args()

    client: OpenAI | None = None
    model: str | None = args.served_model_name
    if args.draft_source == "model":
        if not model or not str(model).strip():
            print(
                "draft-source=model 需要 --served-model-name 或环境变量 SERVED_MODEL_NAME",
                file=sys.stderr,
            )
            return 1
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        model = str(model).strip()

    def run_one(in_path: Path) -> int:
        src_lang: str | None = None
        tgt_lang: str | None = None
        if args.draft_source == "model":
            src_lang, tgt_lang = parse_pair_from_nllb_mt_path(in_path)
        name = in_path.name.replace("nllb_mt_", "nllb_draft_refine_", 1)
        out_path = args.out_dir / name
        prev_path = args.out_dir / "previews" / f"{out_path.stem}.preview_{PREVIEW_N}.jsonl"
        rng = random.Random(args.seed ^ hash(in_path.name))
        return stream_process_file(
            in_path=in_path,
            out_path=out_path,
            prev_path=prev_path,
            sentences_per_doc=args.sentences_per_doc,
            instruction=args.instruction,
            draft_source=args.draft_source,
            noise=args.noise,
            rng=rng,
            limit=args.limit,
            client=client,
            model=model,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_tokens=args.max_tokens,
            model_family=args.model_family,
            translate_workers=args.translate_workers,
        )

    if args.export_all_pair_files:
        files = sorted(
            p
            for p in args.in_dir.glob("nllb_mt_*.jsonl")
            if "_all" not in p.stem and p.name != "nllb_mt_all.jsonl"
        )
        if not files:
            print(f"未找到 {args.in_dir}/nllb_mt_*.jsonl", file=sys.stderr)
            return 1
        total = 0
        for p in files:
            n = run_one(p)
            out_p = args.out_dir / p.name.replace("nllb_mt_", "nllb_draft_refine_", 1)
            print(f"完成 {p.name} → {out_p} 共 {n} 条")
            total += n
        print(f"合计 {total} 条")
        return 0

    if not args.in_file:
        ap.error("请指定 --in-file，或使用 --export-all-pair-files")
    n = run_one(args.in_file)
    out_path = args.out_dir / args.in_file.name.replace("nllb_mt_", "nllb_draft_refine_", 1)
    prev_path = args.out_dir / "previews" / f"{out_path.stem}.preview_{PREVIEW_N}.jsonl"
    print(f"完成 {out_path} 共 {n} 条；预览 {prev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
