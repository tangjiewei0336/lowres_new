#!/usr/bin/env python3
"""
从 NLLB 镜像流式导出句对，格式为英文指令 + Input + Response 的整块文本（JSONL）。

每条样本一行 JSON：{"text": "Instruction: ...\\nInput: ...\\nResponse: ..."}

与 scripts/prepare/prepare_nllb_for_llamafactory.py 使用相同的拉取逻辑与语言对解析；仅序列化格式不同。

用法：
  conda activate lowres
  python scripts/prepare/prepare_nllb_instruction_input_response_jsonl.py --pairs-config training/ccmatrix_pair_limits.json --export-from-config

输出（默认 training/data/multilingual/nllb_iir/）：
  nllb_iir_<src>__<tgt>.jsonl
  previews/nllb_iir_<src>__<tgt>.preview_50.jsonl

在 LLaMA-Factory 的 dataset_info.json 中可注册为单列 text（按你使用的版本配置 formatting / columns）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PREVIEW_N = 50

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCRIPTS_PARENT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_PARENT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_PARENT))
from flores_lang_zh import english_translation_instruction  # noqa: E402

_ALLENAI_URL = "https://storage.googleapis.com/allennlp-data-bucket/nllb/"
_STATMT_URL = "http://data.statmt.org/cc-matrix/"


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_nllb_lang_pairs() -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, str]]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id="allenai/nllb",
            filename="nllb_lang_pairs.py",
            repo_type="dataset",
        )
    )
    ns: dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), ns)
    nllb = set(ns["NLLB_PAIRS"])
    ccm = set(ns["CCMATRIX_PAIRS"])
    mapping = dict(ns["CCMATRIX_MAPPING"])
    return nllb, ccm, mapping


def resolve_source(
    src: str,
    tgt: str,
    nllb: set[tuple[str, str]],
    ccm: set[tuple[str, str]],
    mp: dict[str, str],
) -> tuple[str, str, str, str] | None:
    if (src, tgt) in nllb:
        return f"{_ALLENAI_URL}{src}-{tgt}.gz", src, tgt, "allenai"
    if (src, tgt) in ccm:
        a, b = mp.get(src), mp.get(tgt)
        if not a or not b:
            return None
        return f"{_STATMT_URL}{a}-{b}.bitextf.tsv.gz", src, tgt, "statmt"
    if (tgt, src) in nllb:
        return f"{_ALLENAI_URL}{tgt}-{src}.gz", tgt, src, "allenai"
    if (tgt, src) in ccm:
        a, b = mp.get(tgt), mp.get(src)
        if not a or not b:
            return None
        return f"{_STATMT_URL}{a}-{b}.bitextf.tsv.gz", tgt, src, "statmt"
    return None


def parse_line_statmt(line: str) -> tuple[str, str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    try:
        float(parts[0])
    except ValueError:
        return None
    t0, t1 = parts[1].strip(), parts[2].strip()
    if not t0 or not t1:
        return None
    return t0, t1


def parse_line_allenai(line: str) -> tuple[str, str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 9:
        return None
    t0, t1 = parts[0].strip(), parts[1].strip()
    if not t0 or not t1:
        return None
    return t0, t1


def map_to_user_pair(
    text_csrc: str,
    text_ctgt: str,
    canon_src: str,
    canon_tgt: str,
    user_src: str,
    user_tgt: str,
) -> tuple[str, str] | None:
    if user_src == canon_src and user_tgt == canon_tgt:
        return text_csrc, text_ctgt
    if user_src == canon_tgt and user_tgt == canon_src:
        return text_ctgt, text_csrc
    return None


def format_sample_block(src_t: str, tgt_t: str, user_src: str, user_tgt: str) -> str:
    instruction = english_translation_instruction(user_src, user_tgt)
    return f"Instruction: {instruction}\nInput: {src_t}\nResponse: {tgt_t}"


def stream_export_pair(
    url: str,
    kind: str,
    canon_src: str,
    canon_tgt: str,
    user_src: str,
    user_tgt: str,
    limit: int | None,
    out_path: Path,
    prev_path: Path,
) -> int:
    preview_lines: list[str] = []
    written = 0

    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; lowres-nllb-iir/1.0)"})
    try:
        resp = urlopen(req, timeout=300)
    except (HTTPError, URLError) as e:
        print(f"下载失败 {user_src}->{user_tgt} url={url!r}: {e}", file=sys.stderr)
        return 0

    try:
        gz = gzip.open(resp, "rt", encoding="utf-8", errors="replace", newline="")
    except Exception as e:
        print(f"打开 gzip 失败 {url!r}: {e}", file=sys.stderr)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for line in gz:
            if limit is not None and written >= limit:
                break
            if kind == "statmt":
                pr = parse_line_statmt(line)
            else:
                pr = parse_line_allenai(line)
            if not pr:
                continue
            t_csrc, t_ctgt = pr
            mapped = map_to_user_pair(t_csrc, t_ctgt, canon_src, canon_tgt, user_src, user_tgt)
            if not mapped:
                continue
            src_t, tgt_t = mapped
            block = format_sample_block(src_t, tgt_t, user_src, user_tgt)
            rec = {"text": block}
            js = json.dumps(rec, ensure_ascii=False)
            fo.write(js + "\n")
            if written < PREVIEW_N:
                preview_lines.append(js)
            written += 1

    try:
        gz.close()
    except Exception:
        pass

    prev_path.parent.mkdir(parents=True, exist_ok=True)
    prev_path.write_text("\n".join(preview_lines) + ("\n" if preview_lines else ""), encoding="utf-8")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="从 NLLB 镜像导出 Instruction/Input/Response 文本块 jsonl")
    ap.add_argument(
        "--pairs-config",
        type=Path,
        default=root() / "training" / "ccmatrix_pair_limits.json",
        help="含 pairs: [{src_lang, tgt_lang, limit?}, ...] 的 JSON",
    )
    ap.add_argument("--export-from-config", action="store_true", help="按 pairs-config 批量导出")
    ap.add_argument("--src-lang", type=str, default=None)
    ap.add_argument("--tgt-lang", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--out-subdir",
        type=str,
        default="multilingual/nllb_iir",
        help="相对于 training/data 的子目录",
    )
    args = ap.parse_args()

    nllb, ccm, mp = load_nllb_lang_pairs()

    data_root = root() / "training" / "data"
    out_dir = data_root / args.out_subdir
    prev_dir = out_dir / "previews"
    ensure_dir(out_dir)
    ensure_dir(prev_dir)

    def run_one(user_src: str, user_tgt: str, lim: int | None) -> None:
        resolved = resolve_source(user_src, user_tgt, nllb, ccm, mp)
        if not resolved:
            print(
                f"跳过 {user_src}->{user_tgt}：不在 NLLB_PAIRS/CCMATRIX_PAIRS 或缺少 CCMATRIX_MAPPING 码",
                file=sys.stderr,
            )
            return
        url, canon_src, canon_tgt, kind = resolved
        out_path = out_dir / f"nllb_iir_{user_src}__{user_tgt}.jsonl"
        prev_path = prev_dir / f"nllb_iir_{user_src}__{user_tgt}.preview_{PREVIEW_N}.jsonl"
        print(f"拉取 [{kind}] {user_src}->{user_tgt} canon=({canon_src},{canon_tgt})")
        n = stream_export_pair(
            url=url,
            kind=kind,
            canon_src=canon_src,
            canon_tgt=canon_tgt,
            user_src=user_src,
            user_tgt=user_tgt,
            limit=lim,
            out_path=out_path,
            prev_path=prev_path,
        )
        print(f"完成 {out_path} 共 {n} 条；预览 {prev_path}")
        if n == 0:
            print(f"警告：{user_src}->{user_tgt} 写出 0 条（URL 不可达、格式不符或 limit=0）", file=sys.stderr)

    if args.export_from_config:
        conf = json.loads(args.pairs_config.read_text(encoding="utf-8"))
        pairs = conf.get("pairs") or []
        if not isinstance(pairs, list) or not pairs:
            raise SystemExit(f"配置文件无 pairs: {args.pairs_config}")
        for it in pairs:
            run_one(str(it["src_lang"]), str(it["tgt_lang"]), int(it["limit"]) if it.get("limit") is not None else None)
        return 0

    if not args.src_lang or not args.tgt_lang:
        ap.error("请提供 --src-lang/--tgt-lang，或使用 --export-from-config")
    run_one(args.src_lang, args.tgt_lang, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
