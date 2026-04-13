#!/usr/bin/env python3
"""
将 ccMatrix（ModelScope MsDataset）导出为 LLaMAFactory 可用的 Alpaca jsonl：
  {"instruction": "...", "input": "<src>", "output": "<tgt>"}

注意：
- ccMatrix 在 ModelScope 上的 dataset_id 需要你在 modelscope_sources.json 里配置正确（目前可能是占位）。
- 不同 ccMatrix 版本字段名可能不同；本脚本提供 --src-key/--tgt-key 覆盖。

用法：
  conda activate lowres
  python scripts/prepare_ccmatrix_for_llamafactory.py --src-lang eng_Latn --tgt-lang zho_Hans --limit 200000
  # 按 evaluation_config.json 生成的语言对配置批量导出（每个语言对各自 limit）
  python scripts/prepare_ccmatrix_for_llamafactory.py --pairs-config training/ccmatrix_pair_limits.json --export-from-config
  # 只统计每个语言对在数据集中能取到多少条（不写出文件）
  python scripts/prepare_ccmatrix_for_llamafactory.py --pairs-config training/ccmatrix_pair_limits.json --count-only

输出：
  training/data/ccmatrix_mt_<src>__<tgt>.jsonl
  training/data/previews/ccmatrix_mt_<src>__<tgt>.preview_50.jsonl

instruction 使用中文语言名（与 prepare_nllb_for_llamafactory 一致，见 scripts/flores_lang_zh.py）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

PREVIEW_N = 50

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from flores_lang_zh import flores_code_to_zh_name  # noqa: E402

# FLORES-200 语言码 -> 常用 ISO639-1（用于 HuggingFace yhavinga/ccmatrix 配置名，如 en-zh）
HF_LANG_MAP: dict[str, str] = {
    "eng_Latn": "en",
    "zho_Hans": "zh",
    "spa_Latn": "es",
    "ind_Latn": "id",
    "vie_Latn": "vi",
    "tha_Thai": "th",
    "tgl_Latn": "tl",  # Tagalog
}


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_ms_rows(ds: Any) -> Iterator[dict[str, Any]]:
    n = len(ds)  # type: ignore[arg-type]
    for i in range(n):
        row = ds[i]
        if not isinstance(row, dict):
            row = dict(row)
        yield row


def iter_hf_rows(ds: Any) -> Iterator[dict[str, Any]]:
    # HF streaming dataset：可直接迭代
    for row in ds:
        if not isinstance(row, dict):
            row = dict(row)
        yield row


def pick_text(row: dict[str, Any], key: str) -> str | None:
    v = row.get(key)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def pick_translation(row: dict[str, Any], src_code: str, tgt_code: str) -> tuple[str | None, str | None]:
    """HuggingFace yhavinga/ccmatrix: row 形如 {'translation': {'en': '...', 'zh': '...'}, ...}"""
    tr = row.get("translation")
    if isinstance(tr, dict):
        s = tr.get(src_code)
        t = tr.get(tgt_code)
        if isinstance(s, str) and isinstance(t, str) and s.strip() and t.strip():
            return s.strip(), t.strip()
    return None, None


def pick_st_ccmatrix(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """sentence-transformers/parallel-sentences-ccmatrix: {'english':..., 'non_english':...}"""
    e = row.get("english")
    n = row.get("non_english")
    if isinstance(e, str) and isinstance(n, str) and e.strip() and n.strip():
        return e.strip(), n.strip()
    return None, None


def _load_pairs_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _export_one_pair(
    ds: Any,
    src_lang: str,
    tgt_lang: str,
    limit: int | None,
    out_dir: Path,
    prev_dir: Path,
    src_key: str | None,
    tgt_key: str | None,
) -> tuple[int, Path, Path]:
    # 字段名猜测：优先使用显式传入，否则按常见命名尝试
    src_candidates = [src_key] if src_key else []
    tgt_candidates = [tgt_key] if tgt_key else []
    src_candidates += [src_lang, "source", "src", "sentence1", "text1"]
    tgt_candidates += [tgt_lang, "target", "tgt", "sentence2", "text2"]
    src_candidates = [x for x in src_candidates if x]
    tgt_candidates = [x for x in tgt_candidates if x]

    out_path = out_dir / f"ccmatrix_mt_{src_lang}__{tgt_lang}.jsonl"
    prev_path = prev_dir / f"ccmatrix_mt_{src_lang}__{tgt_lang}.preview_{PREVIEW_N}.jsonl"

    written = 0
    preview_lines: list[str] = []
    instruction = (
        f"请将以下 {flores_code_to_zh_name(src_lang)} 文本翻译为 "
        f"{flores_code_to_zh_name(tgt_lang)}，只输出译文。"
    )

    # 兼容 MsDataset 与 HF streaming
    row_iter = iter_ms_rows(ds) if hasattr(ds, "__len__") and hasattr(ds, "__getitem__") else iter_hf_rows(ds)

    src_hf = _hf_code(src_lang)
    tgt_hf = _hf_code(tgt_lang)
    st_conf, st_swap = _st_config_for_pair(src_lang, tgt_lang)

    with open(out_path, "w", encoding="utf-8") as fo:
        for row in row_iter:
            if limit and written >= limit:
                break

            # HuggingFace sentence-transformers 版本：优先从 english/non_english 取
            src, tgt = (None, None)
            if st_conf:
                e, n = pick_st_ccmatrix(row)
                if e and n:
                    src, tgt = (n, e) if st_swap else (e, n)

            # 其它情况：从 translation 或列名猜测
            if not src or not tgt:
                src, tgt = pick_translation(row, src_hf, tgt_hf)

            if not src or not tgt:
                src = None
                for k in src_candidates:
                    src = pick_text(row, k)
                    if src:
                        break
                tgt = None
                for k in tgt_candidates:
                    tgt = pick_text(row, k)
                    if tgt:
                        break

            if not src or not tgt:
                continue

            rec = {"instruction": instruction, "input": src, "output": tgt}
            line = json.dumps(rec, ensure_ascii=False)
            fo.write(line + "\n")

            if written < PREVIEW_N:
                preview_lines.append(line)
            written += 1

    prev_path.write_text("\n".join(preview_lines) + ("\n" if preview_lines else ""), encoding="utf-8")
    return written, out_path, prev_path


def _count_one_pair(
    ds: Any,
    src_lang: str,
    tgt_lang: str,
    src_key: str | None,
    tgt_key: str | None,
) -> int:
    src_candidates = [src_key] if src_key else []
    tgt_candidates = [tgt_key] if tgt_key else []
    src_candidates += [src_lang, "source", "src", "sentence1", "text1"]
    tgt_candidates += [tgt_lang, "target", "tgt", "sentence2", "text2"]
    src_candidates = [x for x in src_candidates if x]
    tgt_candidates = [x for x in tgt_candidates if x]

    c = 0
    row_iter = iter_ms_rows(ds) if hasattr(ds, "__len__") and hasattr(ds, "__getitem__") else iter_hf_rows(ds)
    src_hf = _hf_code(src_lang)
    tgt_hf = _hf_code(tgt_lang)
    st_conf, st_swap = _st_config_for_pair(src_lang, tgt_lang)
    for row in row_iter:
        src, tgt = (None, None)
        if st_conf:
            e, n = pick_st_ccmatrix(row)
            if e and n:
                src, tgt = (n, e) if st_swap else (e, n)
        if not src or not tgt:
            src, tgt = pick_translation(row, src_hf, tgt_hf)
        if not src or not tgt:
            src = None
            for k in src_candidates:
                src = pick_text(row, k)
                if src:
                    break
            if not src:
                continue
            tgt = None
            for k in tgt_candidates:
                tgt = pick_text(row, k)
                if tgt:
                    break
            if not tgt:
                continue
        c += 1
    return c


def _hf_code(lang: str) -> str:
    if lang in HF_LANG_MAP:
        return HF_LANG_MAP[lang]
    # 兜底：取下划线前半段（例如 xxx_Latn -> xxx）
    return lang.split("_", 1)[0]


def _hf_config_name(src_lang: str, tgt_lang: str) -> str:
    return f"{_hf_code(src_lang)}-{_hf_code(tgt_lang)}"


def _st_config_for_pair(src_lang: str, tgt_lang: str) -> tuple[str | None, bool]:
    """
    sentence-transformers/parallel-sentences-ccmatrix 仅提供 en-xx 配置。
    返回 (config_name, swap)：
    - swap=False：输出为 src->tgt，取 english->non_english
    - swap=True：输出为 src->tgt，但数据集给的是 english/non_english，需要反过来用 non_english->english
    """
    src = _hf_code(src_lang)
    tgt = _hf_code(tgt_lang)
    if src == "en" and tgt != "en":
        return f"en-{tgt}", False
    if tgt == "en" and src != "en":
        return f"en-{src}", True
    return None, False


def _load_ccmatrix_auto(
    dataset_id: str,
    split: str,
    subset: str | None,
    backend: str,
    src_lang: str | None,
    tgt_lang: str | None,
) -> tuple[str, Any]:
    """
    返回 (backend_used, dataset_obj)
    - modelscope: MsDataset.load(dataset_id, ...)
    - huggingface: datasets.load_dataset('sentence-transformers/parallel-sentences-ccmatrix', 'en-xx', streaming=True)
    """
    be = backend.lower()
    if be not in ("auto", "modelscope", "huggingface", "hf"):
        raise ValueError(f"非法 backend: {backend}")

    # 1) 尝试 ModelScope
    if be in ("auto", "modelscope"):
        try:
            from modelscope.msdatasets import MsDataset

            load_kw: dict[str, Any] = {"split": split, "trust_remote_code": True}
            if subset:
                load_kw["subset_name"] = subset
            ds = MsDataset.load(dataset_id, **load_kw)
            return "modelscope", ds
        except Exception as e:
            if be == "modelscope":
                raise
            print(f"ModelScope 加载 ccMatrix 失败，改用 HuggingFace：{e}")

    # 2) HuggingFace 兜底（无脚本版本）
    from datasets import load_dataset

    if not src_lang or not tgt_lang:
        raise ValueError("使用 HuggingFace backend 时必须提供 src_lang/tgt_lang（用于确定配置名）")
    conf, swap = _st_config_for_pair(src_lang, tgt_lang)
    if not conf:
        raise ValueError(
            "HuggingFace 兜底使用 sentence-transformers/parallel-sentences-ccmatrix（仅 en-xx）。"
            f"当前语言对不包含英语，无法加载：{src_lang}->{tgt_lang}"
        )
    ds = load_dataset("sentence-transformers/parallel-sentences-ccmatrix", conf, split="train", streaming=True)
    # swap 信息通过给调用方自行处理（此处只返回数据集本体）
    return "huggingface", ds


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ccMatrix 为 LLaMAFactory Alpaca jsonl")
    ap.add_argument("--dataset-id", type=str, default=None, help="覆盖 ModelScope ccMatrix dataset_id")
    ap.add_argument("--subset", type=str, default=None, help="subset_name（如需要）")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="数据来源：auto/modelscope/huggingface。auto 会先尝试 ModelScope，失败则用 HuggingFace yhavinga/ccmatrix。",
    )
    ap.add_argument("--src-lang", type=str, default=None)
    ap.add_argument("--tgt-lang", type=str, default=None)
    ap.add_argument("--src-key", type=str, default=None, help="源句字段名（默认尝试 src_lang / source / src）")
    ap.add_argument("--tgt-key", type=str, default=None, help="目标句字段名（默认尝试 tgt_lang / target / tgt）")
    ap.add_argument("--limit", type=int, default=None, help="最多导出条数（默认全量）")
    ap.add_argument("--seed", type=int, default=42, help="当需要采样时使用（目前仅保留接口）")
    ap.add_argument(
        "--pairs-config",
        type=Path,
        default=None,
        help="语言对批量配置文件（见 training/ccmatrix_pair_limits.json）",
    )
    ap.add_argument("--export-from-config", action="store_true", help="按 --pairs-config 批量导出每个语言对")
    ap.add_argument("--count-only", action="store_true", help="按 --pairs-config 只统计每个语言对可用条目数，不写文件")
    args = ap.parse_args()

    if not args.export_from_config and not args.count_only:
        if not args.src_lang or not args.tgt_lang:
            ap.error("必须提供 --src-lang 与 --tgt-lang，或使用 --pairs-config + (--export-from-config/--count-only)")

    os.environ.setdefault("MODELSCOPE_CACHE", str(root() / "datasets" / "cache" / "modelscope"))
    with open(root() / "modelscope_sources.json", encoding="utf-8") as f:
        cfg = json.load(f)
    dataset_id = args.dataset_id or cfg["datasets"]["ccmatrix"]["dataset_id"]

    out_dir = root() / "training" / "data"
    prev_dir = out_dir / "previews"
    ensure_dir(out_dir)
    ensure_dir(prev_dir)

    if args.export_from_config or args.count_only:
        cfg_path = args.pairs_config or (root() / "training" / "ccmatrix_pair_limits.json")
        conf = _load_pairs_config(cfg_path)
        pairs = conf.get("pairs") or []
        if not isinstance(pairs, list) or not pairs:
            raise SystemExit(f"配置文件无 pairs: {cfg_path}")

        # 配置模式下：允许不传 --src-lang/--tgt-lang。
        # auto/backend=modelscope：尝试一次 MsDataset（同一份 ds 可复用）。
        # auto/backend=huggingface：按每个语言对分别 load_dataset（不同 config）。
        backend_used: str = "auto"
        ds: Any | None = None
        if args.backend.lower() in ("auto", "modelscope"):
            try:
                backend_used, ds = _load_ccmatrix_auto(
                    dataset_id=dataset_id,
                    split=args.split,
                    subset=args.subset,
                    backend="modelscope",
                    src_lang="eng_Latn",  # 仅用于通过签名；ModelScope 不依赖该值
                    tgt_lang="zho_Hans",
                )
            except Exception as e:
                if args.backend.lower() == "modelscope":
                    raise
                print(f"ModelScope 加载 ccMatrix 失败，改用 HuggingFace：{e}")
                backend_used, ds = "huggingface", None
        else:
            backend_used, ds = "huggingface", None

        print(f"ccMatrix backend={backend_used}")

        for it in pairs:
            src_lang = str(it["src_lang"])
            tgt_lang = str(it["tgt_lang"])
            lim = it.get("limit")
            lim_i = int(lim) if lim is not None else None

            # HuggingFace：需要按每个语言对加载对应 config（避免全量读超大数据）
            ds_pair: Any
            if backend_used == "huggingface":
                try:
                    _, ds_pair = _load_ccmatrix_auto(
                        dataset_id=dataset_id,
                        split=args.split,
                        subset=args.subset,
                        backend="huggingface",
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                    )
                except Exception as e:
                    print(f"跳过 {src_lang}->{tgt_lang}: {e}")
                    continue
            else:
                assert ds is not None
                ds_pair = ds

            if args.count_only:
                c = _count_one_pair(ds_pair, src_lang, tgt_lang, args.src_key, args.tgt_key)
                print(f"COUNT {src_lang}->{tgt_lang}: {c}")
            else:
                written, out_path, prev_path = _export_one_pair(
                    ds=ds_pair,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    limit=lim_i,
                    out_dir=out_dir,
                    prev_dir=prev_dir,
                    src_key=args.src_key,
                    tgt_key=args.tgt_key,
                )
                print(f"完成：{out_path} 写出 {written} 条；预览 {prev_path}")
                if written == 0:
                    print(
                        f"警告：{src_lang}->{tgt_lang} 写出 0 条。通常是 dataset_id/字段名不匹配；可用 --src-key/--tgt-key 指定列名。",
                    )
        return 0

    # 单对模式：数据加载：支持 ModelScope + HuggingFace 兜底
    backend_used, ds = _load_ccmatrix_auto(
        dataset_id=dataset_id,
        split=args.split,
        subset=args.subset,
        backend=args.backend,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
    print(f"ccMatrix backend={backend_used}")

    assert args.src_lang and args.tgt_lang
    written, out_path, prev_path = _export_one_pair(
        ds=ds,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        limit=args.limit,
        out_dir=out_dir,
        prev_dir=prev_dir,
        src_key=args.src_key,
        tgt_key=args.tgt_key,
    )
    print(f"完成：{out_path} 写出 {written} 条；预览 {prev_path}")
    if written == 0:
        print("警告：写出 0 条。通常是 dataset_id/字段名不匹配。你可以用 --src-key/--tgt-key 指定列名。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

