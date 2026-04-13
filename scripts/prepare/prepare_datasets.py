#!/usr/bin/env python3
"""
从 ModelScope 下载并规范化 FLORES + NTREX 到 datasets/processed/，并生成各文件前 50 条预览。
依赖 conda 环境 lowres。环境变量 MODELSCOPE_CACHE 可选。
"""
from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Iterator

PREVIEW_N = 50

# facebook/flores 等数据集含远程脚本，须显式信任（与 HF datasets 一致）
_MS_TRUST_REMOTE = True


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_ms_rows(ds: Any) -> Iterator[dict[str, Any]]:
    n = len(ds)  # type: ignore[arg-type]
    for i in range(n):
        row = ds[i]
        if not isinstance(row, dict):
            row = dict(row)
        yield row


def norm_sentence_col(row: dict[str, Any]) -> str | None:
    for k in ("sentence", "text", "Sentence", "TEXT"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def norm_id_col(row: dict[str, Any]) -> str:
    for k in ("id", "ID", "sentence_id", "seg_id"):
        v = row.get(k)
        if v is not None:
            return str(v)
    return ""


def load_flores_lang_table(dataset_id: str, lang: str, split: str) -> dict[str, str]:
    from modelscope.msdatasets import MsDataset

    ds = MsDataset.load(
        dataset_id, subset_name=lang, split=split, trust_remote_code=_MS_TRUST_REMOTE
    )
    out: dict[str, str] = {}
    for row in iter_ms_rows(ds):
        sid = norm_id_col(row)
        sent = norm_sentence_col(row)
        if sid and sent:
            out[sid] = sent
    if not out:
        raise RuntimeError(f"FLORES 解析为空: {dataset_id} {lang} {split}")
    return out


def expand_pairs(groups: list[list[str]], bidirectional: bool) -> list[tuple[str, str]]:
    """
    配对规则：从任意两组中各取一个语言，做笛卡尔积（不是组内两两组合）。
    例如 groups=[[A,B],[C,D]] -> A<->C, A<->D, B<->C, B<->D（若 bidirectional=True 则双向）。
    """
    pairs: list[tuple[str, str]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            if not g1 or not g2:
                continue
            for a, b in product(g1, g2):
                pairs.append((a, b))
                if bidirectional:
                    pairs.append((b, a))
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_preview(src_path: Path, preview_path: Path, n: int = PREVIEW_N) -> None:
    lines: list[str] = []
    with open(src_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.rstrip("\n"))
    ensure_dir(preview_path.parent)
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def build_flores_parallel(
    dataset_id: str, split: str, src: str, tgt: str, src_tbl: dict[str, str], tgt_tbl: dict[str, str]
) -> list[dict[str, Any]]:
    common = sorted(set(src_tbl.keys()) & set(tgt_tbl.keys()))
    rows: list[dict[str, Any]] = []
    for sid in common:
        rows.append(
            {
                "dataset": "flores",
                "split": split,
                "src_lang": src,
                "tgt_lang": tgt,
                "sample_id": sid,
                "source_text": src_tbl[sid],
                "reference_text": tgt_tbl[sid],
            }
        )
    return rows


def _find_eng_value(row: dict[str, Any]) -> str | None:
    for k, v in row.items():
        if not isinstance(v, str):
            continue
        lk = str(k).lower()
        if lk in ("eng", "english", "en", "eng_latn", "source", "src"):
            if v.strip():
                return v.strip()
    return None


def _pick_tgt_text(row: dict[str, Any], tgt_key: str) -> str | None:
    candidates = [tgt_key, tgt_key.replace("_", "-"), tgt_key.split("_")[0]]
    for c in candidates:
        if c in row and isinstance(row[c], str) and row[c].strip():
            return row[c].strip()
    # 模糊：列名以 tgt 的语言码开头
    prefix = tgt_key.split("_")[0].lower()
    for k, v in row.items():
        if not isinstance(v, str) or not v.strip():
            continue
        if str(k).lower().startswith(prefix):
            return v.strip()
    return None


def try_build_ntrex_rows(
    dataset_id: str,
    split: str | None,
    eng_flores: str,
    tgt_flores: str,
    ntrex_tgt_key: str,
) -> list[dict[str, Any]] | None:
    from modelscope.msdatasets import MsDataset

    kwargs: dict[str, Any] = {}
    if split:
        kwargs["split"] = split
    try:
        ds = MsDataset.load(dataset_id, trust_remote_code=_MS_TRUST_REMOTE, **kwargs)
    except Exception:
        return None
    rows_out: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_ms_rows(ds)):
        eng = _find_eng_value(row)
        if eng is None:
            # 有些表用列名即语言码
            v = row.get("eng_Latn") or row.get("eng")
            if isinstance(v, str) and v.strip():
                eng = v.strip()
        tgt_text = _pick_tgt_text(row, ntrex_tgt_key)
        if eng and tgt_text:
            rows_out.append(
                {
                    "dataset": "ntrex",
                    "split": split or "default",
                    "src_lang": eng_flores,
                    "tgt_lang": tgt_flores,
                    "sample_id": f"{split or 'default'}:{idx}",
                    "source_text": eng,
                    "reference_text": tgt_text,
                }
            )
    return rows_out if rows_out else None


def discover_ntrex_eng_tgt(dataset_id: str) -> list[dict[str, Any]]:
    """尝试多种 split/结构，返回非空平行句列表。"""
    splits = ["test", "train", "validation", "dev", None]
    for sp in splits:
        r = try_build_ntrex_rows(dataset_id, sp, "eng_Latn", "zho_Hans", "zho_Hans")
        if r:
            return r
    return []


def main() -> int:
    os.environ.setdefault("MODELSCOPE_CACHE", str(root() / "datasets" / "cache" / "modelscope"))
    eval_cfg = load_json(root() / "evaluation_config.json")
    sources = load_json(root() / "modelscope_sources.json")
    mapping = load_json(root() / "datasets" / "flores_ntrex_mapping.json")
    mapping.pop("_comment", None)

    flores_id = sources["datasets"]["flores"]["dataset_id"]
    ntrex_id = sources["datasets"]["ntrex"]["dataset_id"]
    split = eval_cfg["split"]
    bidir = bool(eval_cfg.get("bidirectional", True))
    groups = eval_cfg["language_pair_groups"]

    processed = root() / "datasets" / "processed"
    previews = root() / "datasets" / "previews"
    ensure_dir(processed)
    ensure_dir(previews)

    pairs = expand_pairs(groups, bidir)
    langs = sorted({x for group in groups for x in group})

    print("加载 FLORES 各语言表:", langs)
    flores_tables: dict[str, dict[str, str]] = {}
    for lang in langs:
        flores_tables[lang] = load_flores_lang_table(flores_id, lang, split)
        print(f"  {lang}: {len(flores_tables[lang])} 条")

    for src, tgt in pairs:
        rows = build_flores_parallel(flores_id, split, src, tgt, flores_tables[src], flores_tables[tgt])
        name = f"flores_{split}_{src}__{tgt}.jsonl"
        path = processed / name
        write_jsonl(path, rows)
        write_preview(path, previews / f"{name}.preview_{PREVIEW_N}.jsonl")
        print(f"FLORES 写出 {path.name} ({len(rows)} 条)")

    # NTREX：英语中心。对 evaluation_config 中出现的所有语言 L（L≠英语）生成 eng_Latn→L，
    # 不必属于同一「语言组」内的两两组合（第二组只有非英语言时仍要评 en→spa 等）。
    print("探测 NTREX 数据结构...")
    probe = discover_ntrex_eng_tgt(ntrex_id)
    if not probe:
        print("警告: 未能从 NTREX 解析出英语中心平行句，请检查 MTEB/NTREX 在 ModelScope 上的实际列名并在本脚本中扩展解析逻辑。")
        (processed / "NTREX_LOAD_FAILED.txt").write_text(
            "NTREX 自动解析失败；请手动查看 MsDataset 列名后改 try_build_ntrex_rows / _pick_tgt_text。\n",
            encoding="utf-8",
        )
    else:
        print(f"NTREX 探测到样例 {len(probe)} 条（结构可用）")

    # NTREX：仅保留 english-centered 且在“跨组配对”中实际出现的方向 eng_Latn -> tgt
    eng = "eng_Latn"
    ntrex_targets: list[str] = []
    if eng in langs:
        for src, tgt in pairs:
            if src == eng and tgt != eng and tgt not in ntrex_targets:
                ntrex_targets.append(tgt)

    for tgt in ntrex_targets:
        src = eng
        tk = mapping.get(tgt)
        if not tk:
            print(f"NTREX 跳过（无映射）: {src}->{tgt}")
            continue
        rows: list[dict[str, Any]] | None = None
        for sp in ("test", "train", "validation", "dev", None):
            rows = try_build_ntrex_rows(ntrex_id, sp, eng, tgt, tk)
            if rows:
                break
        if not rows:
            print(f"NTREX 跳过（无数据）: {src}->{tgt} key={tk}")
            continue
        name = f"ntrex_{split}_{src}__{tgt}.jsonl"
        path = processed / name
        write_jsonl(path, rows)
        write_preview(path, previews / f"{name}.preview_{PREVIEW_N}.jsonl")
        print(f"NTREX 写出 {path.name} ({len(rows)} 条)")

    print("prepare_datasets 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
