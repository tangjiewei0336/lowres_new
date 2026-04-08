#!/usr/bin/env python3
"""
为每次评测输出目录绘制 BLEU 热力图：
- 同一模型：分别输出 FLORES 与 NTREX 两张图
- 多个 run_dir：可统一计算全局色标（所有图片共享同一标尺）

输入：run_eval.py 生成的 metrics.csv / metrics_{corpus}.csv
输出：<run_dir>/plots/heatmap_bleu_{corpus}.png

用法示例：
  conda activate lowres
  python scripts/plot_bleu_heatmap.py eval_multilingual/baseline_qwen3_4b_1712450000

多个 run 统一标尺：
  python scripts/plot_bleu_heatmap.py eval_multilingual/run1 eval_multilingual/run2
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_eval_langs() -> list[str]:
    import json

    cfg = json.loads((root() / "evaluation_config.json").read_text(encoding="utf-8"))
    # 按出现顺序拼接并去重
    seen: set[str] = set()
    langs: list[str] = []
    for group in cfg.get("language_pair_groups", []):
        for x in group:
            if x not in seen:
                seen.add(x)
                langs.append(x)
    return langs


def parse_pair(s: str) -> tuple[str, str]:
    if "->" not in s:
        raise ValueError(f"非法语言对: {s}")
    a, b = s.split("->", 1)
    return a.strip(), b.strip()


def read_metrics_rows(run_dir: Path) -> list[dict[str, str]]:
    # 优先总表 metrics.csv（含 corpus 列）
    p = run_dir / "metrics.csv"
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise FileNotFoundError(f"未找到 {p}")


def build_matrix(
    rows: list[dict[str, str]],
    corpus: str,
    langs: list[str],
) -> list[list[float | None]]:
    # 初始化为 None
    idx = {l: i for i, l in enumerate(langs)}
    n = len(langs)
    mat: list[list[float | None]] = [[None for _ in range(n)] for _ in range(n)]
    for r in rows:
        if str(r.get("corpus", "")).strip() != corpus:
            continue
        pair = str(r.get("pair", "")).strip()
        if not pair:
            continue
        src, tgt = parse_pair(pair)
        if src not in idx or tgt not in idx:
            continue
        try:
            v = float(r["bleu"])
        except Exception:
            continue
        mat[idx[src]][idx[tgt]] = v
    return mat


def build_matrix_metric(
    rows: list[dict[str, str]],
    corpus: str,
    langs: list[str],
    metric: str,
) -> list[list[float | None]]:
    idx = {l: i for i, l in enumerate(langs)}
    n = len(langs)
    mat: list[list[float | None]] = [[None for _ in range(n)] for _ in range(n)]
    for r in rows:
        if str(r.get("corpus", "")).strip() != corpus:
            continue
        pair = str(r.get("pair", "")).strip()
        if not pair:
            continue
        src, tgt = parse_pair(pair)
        if src not in idx or tgt not in idx:
            continue
        try:
            raw = str(r.get(metric, "")).strip()
            if not raw:
                continue
            v = float(raw)
        except Exception:
            continue
        mat[idx[src]][idx[tgt]] = v
    return mat


def iter_values(mats: Iterable[list[list[float | None]]]) -> Iterable[float]:
    for m in mats:
        for row in m:
            for v in row:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                yield float(v)


@dataclass(frozen=True)
class Scale:
    vmin: float
    vmax: float


def compute_global_scale(all_mats: list[list[list[float | None]]]) -> Scale:
    vals = list(iter_values(all_mats))
    if not vals:
        return Scale(0.0, 1.0)
    return Scale(min(vals), max(vals))


def plot_heatmap(
    mat_bleu: list[list[float | None]],
    mat_comet: list[list[float | None]] | None,
    langs: list[str],
    title: str,
    out_path: Path,
    scale: Scale,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    arr = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in mat_bleu],
        dtype=float,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 适当增大画布：语言越多越大，避免标注重叠
    fig_w = max(8.0, 0.85 * len(langs))
    fig_h = max(6.0, 0.85 * len(langs))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)

    # 白 -> 粉 -> 红（低分更浅，高分更红）
    cmap = LinearSegmentedColormap.from_list(
        "white_pink_red",
        ["#ffffff", "#f7b6d2", "#d62728"],
    )
    im = ax.imshow(arr, vmin=scale.vmin, vmax=scale.vmax, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(langs)))
    ax.set_xticklabels(langs, rotation=45, ha="right")
    ax.set_yticklabels(langs)
    ax.set_xlabel("tgt_lang")
    ax.set_ylabel("src_lang")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("BLEU")

    # 单元格数值标注：BLEU + COMET（若缺失则跳过对应值）
    # 注：颜色按 BLEU；字体颜色根据底色深浅自动选择黑/白
    n = len(langs)
    norm = im.norm
    for i in range(n):
        for j in range(n):
            v_bleu = mat_bleu[i][j]
            v_comet = None
            if mat_comet is not None:
                v_comet = mat_comet[i][j]
            if v_bleu is None and v_comet is None:
                continue

            lines: list[str] = []
            if v_bleu is not None and not (isinstance(v_bleu, float) and math.isnan(v_bleu)):
                lines.append(f"B {float(v_bleu):.2f}")
            if v_comet is not None and not (isinstance(v_comet, float) and math.isnan(v_comet)):
                lines.append(f"C {float(v_comet):.3f}")
            if not lines:
                continue

            # 根据 BLEU 底色选择文字颜色（BLEU 缺失时用黑色）
            if v_bleu is None:
                txt_color = "black"
            else:
                rgba = cmap(float(norm(float(v_bleu))))
                # 感知亮度（越亮越接近白底）
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "black" if lum > 0.6 else "white"

            ax.text(
                j,
                i,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _fmt_delta(v: float, decimals: int) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}"


def plot_delta_heatmap(
    mat_bleu_a: list[list[float | None]],
    mat_bleu_b: list[list[float | None]],
    mat_comet_a: list[list[float | None]] | None,
    mat_comet_b: list[list[float | None]] | None,
    langs: list[str],
    title: str,
    out_path: Path,
) -> None:
    """
    绘制 runA 相对 runB 的变化：Δ = A - B。
    - 底色：按 ΔBLEU，增加为黄色，减少为绿色
    - 文本：同一格内写 ΔBLEU 与 ΔCOMET（若缺失则跳过）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    n = len(langs)
    d_bleu = np.full((n, n), np.nan, dtype=float)
    d_comet = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            a = mat_bleu_a[i][j]
            b = mat_bleu_b[i][j]
            if a is not None and b is not None and not (math.isnan(float(a)) or math.isnan(float(b))):
                d_bleu[i, j] = float(a) - float(b)
            if mat_comet_a is not None and mat_comet_b is not None:
                ca = mat_comet_a[i][j]
                cb = mat_comet_b[i][j]
                if ca is not None and cb is not None and not (math.isnan(float(ca)) or math.isnan(float(cb))):
                    d_comet[i, j] = float(ca) - float(cb)

    # 颜色范围：以 ΔBLEU 的最大绝对值为对称范围
    finite = d_bleu[np.isfinite(d_bleu)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # 绿（负）-> 白（0）-> 红（正）
    cmap = LinearSegmentedColormap.from_list(
        "green_white_red",
        ["#2ca02c", "#ffffff", "#d62728"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(8.0, 0.85 * len(langs))
    fig_h = max(6.0, 0.85 * len(langs))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)

    im = ax.imshow(d_bleu, norm=norm, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(langs)))
    ax.set_xticklabels(langs, rotation=45, ha="right")
    ax.set_yticklabels(langs)
    ax.set_xlabel("tgt_lang")
    ax.set_ylabel("src_lang")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ΔBLEU (runA - runB)")

    for i in range(n):
        for j in range(n):
            vdb = d_bleu[i, j]
            vdc = d_comet[i, j]
            if not np.isfinite(vdb) and not np.isfinite(vdc):
                continue

            lines: list[str] = []
            if np.isfinite(vdb):
                lines.append(f"ΔB {_fmt_delta(float(vdb), 2)}")
            if np.isfinite(vdc):
                lines.append(f"ΔC {_fmt_delta(float(vdc), 3)}")
            if not lines:
                continue

            rgba = cmap(float(norm(float(vdb))) if np.isfinite(vdb) else 0.0)
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "black" if lum > 0.6 else "white"

            ax.text(
                j,
                i,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="按数据集绘制 BLEU 语言对热力图（统一标尺）")
    ap.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="run_eval.py 输出目录（如 eval_multilingual/baseline_xxx_...）",
    )
    ap.add_argument(
        "--corpora",
        type=str,
        default="flores,ntrex",
        help="要绘制的数据集，逗号分隔（默认 flores,ntrex）",
    )
    args = ap.parse_args()

    langs = load_eval_langs()
    corpora = [x.strip() for x in args.corpora.split(",") if x.strip()]

    # 双 run 对比模式：画 Δ热力图（runA - runB）
    if len(args.run_dirs) == 2:
        run_a = args.run_dirs[0] if args.run_dirs[0].is_absolute() else (root() / args.run_dirs[0])
        run_b = args.run_dirs[1] if args.run_dirs[1].is_absolute() else (root() / args.run_dirs[1])
        rows_a = read_metrics_rows(run_a)
        rows_b = read_metrics_rows(run_b)

        out_base = run_a / "plots"
        for corp in corpora:
            a_bleu = build_matrix_metric(rows_a, corp, langs, metric="bleu")
            b_bleu = build_matrix_metric(rows_b, corp, langs, metric="bleu")
            a_comet = build_matrix_metric(rows_a, corp, langs, metric="comet")
            b_comet = build_matrix_metric(rows_b, corp, langs, metric="comet")
            out_path = out_base / f"heatmap_delta_bleu_comet_{corp}__{run_a.name}__vs__{run_b.name}.png"
            plot_delta_heatmap(
                mat_bleu_a=a_bleu,
                mat_bleu_b=b_bleu,
                mat_comet_a=a_comet,
                mat_comet_b=b_comet,
                langs=langs,
                title=f"{run_a.name} vs {run_b.name} | {corp} | Δ = A - B",
                out_path=out_path,
            )
            print(f"写出: {out_path}")
        return 0

    # 读取所有 run 的矩阵，用于计算全局色标
    mats_for_scale: list[list[list[float | None]]] = []
    rows_by_run: dict[Path, list[dict[str, str]]] = {}
    for rd in args.run_dirs:
        run_dir = rd if rd.is_absolute() else (root() / rd)
        rows = read_metrics_rows(run_dir)
        rows_by_run[run_dir] = rows
        for corp in corpora:
            mats_for_scale.append(build_matrix_metric(rows, corp, langs, metric="bleu"))

    scale = compute_global_scale(mats_for_scale)

    for run_dir, rows in rows_by_run.items():
        plots_dir = run_dir / "plots"
        for corp in corpora:
            mat_bleu = build_matrix_metric(rows, corp, langs, metric="bleu")
            mat_comet = build_matrix_metric(rows, corp, langs, metric="comet")
            out_path = plots_dir / f"heatmap_bleu_{corp}.png"
            plot_heatmap(
                mat_bleu=mat_bleu,
                mat_comet=mat_comet,
                langs=langs,
                title=f"{run_dir.name} | {corp} | BLEU (vmin={scale.vmin:.3f}, vmax={scale.vmax:.3f})",
                out_path=out_path,
                scale=scale,
            )
            print(f"写出: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

