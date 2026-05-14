#!/usr/bin/env python3
"""
Select the best LoRA checkpoint for each MoE expert using COMET scores from eval runs.

Expected training layout on the remote machine:

  <adapter_root>/<adapter_name>/checkpoint-200
  <adapter_root>/<adapter_name>/checkpoint-400
  <adapter_root>/<adapter_name>/checkpoint-600
  <adapter_root>/<adapter_name>/checkpoint-782

Expected eval layout:
  any directory tree under --eval-root that contains metrics.csv and optionally
  hypotheses.jsonl. The script infers:
    - language pair from hypotheses.jsonl (preferred) or metrics.csv
    - checkpoint step from the eval run path name (e.g. checkpoint-400)

Default selection policy:
  - only use corpus == flores
  - choose the checkpoint with the highest COMET
  - break ties by larger step

Outputs:
  - updated router manifest with adapter_path rewritten to checkpoint dirs
  - CSV summary of all candidate scores and chosen checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"checkpoint[-_]?(\d+)")


@dataclass
class Candidate:
    pair: str
    step: int
    comet: float
    corpus: str
    run_dir: Path


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_step_from_path(path: Path) -> int | None:
    for part in (path,) + tuple(path.parents):
        m = STEP_RE.search(part.name)
        if m:
            return int(m.group(1))
    m = STEP_RE.search(str(path))
    if m:
        return int(m.group(1))
    return None


def infer_pair_from_hypotheses(path: Path) -> str | None:
    if not path.is_file():
        return None
    pairs: set[str] = set()
    for row in read_jsonl(path):
        src = str(row.get("src_lang", "")).strip()
        tgt = str(row.get("tgt_lang", "")).strip()
        if src and tgt:
            pairs.add(f"{src}->{tgt}")
        eval_pair = str(row.get("eval_pair", "")).strip()
        if eval_pair:
            pairs.add(eval_pair)
        if len(pairs) > 1:
            return None
    return next(iter(pairs)) if pairs else None


def infer_pair_from_metrics(metrics_csv: Path, corpus: str) -> str | None:
    pairs: set[str] = set()
    with open(metrics_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_corpus = str(row.get("corpus", "")).strip()
            if row_corpus and row_corpus != corpus:
                continue
            pair = str(row.get("pair", "")).strip()
            if pair:
                pairs.add(pair)
            if len(pairs) > 1:
                return None
    return next(iter(pairs)) if pairs else None


def read_comet_from_metrics(metrics_csv: Path, pair: str, corpus: str) -> float | None:
    with open(metrics_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("corpus", "")).strip() != corpus:
                continue
            if str(row.get("pair", "")).strip() != pair:
                continue
            val = str(row.get("comet", "")).strip()
            if not val:
                return None
            return float(val)
    return None


def read_pair_comet_from_metrics(metrics_csv: Path, corpus: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(metrics_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("corpus", "")).strip() != corpus:
                continue
            pair = str(row.get("pair", "")).strip()
            val = str(row.get("comet", "")).strip()
            if not pair or not val:
                continue
            try:
                out[pair] = float(val)
            except ValueError:
                continue
    return out


def adapter_name_to_pair(adapter_name: str) -> str:
    prefix = "qwen3_8b_moe_"
    if adapter_name.startswith(prefix):
        body = adapter_name[len(prefix) :]
    else:
        body = adapter_name
    if "_to_" in body:
        src, tgt = body.split("_to_", 1)
        return f"{src}->{tgt}"
    raise ValueError(f"Cannot infer pair from adapter_name: {adapter_name}")


def find_metrics_files(eval_root: Path) -> list[Path]:
    return sorted(
        p for p in eval_root.rglob("metrics.csv")
        if p.is_file()
    )


def choose_best(cands: list[Candidate]) -> Candidate:
    return sorted(cands, key=lambda x: (x.comet, x.step), reverse=True)[0]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select best MoE LoRA checkpoints by COMET.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("training/moe_router_manifest.json"),
        help="Original router manifest with expert adapter root dirs.",
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Root directory containing per-checkpoint eval run outputs.",
    )
    p.add_argument(
        "--corpus",
        default="flores",
        help="Only use this corpus when reading COMET from metrics.csv. Default: flores",
    )
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Candidate checkpoint steps to consider (explicit list).",
    )
    p.add_argument(
        "--step-start",
        type=int,
        default=None,
        help="Start step for range mode (requires --step-end and --step-interval).",
    )
    p.add_argument(
        "--step-end",
        type=int,
        default=None,
        help="End step for range mode (requires --step-start and --step-interval).",
    )
    p.add_argument(
        "--step-interval",
        type=int,
        default=None,
        help="Step interval for range mode (requires --step-start and --step-end).",
    )
    p.add_argument(
        "--include-end-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If range mode is used and end is not aligned to interval, also include end step.",
    )
    p.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("training/moe_router_manifest.best_by_comet.json"),
        help="Output manifest with adapter_path rewritten to best checkpoint dirs.",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("training/moe_best_checkpoints_by_comet.csv"),
        help="Output CSV summary.",
    )
    p.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any expert has no usable COMET candidate.",
    )
    return p


def resolve_wanted_steps(args: argparse.Namespace) -> set[int]:
    has_range_flag = any(
        x is not None for x in (args.step_start, args.step_end, args.step_interval)
    )
    if has_range_flag:
        if not all(
            x is not None for x in (args.step_start, args.step_end, args.step_interval)
        ):
            raise SystemExit(
                "Range mode requires --step-start, --step-end and --step-interval together."
            )
        if args.steps:
            raise SystemExit("Use either --steps or range mode args, not both.")
        if int(args.step_interval) <= 0:
            raise SystemExit("--step-interval must be > 0.")
        if int(args.step_start) <= 0 or int(args.step_end) <= 0:
            raise SystemExit("--step-start/--step-end must be > 0.")
        if int(args.step_end) < int(args.step_start):
            raise SystemExit("--step-end must be >= --step-start.")

        steps = list(
            range(int(args.step_start), int(args.step_end) + 1, int(args.step_interval))
        )
        if args.include_end_step and int(args.step_end) not in steps:
            steps.append(int(args.step_end))
        return set(int(x) for x in steps)

    if args.steps:
        return set(int(x) for x in args.steps)
    return {200, 400, 600, 782}


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = load_json(args.manifest)
    experts = manifest.get("experts") or []
    if not experts:
        raise SystemExit(f"No experts found in {args.manifest}")

    wanted_steps = resolve_wanted_steps(args)
    candidates_by_pair: dict[str, list[Candidate]] = defaultdict(list)
    metrics_files = find_metrics_files(args.eval_root)
    if not metrics_files:
        raise SystemExit(f"No metrics.csv found under {args.eval_root}")

    for metrics_csv in metrics_files:
        run_dir = metrics_csv.parent
        step = infer_step_from_path(run_dir)
        if step is None or step not in wanted_steps:
            continue

        # 优先支持一次评测包含多个 pair 的 metrics.csv（例如 moe_router 整体评测）。
        pair_comets = read_pair_comet_from_metrics(metrics_csv, args.corpus)
        if pair_comets:
            for pair, comet in pair_comets.items():
                candidates_by_pair[pair].append(
                    Candidate(
                        pair=pair,
                        step=step,
                        comet=float(comet),
                        corpus=str(args.corpus),
                        run_dir=run_dir,
                    )
                )
            continue

        # 兼容旧格式：单 pair 评测目录。
        pair = infer_pair_from_hypotheses(run_dir / "hypotheses.jsonl")
        if pair is None:
            pair = infer_pair_from_metrics(metrics_csv, args.corpus)
        if not pair:
            continue
        comet = read_comet_from_metrics(metrics_csv, pair, args.corpus)
        if comet is None:
            continue
        candidates_by_pair[pair].append(
            Candidate(
                pair=pair,
                step=step,
                comet=float(comet),
                corpus=str(args.corpus),
                run_dir=run_dir,
            )
        )

    rows: list[dict[str, Any]] = []
    best_by_pair: dict[str, Candidate] = {}
    missing_pairs: list[str] = []

    for item in experts:
        adapter_name = str(item.get("adapter_name") or "").strip()
        adapter_root = Path(str(item.get("adapter_path") or "").strip())
        pair = adapter_name_to_pair(adapter_name)
        cands = candidates_by_pair.get(pair, [])
        if not cands:
            missing_pairs.append(pair)
            rows.append(
                {
                    "pair": pair,
                    "adapter_name": adapter_name,
                    "adapter_root": str(adapter_root),
                    "best_step": "",
                    "best_comet": "",
                    "best_adapter_path": "",
                    "status": "missing_eval",
                }
            )
            continue
        best = choose_best(cands)
        best_by_pair[pair] = best
        best_path = adapter_root / f"checkpoint-{best.step}"
        rows.append(
            {
                "pair": pair,
                "adapter_name": adapter_name,
                "adapter_root": str(adapter_root),
                "best_step": best.step,
                "best_comet": f"{best.comet:.6f}",
                "best_adapter_path": str(best_path),
                "status": "ok" if best_path.is_dir() else "checkpoint_missing",
            }
        )

    if missing_pairs and args.strict:
        raise SystemExit(
            "Missing COMET eval for pairs: " + ", ".join(sorted(missing_pairs))
        )

    out_manifest = json.loads(json.dumps(manifest))
    for item in out_manifest.get("experts") or []:
        pair = adapter_name_to_pair(str(item.get("adapter_name") or "").strip())
        best = best_by_pair.get(pair)
        if best is None:
            continue
        item["selected_by"] = "comet"
        item["selected_corpus"] = args.corpus
        item["selected_step"] = best.step
        item["selected_comet"] = best.comet
        item["adapter_root"] = item.get("adapter_path")
        item["adapter_path"] = str(Path(str(item["adapter_path"])) / f"checkpoint-{best.step}")
        item["selection_run_dir"] = str(best.run_dir)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair",
                "adapter_name",
                "adapter_root",
                "best_step",
                "best_comet",
                "best_adapter_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest: {args.output_manifest}")
    print(f"Wrote summary: {args.summary_csv}")
    for row in rows:
        print(
            f"{row['pair']}: step={row['best_step'] or 'NA'} comet={row['best_comet'] or 'NA'} status={row['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
