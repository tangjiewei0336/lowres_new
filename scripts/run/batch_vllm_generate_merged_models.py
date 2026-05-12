#!/usr/bin/env python3
"""
按顺序对每个合并后的 HuggingFace 模型目录启动 vLLM，调用 run_eval.py 仅生成 hypotheses（--skip-metrics），再关闭 vLLM。

适用：models/ 下 Qwen3-8B_aug_merged_1000 … Qwen3-8B_aug_merged_10000 等多套权重，单卡/单进程依次评测。

前置：可执行 `vllm`（或在 --vllm-bin 指定）；当前 conda/env 能运行 run_eval.py（openai、tqdm 等）。

示例：
  python scripts/run/batch_vllm_generate_merged_models.py \\
    --models-dir models \\
    --name-glob 'Qwen3-8B_aug_merged_*' \\
    --output-parent-dir eval_multilingual/qwen3_8b_aug_merged_sweep \\
    --model-family qwen3 \\
    --served-model-name merged \\
    --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def merged_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"_merged_(\d+)$", path.name)
    step = int(m.group(1)) if m else 0
    return (step, path.name)


def wait_openai_models(base_v1: str, *, timeout_s: float, interval_s: float = 2.0) -> None:
    """base_v1 形如 http://127.0.0.1:8000/v1"""
    url = base_v1.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "batch-vllm-gen/1"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)
        time.sleep(interval_s)
    raise RuntimeError(f"vLLM 在 {timeout_s}s 内未就绪: {url} 最后错误: {last_err}")


def start_vllm(
    *,
    vllm_bin: str,
    model_path: Path,
    host: str,
    port: int,
    served_name: str,
    tensor_parallel: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    extra_args: list[str],
    log_path: Path,
) -> tuple[subprocess.Popen, object]:
    cmd = [
        vllm_bin,
        "serve",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    cmd.extend(extra_args)
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return proc, log_f


def kill_proc_group(proc: subprocess.Popen, *, grace_s: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    t0 = time.monotonic()
    while proc.poll() is None and time.monotonic() - t0 < grace_s:
        time.sleep(0.3)
    if proc.poll() is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description="批量 vLLM 部署 + run_eval 仅生成 hypotheses")
    ap.add_argument("--models-dir", type=Path, default=root / "models")
    ap.add_argument(
        "--name-glob",
        default="Qwen3-8B_aug_merged_*",
        help="在 models-dir 下匹配的目录名 glob",
    )
    ap.add_argument(
        "--output-parent-dir",
        type=Path,
        required=True,
        help="每次运行的输出父目录；子目录为各模型文件夹名",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-model-name", default="merged", help="与 run_eval 的 --served-model-name 一致")
    ap.add_argument(
        "--vllm-bin",
        default=os.environ.get("VLLM_BIN", "vllm"),
        help="vllm 可执行文件；或 conda 环境里的绝对路径",
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument(
        "--extra-vllm-args",
        default="",
        help='附加 vllm 参数（一条字符串，按 shlex 拆分）。留空且 model-family 为 qwen* 时默认加 --default-chat-template-kwargs \'{"enable_thinking": false}\'。',
    )
    ap.add_argument("--startup-timeout-s", type=float, default=900.0)
    ap.add_argument(
        "--eval-config",
        type=Path,
        default=root / "evaluation_config.json",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=root / "datasets" / "eval_manifest.json",
    )
    ap.add_argument("--model-family", default=os.environ.get("EVAL_MODEL_FAMILY", "qwen3"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--run-eval-extra", default="", help="传给 run_eval.py 的额外参数（引号包裹的一条字符串）")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="若 run_dir/hypotheses.jsonl 已存在且非空则跳过该模型",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models_dir: Path = args.models_dir.resolve()
    if not models_dir.is_dir():
        print(f"models-dir 不存在: {models_dir}", file=sys.stderr)
        return 1

    candidates = sorted(models_dir.glob(args.name_glob), key=merged_sort_key)
    dirs = [p for p in candidates if p.is_dir()]
    if not dirs:
        print(f"未找到匹配目录: {models_dir}/{args.name_glob}", file=sys.stderr)
        return 1

    parent: Path = args.output_parent_dir
    if not parent.is_absolute():
        parent = (root / parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    extra_vllm: list[str] = []
    if str(args.extra_vllm_args).strip():
        import shlex

        extra_vllm = shlex.split(str(args.extra_vllm_args))
    elif str(args.model_family).lower().startswith("qwen"):
        extra_vllm = ["--default-chat-template-kwargs", '{"enable_thinking": false}']

    vllm_path = shutil.which(args.vllm_bin) or args.vllm_bin
    if not args.dry_run and not Path(vllm_path).exists() and shutil.which(args.vllm_bin) is None:
        print(f"找不到 vllm 可执行: {args.vllm_bin}", file=sys.stderr)
        return 1

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"警告: manifest 不存在: {manifest_path}", file=sys.stderr)
    else:
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            rel = man.get("items_jsonl")
            if rel:
                ip = root / rel
                if not ip.is_file():
                    print(f"警告: items_jsonl 不存在: {ip}", file=sys.stderr)
        except Exception as e:
            print(f"警告: 读取 manifest 失败: {e}", file=sys.stderr)

    summary: list[dict] = []
    for mp in dirs:
        tag = mp.name
        run_dir = parent / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        hyp_existing = run_dir / "hypotheses.jsonl"
        if args.resume and hyp_existing.is_file() and hyp_existing.stat().st_size > 0:
            print(f"resume: 跳过已有 {hyp_existing}", file=sys.stderr)
            summary.append({"model": tag, "skipped": True, "run_dir": str(run_dir)})
            continue
        base_v1 = f"http://{args.host}:{args.port}/v1"
        print(f"\n=== {tag} ===\n  model: {mp}\n  out: {run_dir}\n  API: {base_v1}", file=sys.stderr)
        if args.dry_run:
            summary.append({"model": tag, "dry_run": True, "run_dir": str(run_dir)})
            continue

        proc: subprocess.Popen | None = None
        log_fp: object | None = None
        try:
            vllm_log = run_dir / "vllm_serve.log"
            proc, log_fp = start_vllm(
                vllm_bin=vllm_path,
                model_path=mp,
                host=args.host,
                port=args.port,
                served_name=args.served_model_name,
                tensor_parallel=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                extra_args=extra_vllm,
                log_path=vllm_log,
            )
            wait_openai_models(base_v1, timeout_s=args.startup_timeout_s)

            eval_py = root / "scripts" / "run" / "run_eval.py"
            rcmd = [
                sys.executable,
                str(eval_py),
                "--eval-config",
                str(args.eval_config),
                "--manifest",
                str(args.manifest),
                "--base-url",
                base_v1,
                "--api-key",
                args.api_key,
                "--served-model-name",
                args.served_model_name,
                "--model-family",
                args.model_family,
                "--model-tag",
                tag,
                "--output-run-dir",
                str(run_dir),
                "--skip-metrics",
            ]
            if args.run_eval_extra.strip():
                import shlex

                rcmd.extend(shlex.split(args.run_eval_extra))

            print("运行:", " ".join(rcmd), file=sys.stderr)
            r = subprocess.run(rcmd, cwd=str(root))
            summary.append(
                {
                    "model": tag,
                    "run_dir": str(run_dir),
                    "returncode": r.returncode,
                }
            )
            if r.returncode != 0:
                print(f"run_eval 失败 rc={r.returncode}，继续下一模型…", file=sys.stderr)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            summary.append({"model": tag, "error": str(e), "run_dir": str(run_dir)})
        finally:
            if proc is not None:
                kill_proc_group(proc)
            if log_fp is not None:
                try:
                    log_fp.close()
                except Exception:
                    pass

    summary_path = parent / "batch_generation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总: {summary_path}", file=sys.stderr)
    if any(s.get("returncode", 0) != 0 for s in summary if "returncode" in s):
        return 1
    if any("error" in s for s in summary):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
