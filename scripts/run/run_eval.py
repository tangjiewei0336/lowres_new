#!/usr/bin/env python3
"""
通过 vLLM 的 OpenAI 兼容 HTTP 接口批量翻译并计算 BLEU（sacrebleu）与 COMET。
在 conda env lowres 中运行。Qwen3 通过 extra_body 关闭 enable_thinking（非思考/推理模式）；其它基座无此字段。

BLEU 分词（默认 --bleu-tokenize=auto）：
- 目标语为泰语（tha_*）：PyThaiNLP word_tokenize（默认 newmm）词级切分，再对 sacrebleu 使用 tokenize=none（避免 char-level BLEU 偏高、与业界评测习惯一致）。
- FLORES 且非泰语目标：spBLEU，sacrebleu flores200。
- 其它语料（如 NTREX）：中文 zh，泰语同上 PyThai，西班牙语等其余语言 13a。
- --bleu-tokenize=flores200：全语向强制 flores200（与部分论文单栏 spBLEU 完全一致，含泰语 SPM）。
- legacy：仅中文 zh，其它 13a（旧行为）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from flores_lang_zh import english_translation_instruction  # noqa: E402


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(p: Path) -> dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_items_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mt_user_content(src_lang: str, tgt_lang: str, text: str) -> str:
    return f"{english_translation_instruction(src_lang, tgt_lang)}\n\n{text}"


def _tgt_lang_from_eval_pair(eval_pair: str) -> str:
    if "->" not in eval_pair:
        return ""
    return eval_pair.split("->", 1)[1].strip()


# 泰语 BLEU：sacrebleu 无内置 PyThai 分词器；先词级切分再 tokenize=none（与 PyThaiNLP 评测实践一致）
_THAI_SACREbleu_TOK = "pythai_newmm"


def _segment_thai_pythai_words(texts: list[str], *, engine: str = "newmm") -> list[str]:
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError as e:
        raise RuntimeError(
            "泰语 BLEU 需要 PyThaiNLP：pip install pythainlp"
        ) from e
    out: list[str] = []
    for s in texts:
        # keep_whitespace=False：避免参考里已有空格时被拆成空 token，导致词边界与假设不一致
        toks = word_tokenize((s or "").strip(), engine=engine, keep_whitespace=False)
        out.append(" ".join(toks))
    return out


def sacrebleu_tokenize_heuristic_tgt(eval_pair: str) -> str:
    """非 FLORES 语料等：中文不用纯 13a（泰语在 auto 主分支已优先处理）。"""
    tgt = _tgt_lang_from_eval_pair(eval_pair)
    if tgt.startswith("zho_") or tgt.startswith("cmn_"):
        return "zh"
    return "13a"


def sacrebleu_tokenize_for_group(eval_corpus: str, eval_pair: str, policy: str) -> str:
    """policy: auto | flores200 | legacy"""
    pol = (policy or "auto").strip().lower()
    tgt = _tgt_lang_from_eval_pair(eval_pair)
    if pol == "legacy":
        if tgt.startswith("zho_") or tgt.startswith("cmn_"):
            return "zh"
        return "13a"
    if pol == "flores200":
        return "flores200"
    # auto：泰语始终词级（PyThai），不用 char / 不单依赖 flores200 的 SPM
    if tgt.startswith("tha_"):
        return _THAI_SACREbleu_TOK
    if (eval_corpus or "").lower() == "flores":
        return "flores200"
    return sacrebleu_tokenize_heuristic_tgt(eval_pair)


def _corpus_bleu_score(hyps: list[str], refs: list[str], tokenize: str) -> float:
    import sacrebleu

    if tokenize == _THAI_SACREbleu_TOK:
        hyps_t = _segment_thai_pythai_words(hyps)
        refs_t = _segment_thai_pythai_words(refs)
        return float(sacrebleu.corpus_bleu(hyps_t, [refs_t], tokenize="none").score)
    return float(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize).score)


def _segment_with_sacrebleu_tokenizer(texts: list[str], tokenize: str) -> list[str]:
    """将 sacrebleu 的 tokenize 结果（空格分隔）输出，便于人工排查。"""
    import sacrebleu

    # BLEU(tokenize=...) 的 tokenizer 会把输入转成以空格分隔的“token串”
    metric = sacrebleu.BLEU(tokenize=tokenize, force=True)
    tok_fn = metric.tokenizer
    return [tok_fn((t or "").strip()) for t in texts]


def corpus_bleu_with_fallbacks(
    hyps: list[str],
    refs: list[str],
    eval_corpus: str,
    eval_pair: str,
    policy: str,
) -> tuple[float, str]:
    """返回 (bleu, 实际使用的 sacrebleu tokenize 名，可能带 fallback 标记)。"""
    tok = sacrebleu_tokenize_for_group(eval_corpus, eval_pair, policy)
    try:
        return _corpus_bleu_score(hyps, refs, tok), tok
    except Exception as e:
        if tok == "ja-mecab":
            print(
                f"BLEU: ja-mecab 不可用（可 pip install 'sacrebleu[ja]'），已改用 char: {e}",
                file=sys.stderr,
            )
            return _corpus_bleu_score(hyps, refs, "char"), "char_ja_fallback"
        if tok == "flores200":
            print(
                f"BLEU: flores200(spBLEU) 失败（需 sentencepiece 且能下载 SPM），改用目标语启发式: {e}",
                file=sys.stderr,
            )
            tgt = _tgt_lang_from_eval_pair(eval_pair)
            if tgt.startswith("tha_"):
                tok2 = _THAI_SACREbleu_TOK
            else:
                tok2 = sacrebleu_tokenize_heuristic_tgt(eval_pair)
            try:
                return _corpus_bleu_score(hyps, refs, tok2), f"{tok2}_spbleu_fallback"
            except Exception as e2:
                if tok2 == "ja-mecab":
                    print(f"BLEU: 启发式 ja-mecab 仍失败，再用 char: {e2}", file=sys.stderr)
                    return _corpus_bleu_score(hyps, refs, "char"), "char_ja_fallback"
                raise
        if tok == _THAI_SACREbleu_TOK:
            print(
                f"BLEU: PyThaiNLP 泰语分词失败，退回 intl（仍非 char）: {e}",
                file=sys.stderr,
            )
            try:
                return _corpus_bleu_score(hyps, refs, "intl"), "intl_thai_fallback"
            except Exception as e2:
                print(f"BLEU: intl 仍失败，最后退回 13a: {e2}", file=sys.stderr)
                return _corpus_bleu_score(hyps, refs, "13a"), "13a_thai_fallback"
        raise


def build_extra_body(model_family: str) -> dict[str, Any] | None:
    fam = (model_family or "").lower()
    if fam in ("qwen3", "qwen", "qwen3-4b"):
        # 与 vLLM --default-chat-template-kwargs 一致：关闭 Qwen3 思考链，仅输出译文
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def comet_disabled(comet_model: str) -> bool:
    return str(comet_model).lower() in ("none", "off", "disable", "disabled")


def _find_comet_ckpt(path: Path) -> str | None:
    if path.is_file():
        return str(path)
    cand = path / "checkpoints" / "model.ckpt"
    if cand.is_file():
        return str(cand)
    ckpts = sorted(path.glob("**/*.ckpt"))
    if ckpts:
        return str(ckpts[0])
    return None


def prepare_comet_checkpoint(comet_model: str, run_dir: Path) -> tuple[str | None, Any | None, Any | None]:
    """
    Import and download/resolve COMET before translation starts.

    This makes network/model-cache failures fail early instead of after all vLLM
    generations have already completed. Returns (checkpoint, torch, load_fn).
    """
    if comet_disabled(comet_model):
        print("COMET 已禁用（--comet-model=none）。", file=sys.stderr)
        return None, None, None

    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        print(f"unbabel-comet/torch 不可用，跳过 COMET: {e}", file=sys.stderr)
        return None, None, None

    cm = str(comet_model)
    p = Path(cm)
    if p.exists():
        comet_ckpt = _find_comet_ckpt(p)
        if comet_ckpt:
            print(f"COMET 使用本地模型: {comet_ckpt}", file=sys.stderr)
            return comet_ckpt, torch, load_from_checkpoint
        print(f"COMET 本地路径存在但未找到 ckpt，将跳过 COMET: {p}", file=sys.stderr)
        return None, torch, load_from_checkpoint

    # Backward-compatible shorthand: default local path
    # models/Unbabel_wmt22-comet-da maps to remote Unbabel/wmt22-comet-da.
    remote_name = cm
    if cm.startswith("models/") and "/" not in Path(cm).name:
        remote_name = Path(cm).name.replace("_", "/", 1)

    save_dir = run_dir / "comet_ckpt"
    if cm.startswith("models/") and "/" not in Path(cm).name:
        save_dir = p

    try:
        print(f"COMET 开始预下载模型: {remote_name}", file=sys.stderr)
        comet_ckpt = download_model(remote_name, saving_directory=str(save_dir))
        print(f"COMET 预下载完成: {comet_ckpt}", file=sys.stderr)
        return comet_ckpt, torch, load_from_checkpoint
    except Exception as e:
        print(f"COMET 模型下载失败，将跳过 COMET: {e}", file=sys.stderr)
        return None, torch, load_from_checkpoint


def call_translate(
    client: OpenAI,
    model: str,
    src_lang: str,
    tgt_lang: str,
    text: str,
    max_tokens: int,
    model_family: str,
) -> str:
    extra = build_extra_body(model_family)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional machine translation engine.",
            },
            {"role": "user", "content": mt_user_content(src_lang, tgt_lang, text)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra:
        kwargs["extra_body"] = extra
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    out = (choice.message.content or "").strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM OpenAI 兼容接口 + BLEU/COMET 评估")
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=root() / "evaluation_config.json",
        help="evaluation_config.json 路径",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root() / "datasets" / "eval_manifest.json",
        help="expand_language_pairs 生成的 manifest",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=os.environ.get("SERVED_MODEL_NAME", "base"),
        help="与 vLLM --served-model-name 一致",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
        help="vLLM OpenAI API base，含 /v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default=os.environ.get("EVAL_MODEL_FAMILY", "generic"),
        help="qwen3 时关闭思考模式；smollm / generic 等按普通对话",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=os.environ.get("EVAL_MODEL_TAG", "run"),
        help="输出子目录名",
    )
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=int(os.environ.get("COMET_BATCH_SIZE", "8")),
    )
    parser.add_argument(
        "--comet-model",
        type=str,
        default=os.environ.get("COMET_MODEL", "models/Unbabel_wmt22-comet-da"),
        help=(
            "COMET 模型名或本地 ckpt 路径。设为 'none' 可禁用。"
            "默认 Unbabel/wmt22-comet-da（需要可访问 HuggingFace；若无法下载将自动跳过 COMET）。"
        ),
    )
    parser.add_argument(
        "--bleu-tokenize",
        type=str,
        choices=("auto", "flores200", "legacy"),
        default=os.environ.get("BLEU_TOKENIZE", "auto"),
        help=(
            "sacrebleu 分词策略：auto=FLORES 用 flores200(spBLEU)，"
            "其它语料按目标语(zh/char/13a 等)；"
            "flores200=全部句对强制 spBLEU；legacy=旧版仅中文 zh 其余 13a。"
        ),
    )
    parser.add_argument(
        "--dump-segmentation",
        action="store_true",
        help="在 hypotheses.jsonl 中额外写出分词后的 hypothesis/reference（主要用于泰语 PyThaiNLP）。",
    )
    args = parser.parse_args()

    eval_cfg = load_json(args.eval_config)
    manifest = load_json(args.manifest)
    items_path = root() / manifest["items_jsonl"]
    if not items_path.is_file():
        print(f"未找到 items_jsonl: {items_path}", file=sys.stderr)
        return 1

    items = read_items_jsonl(items_path)
    if not items:
        print("评估条目为空", file=sys.stderr)
        return 1

    max_tokens = int(eval_cfg.get("max_tokens", 512))
    max_workers = int(eval_cfg.get("max_workers", 8))
    base_out = root() / eval_cfg.get("output_dir", "eval_multilingual")
    run_dir = base_out / f"{args.model_tag}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    comet_ckpt, torch_mod, comet_load_from_checkpoint = prepare_comet_checkpoint(
        str(args.comet_model),
        run_dir,
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    results: list[dict[str, Any]] = []
    lock_results: list[dict[str, Any]] = []

    def _one(it: dict[str, Any]) -> dict[str, Any]:
        src = it["source_text"]
        ref = it["reference_text"]
        sl = it["src_lang"]
        tl = it["tgt_lang"]
        hyp = call_translate(
            client,
            args.served_model_name,
            sl,
            tl,
            src,
            max_tokens,
            args.model_family,
        )
        out = {**it, "hypothesis": hyp}
        if args.dump_segmentation:
            corp = str(it.get("eval_corpus", "?"))
            pair = str(it.get("eval_pair") or (sl + "->" + tl))
            tok = sacrebleu_tokenize_for_group(corp, pair, str(args.bleu_tokenize))
            if tok == _THAI_SACREbleu_TOK:
                out["hypothesis_segmented"] = _segment_thai_pythai_words([hyp])[0]
                out["reference_segmented"] = _segment_thai_pythai_words([ref])[0]
                out["segmentation_tokenizer"] = tok
            elif tok in ("zh", "flores200"):
                # 中文：zh 分词；FLORES 默认 flores200（spBLEU）也常用于中文
                try:
                    out["hypothesis_segmented"] = _segment_with_sacrebleu_tokenizer([hyp], tok)[0]
                    out["reference_segmented"] = _segment_with_sacrebleu_tokenizer([ref], tok)[0]
                    out["segmentation_tokenizer"] = tok
                except Exception as e:
                    # dump 不影响主流程
                    out["segmentation_tokenizer"] = f"{tok}_dump_failed"
                    out["segmentation_error"] = str(e)
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, it) for it in items]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="translate"):
            lock_results.append(fut.result())

    # 保序：按原 items 顺序输出（as_completed 无序）
    def key_fn(x: dict[str, Any]) -> tuple:
        return (
            x.get("eval_corpus", x.get("dataset", "")),
            x.get("src_lang", ""),
            x.get("tgt_lang", ""),
            str(x.get("sample_id", "")),
        )
    by_key = {key_fn(r): r for r in lock_results}
    for it in items:
        results.append(by_key[key_fn(it)])

    hyp_path = run_dir / "hypotheses.jsonl"
    with open(hyp_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 输出格式：按数据集(corpus)->语言对(pair) 分层，便于对比
    metrics: dict[str, Any] = {
        "by_corpus": {},  # corpus -> { by_pair: {pair: {...}}, overall: {...} }
        "overall": {},  # 全部样本总体
    }

    # BLEU 按 eval_pair + corpus 聚合
    try:
        import sacrebleu
    except ImportError as e:
        print(f"sacrebleu 未安装: {e}", file=sys.stderr)
        return 1

    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in results:
        key = f"{r.get('eval_corpus','?')}|{r.get('eval_pair') or r['src_lang'] + '->' + r['tgt_lang']}"
        grouped[key].append((r["hypothesis"], r["reference_text"], r["source_text"]))

    bleu_rows: list[dict[str, Any]] = []
    corpus_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, triples in grouped.items():
        hyps = [t[0] for t in triples]
        refs = [t[1] for t in triples]
        corp, pair = key.split("|", 1)
        bleu_score, bleu_tok = corpus_bleu_with_fallbacks(
            hyps, refs, corp, pair, str(args.bleu_tokenize)
        )
        if corp not in metrics["by_corpus"]:
            metrics["by_corpus"][corp] = {"by_pair": {}, "overall": {}}
        row = {
            "corpus": corp,
            "pair": pair,
            "bleu": bleu_score,
            "bleu_tokenizer": bleu_tok,
            "num": len(hyps),
        }
        bleu_rows.append(row)
        corpus_to_rows[corp].append(row)
        metrics["by_corpus"][corp]["by_pair"][pair] = {
            "bleu": row["bleu"],
            "bleu_tokenizer": bleu_tok,
            "num": row["num"],
        }

    # overall BLEU：语向混用不同分词器，不宜对整表跑一次 corpus_bleu；按各语言对样本数加权平均
    total_n = sum(r["num"] for r in bleu_rows)
    metrics["overall"]["bleu"] = (
        float(sum(r["bleu"] * r["num"] for r in bleu_rows) / total_n) if total_n else 0.0
    )

    corpus_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        corpus_group[str(r.get("eval_corpus", "?"))].append(r)
    for corp, rows_c in corpus_to_rows.items():
        n_c = sum(r["num"] for r in rows_c)
        bleu_c = float(sum(r["bleu"] * r["num"] for r in rows_c) / n_c) if n_c else 0.0
        if corp not in metrics["by_corpus"]:
            metrics["by_corpus"][corp] = {"by_pair": {}, "overall": {}}
        metrics["by_corpus"][corp]["overall"]["bleu"] = bleu_c
    for corp, rr in corpus_group.items():
        if corp not in metrics["by_corpus"]:
            metrics["by_corpus"][corp] = {"by_pair": {}, "overall": {}}
        metrics["by_corpus"][corp]["overall"]["num"] = len(rr)

    # COMET：只跑一次全量预测，再按 (corpus, pair) 聚合均值；无 GPU 时 gpus=0
    comet_scores_by_key: dict[str, float] = {}
    if comet_ckpt and torch_mod is not None and comet_load_from_checkpoint is not None:
        gpus = 1 if torch_mod.cuda.is_available() else 0
        comet_model = comet_load_from_checkpoint(comet_ckpt)

        all_data = [
            {"src": r["source_text"], "mt": r["hypothesis"], "ref": r["reference_text"]}
            for r in results
        ]
        out_all = comet_model.predict(all_data, batch_size=args.comet_batch_size, gpus=gpus)
        scores_all = out_all.get("scores", [])
        if not isinstance(scores_all, list) or len(scores_all) != len(results):
            print("COMET 输出 scores 长度异常，将跳过 COMET。", file=sys.stderr)
        else:
            # 写 overall
            metrics["overall"]["comet"] = float(sum(scores_all) / max(1, len(scores_all)))

            # 聚合：按 key 统计 sum / count
            sums: dict[str, float] = defaultdict(float)
            cnts: dict[str, int] = defaultdict(int)
            sums_corpus: dict[str, float] = defaultdict(float)
            cnts_corpus: dict[str, int] = defaultdict(int)

            for r, s in zip(results, scores_all, strict=True):
                corp = str(r.get("eval_corpus", "?"))
                pair = str(r.get("eval_pair") or (r["src_lang"] + "->" + r["tgt_lang"]))
                k = f"{corp}|{pair}"
                try:
                    sv = float(s)
                except Exception:
                    continue
                sums[k] += sv
                cnts[k] += 1
                sums_corpus[corp] += sv
                cnts_corpus[corp] += 1

            for k, total in sums.items():
                comet_scores_by_key[k] = float(total / max(1, cnts[k]))
                corp, pair = k.split("|", 1)
                if corp not in metrics["by_corpus"]:
                    metrics["by_corpus"][corp] = {"by_pair": {}, "overall": {}}
                if pair not in metrics["by_corpus"][corp]["by_pair"]:
                    metrics["by_corpus"][corp]["by_pair"][pair] = {}
                metrics["by_corpus"][corp]["by_pair"][pair]["comet"] = comet_scores_by_key[k]

            for corp, total in sums_corpus.items():
                if corp not in metrics["by_corpus"]:
                    metrics["by_corpus"][corp] = {"by_pair": {}, "overall": {}}
                metrics["by_corpus"][corp]["overall"]["comet"] = float(total / max(1, cnts_corpus[corp]))

    metrics["bleu_tokenize_policy"] = str(args.bleu_tokenize)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["corpus", "pair", "bleu", "bleu_tokenizer", "comet", "num"],
        )
        w.writeheader()
        for row in bleu_rows:
            key = f"{row['corpus']}|{row['pair']}"
            w.writerow(
                {
                    **row,
                    "comet": comet_scores_by_key.get(key, ""),
                }
            )

    # 额外输出：每个数据集单独一张表
    for corp, rows in corpus_to_rows.items():
        p = run_dir / f"metrics_{corp}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["pair", "bleu", "bleu_tokenizer", "comet", "num"],
            )
            w.writeheader()
            for row in sorted(rows, key=lambda x: x["pair"]):
                key = f"{corp}|{row['pair']}"
                w.writerow(
                    {
                        "pair": row["pair"],
                        "bleu": row["bleu"],
                        "bleu_tokenizer": row["bleu_tokenizer"],
                        "comet": comet_scores_by_key.get(key, ""),
                        "num": row["num"],
                    }
                )

    print(f"完成。输出目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
