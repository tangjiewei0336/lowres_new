#!/usr/bin/env python3
"""
Semantic-reward fine-tuning for translation (REINFORCE + optional SFT anchor).

Instead of token-level NLL on a single reference, optimize frozen cross-lingual
embeddings (LaBSE / multilingual-E5) and optional fluency from a frozen LM.

Reward modes:
  - align:      cos(E(x), E(y))           direct cross-lingual alignment
  - roundtrip:  cos(E(x), E(x_hat))       back-translation consistency
  - both:       weighted sum of the above

Example (zh->en, align + fluency, mixed with SFT):
  conda activate lowres
  python scripts/train/semantic_rl_train.py \\
    --model-path models/Qwen3-8B_latest \\
    --data training/data/multilingual/nllb/nllb_mt_zho_Hans__eng_Latn.jsonl \\
    --src-lang zho_Hans --tgt-lang eng_Latn \\
    --reward-mode align \\
    --embedding-model intfloat/multilingual-e5-base \\
    --fluency-model models/Qwen3-8B_latest \\
    --fluency-weight 0.1 \\
    --sft-weight 0.5 \\
    --output-dir models/Qwen3-8B_zho_en_semantic_rl_lora
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from flores_lang_zh import english_translation_instruction  # noqa: E402
from train.semantic_reward import (  # noqa: E402
    FrozenFluencyScorer,
    FrozenSemanticEncoder,
    combine_rewards,
    reinforce_loss,
    sft_cross_entropy,
)


@dataclass(frozen=True)
class TranslationExample:
    instruction: str
    source: str
    reference: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_text_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    src = row.get("input") or row.get("source_text") or row.get("src") or row.get("source")
    tgt = row.get("output") or row.get("target_text") or row.get("tgt") or row.get("target")
    if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
        return src.strip(), tgt.strip()
    return None


def load_translation_jsonl(
    path: Path,
    *,
    src_lang: str,
    tgt_lang: str,
    limit: int = 0,
) -> list[TranslationExample]:
    instruction = english_translation_instruction(src_lang, tgt_lang)
    examples: list[TranslationExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pair = resolve_text_pair(row)
            if pair is None:
                continue
            src, tgt = pair
            examples.append(TranslationExample(instruction=instruction, source=src, reference=tgt))
            if limit and len(examples) >= limit:
                break
    return examples


class TranslationJsonlDataset(Dataset[TranslationExample]):
    def __init__(self, examples: list[TranslationExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TranslationExample:
        return self.examples[idx]


def build_user_content(instruction: str, source: str) -> str:
    return f"{instruction}\n\n{source}"


def apply_chat_prompt(tokenizer: Any, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class TranslationPolicy(nn.Module):
    def __init__(
        self,
        model_path: str,
        *,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora_cfg)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to_device(self, device: str | torch.device) -> TranslationPolicy:
        self.model.to(device)
        return self

    def tokenize_prompt(self, prompt: str) -> dict[str, Tensor]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in encoded.items()}

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        encoded = self.tokenize_prompt(prompt)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = encoded["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def sequence_logprob(self, prompt: str, response: str) -> Tensor:
        """Sum of log-probs over response tokens (differentiable w.r.t. LoRA)."""
        full_text = prompt + response
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        input_ids = full["input_ids"].to(self.device)
        attention_mask = full["attention_mask"].to(self.device)
        prompt_len = len(prompt_ids)
        if input_ids.shape[1] <= prompt_len:
            return torch.zeros((), device=self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        target_ids = input_ids[:, prompt_len:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprob = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return token_logprob.sum()

    def sft_logits(self, prompt: str, reference: str) -> tuple[Tensor, Tensor]:
        full_text = prompt + reference + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        input_ids = full["input_ids"].to(self.device)
        attention_mask = full["attention_mask"].to(self.device)
        prompt_len = len(prompt_ids)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits, labels


class RewardTracker:
    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum = momentum
        self.value: float | None = None

    def update(self, batch_mean: float) -> float:
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * batch_mean
        return self.value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Semantic-reward translation fine-tuning.")
    ap.add_argument("--model-path", required=True, help="Base causal LM path (e.g. Qwen3-8B).")
    ap.add_argument("--data", required=True, help="Alpaca/jsonl MT file with input/output fields.")
    ap.add_argument("--src-lang", required=True, help="FLORES source code, e.g. zho_Hans.")
    ap.add_argument("--tgt-lang", required=True, help="FLORES target code, e.g. eng_Latn.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reward-mode", choices=("align", "roundtrip", "both"), default="align")
    ap.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--embedding-device", default=None)
    ap.add_argument("--fluency-model", default=None, help="Optional frozen LM for target fluency.")
    ap.add_argument("--fluency-device", default=None)
    ap.add_argument("--align-weight", type=float, default=1.0)
    ap.add_argument("--roundtrip-weight", type=float, default=1.0)
    ap.add_argument("--fluency-weight", type=float, default=0.0)
    ap.add_argument("--sft-weight", type=float, default=0.5, help="CE anchor on reference translation.")
    ap.add_argument("--rl-weight", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="Max training rows (0 = all).")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def batch_iterator(items: list[TranslationExample], batch_size: int) -> Iterator[list[TranslationExample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def train_one_batch(
    policy: TranslationPolicy,
    semantic_encoder: FrozenSemanticEncoder,
    fluency_scorer: FrozenFluencyScorer | None,
    batch: list[TranslationExample],
    *,
    src_lang: str,
    tgt_lang: str,
    reward_mode: str,
    align_weight: float,
    roundtrip_weight: float,
    fluency_weight: float,
    sft_weight: float,
    rl_weight: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reward_baseline: RewardTracker,
) -> dict[str, float]:
    reverse_instruction = english_translation_instruction(tgt_lang, src_lang)
    sources = [ex.source for ex in batch]
    references = [ex.reference for ex in batch]
    prompts = [
        apply_chat_prompt(policy.tokenizer, build_user_content(ex.instruction, ex.source))
        for ex in batch
    ]

    translations: list[str] = []
    log_probs: list[Tensor] = []
    sft_losses: list[Tensor] = []

    for ex, prompt in zip(batch, prompts):
        translation = policy.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        translations.append(translation)
        log_probs.append(policy.sequence_logprob(prompt, translation))
        if sft_weight > 0:
            logits, labels = policy.sft_logits(prompt, ex.reference)
            sft_losses.append(sft_cross_entropy(logits, labels))

    back_translations: list[str] | None = None
    if reward_mode in ("roundtrip", "both"):
        back_translations = []
        for translation in translations:
            reverse_prompt = apply_chat_prompt(
                policy.tokenizer,
                build_user_content(reverse_instruction, translation),
            )
            back_translations.append(
                policy.generate_text(
                    reverse_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )

    align_sim = semantic_encoder.cosine_similarity(sources, translations) if reward_mode in ("align", "both") else None
    roundtrip_sim = (
        semantic_encoder.cosine_similarity(sources, back_translations)
        if reward_mode in ("roundtrip", "both") and back_translations is not None
        else None
    )
    fluency_nll = (
        fluency_scorer.mean_nll(translations)
        if fluency_scorer is not None and fluency_weight > 0
        else None
    )

    if reward_mode == "align":
        roundtrip_weight = 0.0
    elif reward_mode == "roundtrip":
        align_weight = 0.0

    rewards = combine_rewards(
        align_sim=align_sim,
        roundtrip_sim=roundtrip_sim,
        fluency_nll=fluency_nll,
        align_weight=align_weight,
        roundtrip_weight=roundtrip_weight,
        fluency_weight=fluency_weight,
    )
    baseline = reward_baseline.update(float(rewards.mean().item()))
    batch_log_probs = torch.stack(log_probs)
    pg_loss = reinforce_loss(batch_log_probs, rewards, baseline=torch.full_like(rewards, baseline))
    loss = rl_weight * pg_loss
    if sft_weight > 0 and sft_losses:
        loss = loss + sft_weight * torch.stack(sft_losses).mean()

    loss.backward()

    metrics: dict[str, float] = {
        "loss": float(loss.detach().item()),
        "pg_loss": float(pg_loss.detach().item()),
        "reward": float(rewards.mean().item()),
        "reward_baseline": baseline,
    }
    if align_sim is not None:
        metrics["align_sim"] = float(align_sim.mean().item())
    if roundtrip_sim is not None:
        metrics["roundtrip_sim"] = float(roundtrip_sim.mean().item())
    if fluency_nll is not None:
        metrics["fluency_nll"] = float(fluency_nll.mean().item())
    if sft_losses:
        metrics["sft_loss"] = float(torch.stack(sft_losses).mean().detach().item())
    return metrics


def save_policy(policy: TranslationPolicy, output_dir: Path, step: int) -> None:
    save_path = output_dir / f"checkpoint-{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(save_path)
    policy.tokenizer.save_pretrained(save_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    examples = load_translation_jsonl(
        data_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        limit=args.limit,
    )
    if not examples:
        raise RuntimeError(f"No training examples loaded from {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    policy = TranslationPolicy(
        args.model_path,
        lora_rank=args.lora_rank,
        gradient_checkpointing=True,
    ).to_device(args.device)
    policy.model.print_trainable_parameters()

    semantic_encoder = FrozenSemanticEncoder(
        args.embedding_model,
        device=args.embedding_device or args.device,
    )
    fluency_scorer = None
    if args.fluency_model and args.fluency_weight > 0:
        fluency_scorer = FrozenFluencyScorer(
            args.fluency_model,
            device=args.fluency_device or args.device,
        )

    optimizer = torch.optim.AdamW(
        (p for p in policy.model.parameters() if p.requires_grad),
        lr=args.learning_rate,
    )
    reward_baseline = RewardTracker()

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}

    for epoch in range(args.epochs):
        random.shuffle(examples)
        pbar = tqdm(batch_iterator(examples, args.batch_size), desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            metrics = train_one_batch(
                policy,
                semantic_encoder,
                fluency_scorer,
                batch,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                reward_mode=args.reward_mode,
                align_weight=args.align_weight,
                roundtrip_weight=args.roundtrip_weight,
                fluency_weight=args.fluency_weight,
                sft_weight=args.sft_weight,
                rl_weight=args.rl_weight,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                reward_baseline=reward_baseline,
            )
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_steps == 0:
                    avg = {k: v / args.log_steps for k, v in running.items()}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in avg.items()})
                    running.clear()
                if global_step % args.save_steps == 0:
                    save_policy(policy, output_dir, global_step)

        if len(examples) % (args.batch_size * args.grad_accum) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    save_policy(policy, output_dir, global_step)
    print(f"Done. Final checkpoint: {output_dir / f'checkpoint-{global_step}'}")


if __name__ == "__main__":
    main()
