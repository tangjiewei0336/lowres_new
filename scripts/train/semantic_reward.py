"""Frozen semantic encoder and fluency scorers for translation RL rewards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _needs_e5_prefix(model_name: str) -> bool:
    name = model_name.lower()
    return "e5" in name or "multilingual-e5" in name


@dataclass
class SemanticRewardConfig:
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str | None = None
    e5_prefix: str = "query: "
    fluency_model: str | None = None
    fluency_device: str | None = None


class FrozenSemanticEncoder(nn.Module):
    """Cross-lingual sentence encoder kept frozen during policy training."""

    def __init__(self, model_name: str, *, device: str | None = None, e5_prefix: str = "query: ") -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.use_e5_prefix = _needs_e5_prefix(model_name)
        self.e5_prefix = e5_prefix
        self.encoder = SentenceTransformer(model_name, device=device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _prepare(self, texts: Sequence[str]) -> list[str]:
        if not self.use_e5_prefix:
            return list(texts)
        return [f"{self.e5_prefix}{text}" for text in texts]

    @override
    @torch.no_grad()
    def forward(self, texts: Sequence[str], *, batch_size: int = 32) -> Tensor:
        return self.encode(texts, batch_size=batch_size)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> Tensor:
        if not texts:
            return torch.empty(0, 0)
        vectors = self.encoder.encode(
            self._prepare(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return vectors

    @torch.no_grad()
    def cosine_similarity(self, left: Sequence[str], right: Sequence[str], *, batch_size: int = 32) -> Tensor:
        if len(left) != len(right):
            raise ValueError(f"Batch size mismatch: {len(left)} vs {len(right)}")
        left_vec = self.encode(left, batch_size=batch_size)
        right_vec = self.encode(right, batch_size=batch_size)
        return (left_vec * right_vec).sum(dim=-1)


class FrozenFluencyScorer(nn.Module):
    """Average token NLL from a frozen causal LM; lower NLL means higher fluency."""

    def __init__(self, model_name: str, *, device: str | None = None, trust_remote_code: bool = True) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        if device is not None:
            self.model = self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @override
    @torch.no_grad()
    def forward(self, texts: Sequence[str], *, max_length: int = 256) -> Tensor:
        return self.mean_nll(texts, max_length=max_length)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def mean_nll(self, texts: Sequence[str], *, max_length: int = 256) -> Tensor:
        if not texts:
            return torch.empty(0, device=self.device)
        per_sample: list[Tensor] = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            outputs = self.model(**encoded, labels=encoded["input_ids"])
            per_sample.append(outputs.loss)
        return torch.stack(per_sample)


def combine_rewards(
    *,
    align_sim: Tensor | None = None,
    roundtrip_sim: Tensor | None = None,
    fluency_nll: Tensor | None = None,
    align_weight: float = 1.0,
    roundtrip_weight: float = 0.0,
    fluency_weight: float = 0.0,
) -> Tensor:
    """Higher returned reward is better."""
    if align_sim is None and roundtrip_sim is None and fluency_nll is None:
        raise ValueError("At least one reward component must be provided.")
    device = None
    for tensor in (align_sim, roundtrip_sim, fluency_nll):
        if tensor is not None:
            device = tensor.device
            break
    assert device is not None
    reward = torch.zeros(len(next(t for t in (align_sim, roundtrip_sim, fluency_nll) if t is not None)), device=device)
    if align_sim is not None:
        reward = reward + align_weight * align_sim
    if roundtrip_sim is not None:
        reward = reward + roundtrip_weight * roundtrip_sim
    if fluency_nll is not None:
        reward = reward - fluency_weight * fluency_nll
    return reward


def reinforce_loss(log_probs: Tensor, rewards: Tensor, *, baseline: Tensor | None = None) -> Tensor:
    """Policy gradient loss. log_probs shape: [batch]; rewards shape: [batch]."""
    if baseline is None:
        advantage = rewards - rewards.mean()
    else:
        advantage = rewards - baseline
    return -(advantage.detach() * log_probs).mean()


def sft_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
