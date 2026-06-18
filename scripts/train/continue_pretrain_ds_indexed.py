#!/usr/bin/env python3
"""Continue causal LM pretraining from tokenized DeepSpeed/Megatron indexed data."""

from __future__ import annotations

import argparse
import bisect
import glob
import importlib
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


LOG = logging.getLogger("continue_pretrain_ds_indexed")


class IndexedDatasetOpenError(RuntimeError):
    pass


def _module_candidates() -> Iterable[str]:
    yield "megatron.core.datasets.indexed_dataset"
    yield "megatron.data.indexed_dataset"
    yield "deepspeed.runtime.data_pipeline.data_routing.indexed_dataset"


def _open_indexed_dataset(prefix: str):
    """Open a Megatron-style indexed dataset using whichever package is installed."""
    errors: list[str] = []
    for module_name in _module_candidates():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: import failed: {exc}")
            continue

        for class_name in ("MMapIndexedDataset", "IndexedDataset"):
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            try:
                return cls(prefix)
            except Exception as exc:
                errors.append(f"{module_name}.{class_name}({prefix!r}): {exc}")

    details = "\n  - ".join(errors) if errors else "no indexed dataset module candidates"
    raise IndexedDatasetOpenError(
        "Cannot open DeepSpeed/Megatron indexed dataset. Install Megatron-LM or a "
        "package that provides MMapIndexedDataset, then rerun.\n  - " + details
    )


def _prefix_from_data_file(data_file: str) -> str:
    if data_file.endswith(".bin"):
        return data_file[:-4]
    return data_file


def _safe_link_name(data_file: str, data_root: str) -> str:
    rel = os.path.relpath(data_file, data_root)
    return rel.replace(os.sep, "__").replace(".ds", "")


def _force_symlink(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        existing = os.readlink(dst) if os.path.islink(dst) else None
        if existing == src:
            return
        os.unlink(dst)
    os.symlink(src, dst)


def discover_dataset_prefixes(data_root: str, index_cache_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_root, "**", "train.c*.ds"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_root, "**", "*.ds"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .ds files found under {data_root}")

    Path(index_cache_dir).mkdir(parents=True, exist_ok=True)
    prefixes: list[str] = []
    for file_name in files:
        index_file = file_name + ".index"
        if not os.path.exists(index_file):
            LOG.warning("Skipping %s because %s is missing", file_name, index_file)
            continue
        link_prefix = os.path.join(index_cache_dir, _safe_link_name(file_name, data_root))
        _force_symlink(file_name, link_prefix + ".bin")
        _force_symlink(index_file, link_prefix + ".idx")
        prefixes.append(_prefix_from_data_file(link_prefix + ".bin"))

    if not prefixes:
        raise FileNotFoundError(f"No .ds files with matching .ds.index files found under {data_root}")
    return prefixes


class PackedIndexedCausalLMDataset(Dataset):
    """Packs indexed token documents into fixed-length causal LM blocks."""

    def __init__(
        self,
        prefixes: list[str],
        seq_len: int,
        eos_token_id: int,
        add_eos_between_docs: bool = True,
    ):
        self.datasets = [_open_indexed_dataset(prefix) for prefix in prefixes]
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.add_eos_between_docs = add_eos_between_docs
        self.cumulative_docs: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_docs.append(total)
        self.num_docs = total

        lengths: list[int] = []
        for dataset in self.datasets:
            sizes = getattr(dataset, "sizes", None)
            if sizes is None:
                sizes = [len(dataset[i]) for i in range(len(dataset))]
            lengths.extend(int(x) + (1 if add_eos_between_docs else 0) for x in sizes)
        self.cumulative_tokens = np.cumsum(np.asarray(lengths, dtype=np.int64))
        self.total_tokens = int(sum(lengths))
        self.num_blocks = self.total_tokens // seq_len
        if self.num_blocks <= 0:
            raise ValueError(f"Dataset has only {self.total_tokens} tokens, less than seq_len={seq_len}")

    def __len__(self) -> int:
        return self.num_blocks

    def _get_doc(self, global_doc_idx: int) -> np.ndarray:
        dataset_idx = bisect.bisect_right(self.cumulative_docs, global_doc_idx)
        previous = 0 if dataset_idx == 0 else self.cumulative_docs[dataset_idx - 1]
        doc = self.datasets[dataset_idx][global_doc_idx - previous]
        return np.asarray(doc, dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_start = index * self.seq_len
        doc_idx = bisect.bisect_right(self.cumulative_tokens, target_start)
        seen = 0 if doc_idx == 0 else int(self.cumulative_tokens[doc_idx - 1])

        pieces: list[np.ndarray] = []
        offset = target_start - seen
        while sum(len(piece) for piece in pieces) < self.seq_len:
            doc = self._get_doc(doc_idx)
            if self.add_eos_between_docs:
                eos = np.asarray([self.eos_token_id], dtype=np.int64)
                doc = np.concatenate([doc, eos])
            if offset:
                doc = doc[offset:]
                offset = 0
            pieces.append(doc)
            doc_idx = (doc_idx + 1) % self.num_docs

        input_ids = np.concatenate(pieces)[: self.seq_len].astype(np.int64, copy=False)
        tensor = torch.from_numpy(input_ids.copy())
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--index_cache_dir", default=None)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    set_seed(args.seed)

    index_cache_dir = args.index_cache_dir or os.path.join(args.output_dir, "indexed_dataset_links")
    prefixes = discover_dataset_prefixes(args.data_root, index_cache_dir=index_cache_dir)
    LOG.info("Found %d dataset shards under %s", len(prefixes), args.data_root)
    for prefix in prefixes:
        LOG.info("  %s", prefix)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has neither eos_token_id nor pad_token_id")

    dataset = PackedIndexedCausalLMDataset(prefixes=prefixes, seq_len=args.seq_len, eos_token_id=eos_token_id)
    LOG.info("Packed dataset: docs=%d tokens=%d blocks=%d seq_len=%d", dataset.num_docs, dataset.total_tokens, len(dataset), args.seq_len)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        deepspeed=args.deepspeed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="adamw_torch",
        ddp_timeout=180000000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    LOG.info("Saved continued-pretraining model to %s", args.output_dir)


if __name__ == "__main__":
    main()
