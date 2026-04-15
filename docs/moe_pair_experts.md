# Pair-Level MoE Experts

This experiment uses routed LoRA experts instead of modifying the base
Transformer into a token-level MoE. It is the lowest-risk MoE variant for the
current repository because training already goes through LLaMA-Factory LoRA.

Default expert layout has 20 direction-specific experts:

- `spa_Latn -> zho_Hans`
- `zho_Hans -> spa_Latn`
- `ind_Latn -> zho_Hans`
- `zho_Hans -> ind_Latn`
- `vie_Latn -> zho_Hans`
- `zho_Hans -> vie_Latn`
- `tha_Thai -> zho_Hans`
- `zho_Hans -> tha_Thai`
- `tgl_Latn -> zho_Hans`
- `zho_Hans -> tgl_Latn`
- `spa_Latn -> eng_Latn`
- `eng_Latn -> spa_Latn`
- `ind_Latn -> eng_Latn`
- `eng_Latn -> ind_Latn`
- `vie_Latn -> eng_Latn`
- `eng_Latn -> vie_Latn`
- `tha_Thai -> eng_Latn`
- `eng_Latn -> tha_Thai`
- `tgl_Latn -> eng_Latn`
- `eng_Latn -> tgl_Latn`

## Generate Assets

```bash
conda activate lowres
python scripts/moe/generate_pair_expert_assets.py
```

This writes:

- `training/moe_pair_limits.json`
- `training/moe_dataset_info.snippet.json`
- `training/data/dataset_info.json`
- `training/moe_router_manifest.json`
- `training/moe_experts/llamafactory_qwen3_8b_moe_*.yaml`

## Prepare Training Data

```bash
conda activate lowres
python scripts/prepare/prepare_nllb_for_llamafactory.py \
  --pairs-config training/moe_pair_limits.json \
  --export-from-config \
  --out-subdir multilingual/nllb_moe
```

The generator creates or updates `training/data/dataset_info.json`. The
separate `training/moe_dataset_info.snippet.json` is kept for copying the same
registrations to another LLaMA-Factory checkout if needed.

## Train Experts

Run on a GPU machine with LLaMA-Factory installed:

```bash
conda activate lowres
bash scripts/run/run_train_moe_experts.sh
```

Each config trains one LoRA adapter and writes it under:

```text
/root/lowres_new/models/Qwen3-8B_moe_pair_experts/qwen3_8b_moe_<src>_to_<tgt>
```

Adjust paths during generation if the GPU machine uses a different repository
location:

```bash
python scripts/moe/generate_pair_expert_assets.py \
  --model-name-or-path /path/to/base/Qwen3-8B \
  --dataset-dir /path/to/lowres_new/training/data \
  --output-root /path/to/models/Qwen3-8B_moe_pair_experts
```

## Inference Routing

Serve the base model with all LoRA adapters enabled, then route each request by
`src_lang` and `tgt_lang` using `training/moe_router_manifest.json`.

With vLLM, the simple serving pattern is to expose each LoRA adapter as a
separate model name via `--enable-lora --lora-modules`. The router then calls
the adapter model name matching the language pair.

This is pair-level MoE: the expert choice is made once per request. It is
intentional for this experiment because the translation direction is known
before decoding, and it avoids invasive changes to Qwen/LLaMA-Factory internals.
