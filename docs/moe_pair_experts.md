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

If FineWeb synthetic data is available, build the sentence-level mixed MoE data:

```bash
python scripts/moe/build_mixed_moe_for_llamafactory.py
```

The mix is controlled by:

```text
training/moe_data_mix_config.json
```

Default sources are:

- `nllb`: enabled, up to `100000` rows per direction.
- `fineweb_synth`: enabled but optional, up to `100000` rows per direction.
  If a direction has no FineWeb synthetic file, it contributes `0` rows and is
  skipped without failing.
- `dictionary`: reserved but disabled with `limit=0`.

`enabled=true` means the builder checks a source. `required=true` means
`--strict` fails if that source file is missing. Currently `nllb` is required,
while `fineweb_synth` is optional, so different language pairs can use different
available sources.

The mixed files are written to:

```text
training/data/multilingual/mixed_moe/mixed_moe_<src>__<tgt>.jsonl
```

To train on mixed data instead of pure NLLB, regenerate expert assets with:

```bash
python scripts/moe/generate_pair_expert_assets.py \
  --training-dataset-prefix mixed_moe \
  --training-data-subdir multilingual/mixed_moe \
  --training-file-prefix mixed_moe
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

## Single SFT Baseline

To compare pair-level MoE against one ordinary LoRA adapter, merge all
directions into a single SFT dataset:

```bash
python scripts/moe/build_single_sft_for_llamafactory.py --strict
```

This reuses `training/moe_data_mix_config.json`, so the non-MoE baseline sees
the same NLLB/FineWeb source limits as the MoE mixed data path. Dictionary data
is still reserved but disabled by default.

Train the single adapter:

```bash
bash scripts/run/run_train_single_sft.sh
```

Serve it with vLLM:

```bash
bash scripts/serve/serve_vllm_qwen3_8b_single_sft_lora.sh
```

Then evaluate using the same FLORES/COMET pipeline with:

```bash
export SERVED_MODEL_NAME=qwen3_8b_all_directions_sft
export EVAL_MODEL_TAG=qwen3_8b_single_sft
export EVAL_MODEL_FAMILY=qwen3
bash scripts/run/run_eval_baseline.sh
```

## Inference Routing

Serve the base model with all LoRA adapters enabled, then route each request by
`src_lang` and `tgt_lang` using `training/moe_router_manifest.json`.

With vLLM, expose each LoRA adapter as a separate model name via
`--enable-lora --lora-modules`. The router then calls the adapter model name
matching the language pair.

```bash
conda activate lowres-serve
bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
```

The serve script reads:

```text
training/moe_router_manifest.json
```

and registers every `adapter_name=adapter_path` entry. If the base model or
adapter paths differ on the GPU machine, override them:

```bash
MODEL_PATH=/path/to/Qwen3-8B \
MANIFEST=/path/to/moe_router_manifest.json \
bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
```

Translate through the router:

```bash
python scripts/moe/translate_with_moe_router.py \
  --src-lang tha_Thai \
  --tgt-lang zho_Hans \
  --text "สวัสดีครับ" \
  --print-model
```

List supported routes:

```bash
python scripts/moe/translate_with_moe_router.py --list-pairs
```

This is pair-level MoE: the expert choice is made once per request. It is
intentional for this experiment because the translation direction is known
before decoding, and it avoids invasive changes to Qwen/LLaMA-Factory internals.
