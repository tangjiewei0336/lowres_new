# 低资源多语评估与数据准备（ModelScope + vLLM）

## Conda 环境（必须）

本项目分为两个环境，避免 `vllm` 与 `unbabel-comet` 的 `protobuf` 依赖冲突：

- **评测/数据处理环境**：`lowres`（BLEU + COMET + 数据准备）
- **部署环境**：`lowres-serve`（仅 vLLM 服务）

在仓库根目录执行：

```bash
# 评测/数据
conda create -n lowres python=3.12 -y
conda activate lowres
pip install -r requirements.txt

# vLLM 服务（单独环境）
conda create -n lowres-serve python=3.12 -y
conda activate lowres-serve
pip install -r requirements-serve.txt
```

可选：统一缓存目录

```bash
export MODELSCOPE_CACHE="$PWD/datasets/cache/modelscope"
```

常用模型和 COMET 评测模型可提前下载到 `models/`：

```bash
python scripts/download_models_to_models_dir.py --only qwen3_8b
python scripts/download_models_to_models_dir.py --only comet_wmt22_da
```

## 脚本流程

1. **语法与依赖**：`python scripts/check_syntax_and_imports.py`
2. **ModelScope 连通性与小样本加载**：`python scripts/check_modelscope_download.py`
3. **准备数据**（FLORES + NTREX，生成 `datasets/processed/*.jsonl` 与各文件前 50 条预览）：`python scripts/prepare/prepare_datasets.py`
4. **展开评估清单**：`python scripts/expand_language_pairs.py` → `datasets/eval_manifest.json` 与 `datasets/processed/eval_items_all.jsonl`
5. **启动 vLLM**（另开终端，与评估分离）：见 `scripts/serve/serve_vllm_*.sh`（使用 `lowres-serve`）
6. **评估**：编辑并运行 `scripts/run/run_eval_baseline.sh` 或 `scripts/run/run_eval_after_ccmatrix.sh`

Shell 封装脚本默认使用 `conda` 激活环境：评测脚本用 `lowres`，vLLM serve 脚本建议在 `lowres-serve` 下运行。

## 配置说明

- 评估超参：`evaluation_config.json`
- ModelScope 模型/数据集 ID：`modelscope_sources.json`（数据集以页面为准，必要时修改）
- FLORES ↔ NTREX 语言映射：`datasets/flores_ntrex_mapping.json`（无映射的 NTREX 方向会跳过）

## prepare 数据规格

`python scripts/prepare/prepare_datasets.py` 只准备评测集，不准备 SFT 训练集。脚本读取
`evaluation_config.json` 的 `split`、`language_pair_groups`、`bidirectional`，并从
`modelscope_sources.json` 下载/加载以下数据集。

### prepare_datasets.py: FLORES

- 数据源：`modelscope_sources.json` 中 `datasets.flores.dataset_id`，默认 `facebook/flores`。
- 默认 split：`evaluation_config.json` 中的 `split`，当前为 `dev`。
- 默认语言：`eng_Latn`、`zho_Hans`、`spa_Latn`、`ind_Latn`、`vie_Latn`、`tha_Thai`、`tgl_Latn`。
- 配对规则：只做 `language_pair_groups` 之间的跨组笛卡尔积，不做同组内两两组合；`bidirectional=true` 时写出双向。
- 当前默认方向数：第 1 组 `["eng_Latn", "zho_Hans"]` × 第 2 组 7 个语言，去掉 `eng_Latn->eng_Latn` 和 `zho_Hans->zho_Hans` 后双向，共 24 个方向。
- 每个方向条数：取源/目标语言共同 `sample_id` 的交集；FLORES dev 通常约 997 条/方向，以脚本打印为准。
- 输出文件：`datasets/processed/flores_<split>_<src>__<tgt>.jsonl`。
- 预览文件：`datasets/previews/flores_<split>_<src>__<tgt>.jsonl.preview_50.jsonl`。

每行 JSONL 字段：

```json
{
  "dataset": "flores",
  "split": "dev",
  "src_lang": "spa_Latn",
  "tgt_lang": "zho_Hans",
  "sample_id": "...",
  "source_text": "...",
  "reference_text": "..."
}
```

### NTREX

- 数据源：`modelscope_sources.json` 中 `datasets.ntrex.dataset_id`，默认 `MTEB/NTREX`。
- 用途：补充英语中心评测集。
- 方向规则：只生成 `eng_Latn -> tgt`，且 `tgt` 必须是在跨组配对里实际出现的非英语目标语。
- 当前默认目标语：`zho_Hans`、`spa_Latn`、`ind_Latn`、`vie_Latn`、`tha_Thai`、`tgl_Latn`。
- 语言列映射：见 `datasets/flores_ntrex_mapping.json`；其中 `tgl_Latn` 在 NTREX 中映射为 `fil_Latn`。
- split 探测顺序：`test`、`train`、`validation`、`dev`、默认 split；脚本使用第一个能解析出非空数据的 split。
- 每个方向条数：取决于 ModelScope 上 `MTEB/NTREX` 的实际表结构和 split；以脚本打印为准。
- 输出文件：`datasets/processed/ntrex_<eval_split>_eng_Latn__<tgt>.jsonl`。文件名里的 `<eval_split>` 来自 `evaluation_config.json`，行内 `split` 字段记录实际解析到的 NTREX split。
- 预览文件：`datasets/previews/ntrex_<eval_split>_eng_Latn__<tgt>.jsonl.preview_50.jsonl`。

每行 JSONL 字段：

```json
{
  "dataset": "ntrex",
  "split": "test",
  "src_lang": "eng_Latn",
  "tgt_lang": "zho_Hans",
  "sample_id": "test:0",
  "source_text": "...",
  "reference_text": "..."
}
```

### 合并后的评估清单

`python scripts/expand_language_pairs.py` 会读取 `datasets/processed/` 下已生成的 FLORES 和 NTREX
文件，写出：

- `datasets/processed/eval_items_all.jsonl`
- `datasets/eval_manifest.json`

合并时会给每条样本补充：

- `eval_corpus`：`flores` 或 `ntrex`
- `eval_pair`：如 `spa_Latn->zho_Hans`

限量规则：

- `per_pair_limit`：对每个 `(eval_corpus, eval_pair)` 独立抽样限量；当前默认 200。
- `limit`：在 per-pair 限量之后再做全局限量；当前默认 `null`。

### prepare_ccmatrix_for_llamafactory.py

用途：准备 LLaMA-Factory SFT 翻译数据，输出 Alpaca JSONL。

- 数据源：优先读 `modelscope_sources.json` 中 `datasets.ccmatrix.dataset_id`，默认 `AI-ModelScope/ccmatrix`；`--backend auto` 时 ModelScope 失败会回退到 Hugging Face `sentence-transformers/parallel-sentences-ccmatrix`。
- split：默认 `train`。
- 单对导出：`--src-lang <src> --tgt-lang <tgt> --limit <N>`。
- 批量导出：`--pairs-config training/ccmatrix_pair_limits.json --export-from-config`。
- 当前 `training/ccmatrix_pair_limits.json`：20 个方向，覆盖 `eng_Latn`/`zho_Hans` 与 `spa_Latn`、`ind_Latn`、`vie_Latn`、`tha_Thai`、`tgl_Latn` 互译，每方向默认 `limit=100000`。
- Hugging Face 回退限制：只支持包含英语的 `en-xx` 配置；非英语互译方向需要 ModelScope 源可用或另行准备。
- 输出文件：`training/data/ccmatrix_mt_<src>__<tgt>.jsonl`。
- 预览文件：`training/data/previews/ccmatrix_mt_<src>__<tgt>.preview_50.jsonl`。

每行 JSONL 字段：

```json
{
  "instruction": "Translate the following English text into Spanish. Output only the translation.",
  "input": "...",
  "output": "..."
}
```

### prepare_nllb_for_llamafactory.py

用途：准备 LLaMA-Factory SFT 翻译数据，输出 Alpaca JSONL。该脚本不直接用 `datasets.load_dataset("allenai/nllb")`，而是读取 `allenai/nllb` 的语言对脚本后，从 AllenAI GCS 或 statmt cc-matrix `.gz/.tsv.gz` 流式拉取。

- 数据源：`allenai/nllb` 的 `nllb_lang_pairs.py`；实际文本来自 AllenAI GCS 或 `http://data.statmt.org/cc-matrix/`。
- 单对导出：`--src-lang <src> --tgt-lang <tgt> --limit <N>`。
- 批量导出：`--pairs-config <json> --export-from-config`。
- 默认批量配置：`training/ccmatrix_pair_limits.json`，20 个方向，每方向默认 `limit=100000`。
- MoE 双向 expert 配置：`training/moe_pair_limits.json`，20 个方向，覆盖五个低资源语言分别与中文/英文互译，每方向默认 `limit=100000`。
- 默认输出目录：`training/data/multilingual/nllb/`；MoE 数据建议使用 `--out-subdir multilingual/nllb_moe`。
- 输出文件：`training/data/<out-subdir>/nllb_mt_<src>__<tgt>.jsonl`。
- 预览文件：`training/data/<out-subdir>/previews/nllb_mt_<src>__<tgt>.preview_50.jsonl`。
- instruction 语言：英文模板，与 `scripts/run/run_eval.py` 的评测 prompt 对齐。

MoE 数据有专用入口，避免误把 `evaluation_config.json` 传给 `--pairs-config`：

```bash
conda activate lowres
python scripts/prepare/prepare_nllb_moe_for_llamafactory.py
```

它等价于：

```bash
python scripts/prepare/prepare_nllb_for_llamafactory.py \
  --pairs-config training/moe_pair_limits.json \
  --export-from-config \
  --out-subdir multilingual/nllb_moe
```

每行 JSONL 字段：

```json
{
  "instruction": "Translate the following Spanish text into Simplified Chinese. Output only the translation.",
  "input": "...",
  "output": "..."
}
```

### prepare_nllb_instruction_input_response_jsonl.py

用途：准备与 NLLB 相同来源和语言对的 SFT 数据，但序列化为单列 `text`，适合需要完整 prompt 文本块的训练配置。

- 数据源、语言对解析、`--pairs-config`、`--limit`：与 `prepare_nllb_for_llamafactory.py` 相同。
- 默认输出目录：`training/data/multilingual/nllb_iir/`。
- 输出文件：`training/data/multilingual/nllb_iir/nllb_iir_<src>__<tgt>.jsonl`。
- 预览文件：`training/data/multilingual/nllb_iir/previews/nllb_iir_<src>__<tgt>.preview_50.jsonl`。

每行 JSONL 字段：

```json
{
  "text": "Instruction: Translate the following Spanish text into Simplified Chinese. Output only the translation.\nInput: ...\nResponse: ..."
}
```

### dictionary MoE prepare scripts

用途：准备基于词典的低资源翻译补充数据。第一版覆盖 20 个 MoE 方向：`spa_Latn`、`ind_Latn`、`vie_Latn`、`tha_Thai`、`tgl_Latn` 分别与 `eng_Latn`、`zho_Hans` 双向互译。

- `scripts/dictionary/prepare_muse_dictionary.py`
  - 数据源：facebookresearch MUSE bilingual dictionaries，默认 URL 形如 `https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.txt`。
  - 默认语言：`es/id/vi/th/tl/zh <-> en`，其中 `tl` 映射 `tgl_Latn`。
  - 输出 raw：`training/data/dictionaries/raw/muse/<muse_src>-<muse_tgt>.txt`。
  - 输出 lexicon：`training/data/dictionaries/lexicon/dict_terms_<src>__<tgt>.jsonl`。
- `scripts/dictionary/prepare_cc_cedict_dictionary.py`
  - 数据源：CC-CEDICT via MDBG，默认 URL `https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz`。
  - 用途：补 `zho_Hans <-> eng_Latn`，并过滤明显繁体中文表面词。
  - 输出 lexicon：`dict_terms_zho_Hans__eng_Latn.jsonl` 与 `dict_terms_eng_Latn__zho_Hans.jsonl`。
- `scripts/dictionary/build_pivot_dictionary.py`
  - 用途：用英文精确 pivot 构造低资源语言与简体中文互译词典。
  - 输入目录：`training/data/dictionaries/lexicon/`。
  - 输出目录：`training/data/dictionaries/moe_lexicon/`。
- `scripts/dictionary/build_dictionary_mt_jsonl.py`
  - 用途：把词典 lexicon 转成 LLaMA-Factory Alpaca JSONL。
  - 默认读取 `training/moe_pair_limits.json` 的 20 个方向。
  - 输出目录：`training/data/multilingual/dictionary_moe/`。
  - dataset 注册片段：`training/dictionary_moe_dataset_info.snippet.json`，并默认更新 `training/data/dataset_info.json`。
- 一键入口：`scripts/dictionary/prepare_dictionary_moe_for_llamafactory.py`。
  - 默认先准备 CC-CEDICT，再准备 MUSE；因此自动 pivot 默认使用 MUSE 的 `zh-en` 双语词典，避免把 CC-CEDICT 英文释义倒排成噪声中文翻译。

运行：

```bash
conda activate lowres
python scripts/dictionary/prepare_dictionary_moe_for_llamafactory.py
```

如需先做小样本检查：

```bash
python scripts/dictionary/prepare_dictionary_moe_for_llamafactory.py --limit-per-direction 2000
```

这里的 `--limit-per-direction` 只限制 pivot 后 lexicon 与最终 Alpaca 输出；下载和归一化默认保留完整词典，避免小样本截断导致英文 pivot 没有交集。若确实想限制归一化阶段，可额外传 `--normalize-limit-per-direction`。

词典 lexicon 每行 JSONL 字段：

```json
{
  "source": "muse",
  "src_lang": "spa_Latn",
  "tgt_lang": "eng_Latn",
  "source_text": "...",
  "target_text": "...",
  "target_candidates": ["..."],
  "confidence": 0.82,
  "relation": "translation",
  "source_url": "https://...",
  "license_note": "..."
}
```

最终 Alpaca 数据每行字段：

```json
{
  "instruction": "Translate the following Spanish dictionary term into English. Output only the best translation.",
  "input": "...",
  "output": "..."
}
```

#### dictionary_moe 与句级数据的训练关系

最终指标只看 FLORES 和 COMET 时，`dictionary_moe` 不应替代 NLLB / FineWeb 合成句对。它的定位是低资源方向的词汇锚点数据，用来补术语、常见词和短语覆盖；主优化目标仍应是自然句翻译质量。

方案 A：混合训练，推荐默认方案。

- 每个 pair expert 使用同方向句级数据为主：`training/data/multilingual/nllb_moe/`、`training/data/multilingual/fineweb2_synth/`。
- 当前阶段先混句级数据：所有方向使用 `nllb_moe`；只有配置里显式打开的方向再加入 `fineweb2_synth`。
- 同方向 `dictionary_moe` 先预留但默认不启用；后续如要打开，建议只占 `3% - 10%`，先试 `5%`。
- 示例比例：当前 `NLLB/FineWeb sentence pairs = 100%`，后续词典 ablation 可试 `NLLB/FineWeb = 95%`、`dictionary_moe = 5%`。
- 优点：训练分布仍接近 FLORES/COMET 的句级评测，同时给低资源方向补词汇召回。

方案 B：两阶段训练，但最终阶段仍以句级数据为主。

- Stage 1：`dictionary_moe + 少量句级数据`，短训 `0.1 - 0.3 epoch`，相当于先做词汇热身。
- Stage 2：`句级数据为主 + dictionary_moe 小比例混合`，正式训练 pair expert。
- 不建议 Stage 2 只训词典；否则模型容易偏向短词条翻译，损害自然句输出。

方案 C：词典只做数据增强或质检，不直接训练。

- 用 `dictionary_moe` 检查合成句对里的关键词是否错译。
- 用词典构造 hard examples，或筛掉明显不符合词典约束的伪平行数据。
- 适合后续提高数据质量，但第一版不如方案 A 直接。

当前建议：先采用方案 A 的句级版本。也就是继续训练 pair-level LoRA MoE，每个 expert 至少使用同方向 `nllb_moe`；对已有 FineWeb synthetic 的方向额外加入 `fineweb2_synth`；`dictionary_moe` 只在配置里预留，最终仍只用 FLORES 和 COMET 评估。等 baseline 有结果后，再比较 `0% / 5% / 10% dictionary_moe` 三个 ablation。

#### mixed_moe 句级混合数据

`scripts/moe/build_mixed_moe_for_llamafactory.py` 用来把每个方向的 NLLB 与 FineWeb 合成数据合成一个训练文件。默认配置文件：

```text
training/moe_data_mix_config.json
```

默认数据源：

- `nllb`：启用，每方向最多 `100000` 条，读取 `training/data/multilingual/nllb_moe/nllb_mt_<src>__<tgt>.jsonl`。
- `fineweb_synth`：启用但不要求存在，每方向最多 `100000` 条，读取 `training/data/multilingual/fineweb2_synth/fineweb_synth_<src>__<tgt>.jsonl`。某个方向没有该文件时记为 `0` 条并跳过；有文件时自动混入。
- `dictionary`：预留，默认 `enabled=false` 且 `limit=0`。

`enabled=true` 表示会检查这个来源；`required=true` 才会在 `--strict` 下因为缺文件而失败。当前 `nllb.required=true`，`fineweb_synth.required=false`，所以不同语言对可以来自不同数据源。

生成 mixed 数据：

```bash
python scripts/moe/build_mixed_moe_for_llamafactory.py
```

输出：

```text
training/data/multilingual/mixed_moe/mixed_moe_<src>__<tgt>.jsonl
training/data/multilingual/mixed_moe/previews/mixed_moe_<src>__<tgt>.preview_50.jsonl
training/mixed_moe_dataset_info.snippet.json
```

让 pair expert 训练配置使用 mixed 数据：

```bash
python scripts/moe/generate_pair_expert_assets.py \
  --training-dataset-prefix mixed_moe \
  --training-data-subdir multilingual/mixed_moe \
  --training-file-prefix mixed_moe
```

#### single_sft 非 MoE 对照

为了和 pair-level LoRA MoE 对比，可以把所有方向的数据合并成一个普通 SFT 数据集，训练单个 LoRA adapter：

```bash
python scripts/moe/build_single_sft_for_llamafactory.py --strict
```

该脚本复用 `training/moe_data_mix_config.json`，所以每个方向、每个来源的数据量与 `mixed_moe` 保持一致。默认输出：

```text
training/data/multilingual/single_sft/qwen3_8b_all_directions_sft.jsonl
training/data/multilingual/single_sft/previews/qwen3_8b_all_directions_sft.preview_50.jsonl
training/qwen3_8b_all_directions_sft_dataset_info.snippet.json
```

训练单 LoRA：

```bash
bash scripts/run/run_train_single_sft.sh
```

等价于：

```bash
llamafactory-cli train /root/lowres_new/training/llamafactory_qwen3_8b_all_directions_sft_lora.yaml
```

部署这个普通 SFT adapter：

```bash
bash scripts/serve/serve_vllm_qwen3_8b_single_sft_lora.sh
```

然后把 `SERVED_MODEL_NAME` 设为 adapter 名跑同一套 FLORES/COMET：

```bash
export SERVED_MODEL_NAME=qwen3_8b_all_directions_sft
export EVAL_MODEL_TAG=qwen3_8b_single_sft
export EVAL_MODEL_FAMILY=qwen3
bash scripts/run/run_eval_baseline.sh
```

#### LoRA MoE 部署与调用

训练完成后，每个方向会得到一个 LoRA adapter。推荐用 vLLM 同时加载 Qwen3-8B base 与全部 pair expert adapter：

```bash
conda activate lowres-serve
bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
```

该脚本读取 `training/moe_router_manifest.json`，把每个 `adapter_name=adapter_path` 注册成 vLLM 可调用的 model name。若 GPU 机器上的路径不同：

```bash
MODEL_PATH=/path/to/Qwen3-8B \
MANIFEST=/path/to/moe_router_manifest.json \
bash scripts/serve/serve_vllm_qwen3_8b_moe_lora.sh
```

单条翻译用 router 脚本：

```bash
python scripts/moe/translate_with_moe_router.py \
  --src-lang tha_Thai \
  --tgt-lang zho_Hans \
  --text "สวัสดีครับ" \
  --print-model
```

查看支持的方向：

```bash
python scripts/moe/translate_with_moe_router.py --list-pairs
```

### prepare_fineweb2_monolingual_for_llamafactory.py

用途：准备单语继续预训练数据，输出 LLaMA-Factory 可用的 `{"text": "..."}` JSONL，并生成 `dataset.info` / `dataset_info.json`。

- 数据源：Hugging Face `HuggingFaceFW/fineweb-2`。
- 默认语言：不传 `--lang` 时，从 `evaluation_config.json` 的 `language_pair_groups` 自动抽取全部语言。
- 语言别名：默认将 `zho_Hans` 映射到 FineWeb-2 的 `cmn_Hani`；可用 `--lang-alias SRC=DST` 增补。
- split：默认 `train`。
- limit：默认每语言 `200000` 条；`--limit 0` 表示不限制。
- 文本字段：默认尝试 `text`、`content`、`raw_content`，否则取该行最长字符串字段。
- 最小长度：默认 `--min-chars 20`。
- 默认输出目录：`training/data/monolingual/`。
- 输出文件：`training/data/monolingual/fineweb2_pt_<lang>.jsonl`。
- 预览文件：`training/data/monolingual/previews/fineweb2_pt_<lang>.preview_50.jsonl`。
- dataset 注册：`training/data/monolingual/dataset.info` 和 `dataset_info.json`，数据集名形如 `fineweb2_pt_<lang>`。

每行 JSONL 字段：

```json
{
  "text": "..."
}
```

### prepare_fineweb_english_for_llamafactory.py

用途：从 Hugging Face `HuggingFaceFW/fineweb` 准备英语单语数据，补齐英泰合成数据需要的 English source。

- 数据源：Hugging Face `HuggingFaceFW/fineweb`。
- 默认配置：`sample-10BT`，避免误拉全量 FineWeb。
- split：默认 `train`。
- limit：默认 `250000` 条；`--limit 0` 表示不限制。
- 文本字段：默认尝试 `text`、`content`、`raw_content`，否则取行内字符串字段。
- 过滤：默认 `--min-chars 20`、`--min-latin-ratio 0.55`。
- 默认输出目录：`training/data/monolingual/`。
- 默认输出文件：`training/data/monolingual/fineweb2_pt_eng_Latn.jsonl`。文件名保持这个形式是为了兼容英泰合成脚本的默认输入路径。
- 预览文件：`training/data/monolingual/previews/fineweb2_pt_eng_Latn.preview_50.jsonl`。
- dataset 注册：`training/data/monolingual/dataset.info` 和 `dataset_info.json`，数据集名默认 `fineweb2_pt_eng_Latn`。

运行：

```bash
conda activate lowres
python scripts/prepare/prepare_fineweb_english_for_llamafactory.py --limit 250000
```

每行 JSONL 字段：

```json
{
  "text": "..."
}
```

### prepare_wikipedia_english_for_llamafactory.py

用途：从 Hugging Face `wikimedia/wikipedia` 准备高质量英语单语数据。FineWeb 访问失败时优先用这个脚本补齐英泰合成数据的 English source。

- 数据源：Hugging Face `wikimedia/wikipedia`。
- 默认配置：`20231101.en`。
- split：默认 `train`。
- limit：默认 `250000` 条；`--limit 0` 表示不限制。
- 文本字段：默认 `text`。
- 过滤：默认 `--min-chars 80`、`--min-latin-ratio 0.55`。
- 长文截断：默认 `--max-chars 1200`，避免单条 article 太长；`--max-chars 0` 表示不截断。
- 默认输出文件：`training/data/monolingual/fineweb2_pt_eng_Latn.jsonl`，保持兼容英泰合成脚本。
- 预览文件：`training/data/monolingual/previews/fineweb2_pt_eng_Latn.preview_50.jsonl`。

运行：

```bash
conda activate lowres
python scripts/prepare/prepare_wikipedia_english_for_llamafactory.py --limit 250000
```

如果 config 发现失败，可强制读 parquet：

```bash
python scripts/prepare/prepare_wikipedia_english_for_llamafactory.py \
  --loader parquet \
  --data-files '20231101.en/train-*.parquet' \
  --limit 250000
```

### prepare_oscar_pretrain_for_llamafactory.py

用途：准备 OSCAR-2301 单语继续预训练数据，输出 `{"text": "..."}` JSONL。

- 数据源：Hugging Face `oscar-corpus/OSCAR-2301`，该数据集是 gated dataset，需要先在 Hub 页面同意条款并登录或设置 `HF_TOKEN`。
- 默认语言：`spa_Latn`、`ind_Latn`、`vie_Latn`、`tha_Thai`、`tgl_Latn`。
- 语言映射：`spa_Latn -> es`，`ind_Latn -> id`，`vie_Latn -> vi`，`tha_Thai -> th`，`tgl_Latn -> tl`。
- split：默认 `train`。
- limit：默认每语言 `50000` 条；`--limit 0` 表示不限制。
- 文本字段：默认尝试 `content`、`text`、`oscar`。
- 最小长度：默认 `--min-chars 1`。
- 默认输出目录：`training/data/`。
- 输出文件：`training/data/oscar_pt_<lang>.jsonl`。
- 预览文件：`training/data/previews/oscar_pt_<lang>.preview_50.jsonl`。

每行 JSONL 字段：

```json
{
  "text": "..."
}
```

### prepare_smollm_fineweb_edu_dedup.py

用途：检查/统计 SmolLM 语料中的 FineWeb-Edu 去重子集，不导出全量训练 JSONL。

- 数据源：Hugging Face `HuggingFaceTB/smollm-corpus`。
- subset：`fineweb-edu-dedup`。
- split：`train`。
- 数据形态：Hub 上是 parquet 分片，全量约 1.9 亿行；脚本按分片下载并统计 `metadata.language`。
- 可选限量：`--max-shards N` 只处理前 N 个 parquet 分片，统计结果会标记 `partial=true`。
- 缓存目录：默认设置 `HF_HOME=datasets/cache/huggingface`。
- 输出样本：`datasets/processed/smollm_corpus_fineweb-edu-dedup_train.sample_50.jsonl`。
- 输出预览：`datasets/previews/smollm_corpus_fineweb-edu-dedup_train.sample_50.jsonl.preview_50.jsonl`。
- 输出统计：`datasets/processed/fineweb-edu-dedup_train_language_counts*.json`。

样本 JSONL 字段：

```json
{
  "dataset": "smollm_corpus",
  "subset": "fineweb-edu-dedup",
  "split": "train",
  "sample_id": "...",
  "text": "...",
  "metadata": {}
}
```

## COMET 与网络

`scripts/run/run_eval.py` 会在开始 vLLM 翻译前先解析或下载 COMET 模型，避免生成完全部 hypothesis 后才发现 COMET 权重不可用。首次运行 COMET 需可访问外网；BLEU 仅本地计算。

默认 `--comet-model models/Unbabel_wmt22-comet-da` 会优先查找本地 ckpt；若不存在，会按远程模型 `Unbabel/wmt22-comet-da` 预下载到该目录，后续评测复用本地文件。也可以在评测前显式下载：

```bash
python scripts/download_models_to_models_dir.py --only comet_wmt22_da
```

FLORES/spBLEU 第一次运行可能会触发 sacrebleu 下载 `flores200_sacrebleu_tokenizer_spm.model`。如果离线机器证书或 tinyurl 访问有问题，建议在能联网的机器上准备评测资产：

```bash
python scripts/download_eval_assets.py --bundle eval_assets.tar.gz
```

把 `eval_assets.tar.gz` 拷到离线机器仓库根目录后解压：

```bash
tar -xzf eval_assets.tar.gz
```

解压后应有：

```text
models/Unbabel_wmt22-comet-da/
models/xlm-roberta-large/
models/sacrebleu/flores200_sacrebleu_tokenizer_spm.model
```

`scripts/run/run_eval.py` 会自动优先使用这个本地 SPM 文件，也会在本地 `models/xlm-roberta-large/` 存在时把 COMET 的 `pretrained_model` 指向该目录，避免加载 COMET 时再访问 Hugging Face。也可显式指定：

```bash
python scripts/run/run_eval.py \
  --flores200-spm-model models/sacrebleu/flores200_sacrebleu_tokenizer_spm.model \
  --comet-encoder-model models/xlm-roberta-large
```

如果看到 Lightning 自动升级 checkpoint 的提示，它通常只是 PyTorch Lightning 版本迁移提示，不是网络报错原因；真正需要离线准备的是 COMET 权重、`xlm-roberta-large` encoder 和 FLORES200 SPM。

只跑 BLEU 时可显式禁用：

```bash
python scripts/run/run_eval.py --comet-model none
```

## NLLB-200 3.3B 多语基线

`facebook/nllb-200-3.3B` 是 Meta 发布的 NLLB-200 多语神经机器翻译模型，Hugging Face 页面：
`https://huggingface.co/facebook/nllb-200-3.3B`。它覆盖 NLLB/FLORES 语言代码体系，适合当前
`evaluation_config.json` 中的 `eng_Latn`、`zho_Hans`、`spa_Latn`、`ind_Latn`、`vie_Latn`、
`tha_Thai`、`tgl_Latn` 多方向评估。

它不是传统规则/统计机器翻译模型，而是 encoder-decoder 神经 MT 模型；这里把它作为“覆盖更多语言对”的强多语基线。运行不需要 vLLM，但需要 `transformers`、`huggingface_hub`、`sentencepiece`、`torch`。

下载到项目 `models/`：

```bash
conda activate lowres
python scripts/download_hf_nllb_200_3_3b.py
```

默认目标目录：

```text
models/facebook_nllb-200-3.3B
```

如只想检查路径，不实际下载：

```bash
python scripts/download_hf_nllb_200_3_3b.py --dry-run
```

在现有 FLORES/NTREX manifest 上评估，并输出与 `scripts/run/run_eval.py` 同格式的
`hypotheses.jsonl`、`metrics.json`、`metrics.csv`：

```bash
conda activate lowres
python scripts/run/run_eval_nllb.py
```

只跑 BLEU，跳过 COMET：

```bash
python scripts/run/run_eval_nllb.py --comet-model none
```

常用资源参数：

```bash
python scripts/run/run_eval_nllb.py \
  --batch-size 8 \
  --num-beams 4 \
  --device cuda \
  --dtype fp16
```

如果没有先下载，也可以直接从 Hugging Face repo id 加载；Transformers 会使用本地 HF cache：

```bash
python scripts/run/run_eval_nllb.py \
  --model-name-or-path facebook/nllb-200-3.3B
```

评估前仍需先生成项目的评估数据：

```bash
python scripts/prepare/prepare_datasets.py
python scripts/expand_language_pairs.py
```

## NLLB-MoE 54B 多语基线

`facebook/nllb-moe-54b` 是 NLLB 的 Mixture-of-Experts 版本，Hugging Face 页面：
`https://huggingface.co/facebook/nllb-moe-54b`。模型卡标注为 FLORES-200 / 196 languages，
生成时同样需要把 `forced_bos_token_id` 设成目标语言 id；`scripts/run/run_eval_nllb.py`
已经按每条样本的 `tgt_lang` 自动设置。

注意资源：模型卡说明 checkpoint 约需 350GB 存储，内存不足时需要使用 `accelerate`。建议只在有足够磁盘、内存和 GPU/CPU offload 资源的机器上运行。

下载到项目 `models/`：

```bash
conda activate lowres
python scripts/download_hf_nllb_moe_54b.py
```

默认目标目录：

```text
models/facebook_nllb-moe-54b
```

如只想检查路径，不实际下载：

```bash
python scripts/download_hf_nllb_moe_54b.py --dry-run
```

评估 NLLB-MoE 54B 使用同一个 NLLB 评估脚本。推荐从小 batch 和 BLEU-only 开始：

```bash
python scripts/run/run_eval_nllb.py \
  --model-name-or-path models/facebook_nllb-moe-54b \
  --model-tag nllb_moe_54b \
  --batch-size 1 \
  --num-beams 4 \
  --dtype fp16 \
  --device-map auto \
  --comet-model none
```

如要启用 COMET，去掉 `--comet-model none`；输出格式仍与 `scripts/run/run_eval.py` 一致。
`--device-map auto` 会启用 Transformers/Accelerate 的自动切分；磁盘 offload 默认写到：

```text
models/offload/nllb_moe_54b
```

## 传统机器翻译基线：Apertium English-Spanish

当前项目内已放入一个可离线运行的传统机器翻译模型：

- 模型文件：`models/apertium-en-es/apertium-en-es.jar`
- 下载来源：`https://sourceforge.net/p/apertium/svn/HEAD/tree/builds/apertium-en-es/apertium-en-es.jar?format=raw`
- 方法说明：`https://wiki.apertium.org/wiki/Language_pair_packages`
- 支持方向：`eng_Latn -> spa_Latn` 与 `spa_Latn -> eng_Latn`
- 运行依赖：Java；不需要 vLLM、GPU 或神经网络推理服务

该模型来自 Apertium。Apertium 是开源的 shallow-transfer rule-based machine translation
系统，核心方法不是大模型，也不是 Transformer/NMT，而是传统 RBMT：

1. 用有限状态词典和形态分析器做词形分析。
2. 用词性标注/消歧模块选择上下文中的词法分析结果。
3. 用双语词典做词汇迁移。
4. 用结构迁移规则处理短语、局部语序和形态特征。
5. 用形态生成器输出目标语言表层文本。

因此它适合作为“传统方法”基线，而不是现代神经翻译或 LLM 的总体 SOTA。它的优势是离线、小、可解释、可复现；局限是覆盖语向少，长句和开放领域质量通常弱于现代神经模型。

单句测试：

```bash
printf 'I like apples.\n' | java -jar models/apertium-en-es/apertium-en-es.jar apertium en-es
printf 'Me gustan las manzanas.\n' | java -jar models/apertium-en-es/apertium-en-es.jar apertium es-en
```

使用与现有 vLLM 评估脚本相同的 BLEU/COMET 方法评估 Apertium：

```bash
conda activate lowres
python scripts/run/run_eval_apertium.py
```

`scripts/run/run_eval_apertium.py` 会读取 `datasets/eval_manifest.json`，只保留 Apertium 支持的
`eng_Latn <-> spa_Latn` 样本，其余语向会跳过。输出目录仍在 `evaluation_config.json` 的
`output_dir` 下，文件格式与 `scripts/run/run_eval.py` 保持一致：

- `hypotheses.jsonl`
- `metrics.json`
- `metrics.csv`
- `metrics_flores.csv` / `metrics_ntrex.csv`（取决于 manifest 中实际有的语料）

如只想先跑 BLEU，跳过 COMET：

```bash
python scripts/run/run_eval_apertium.py --comet-model none
```

## HY-MT1.5 与 vLLM

若 `serve_vllm_hunyuan_mt.sh` 启动失败，可能是架构与 vLLM 不兼容；需改用 Transformers 批推理并仍通过 ModelScope 本地路径加载（可自行接同一套 `hypotheses.jsonl` 流程）。

## 训练

见 [docs/training_4x5090.md](docs/training_4x5090.md) 与 `training/llamafactory_ccmatrix_sft.placeholder.yaml`。
