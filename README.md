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
  "instruction": "请将以下 英语 文本翻译为 西班牙语，只输出译文。",
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

每行 JSONL 字段与 ccMatrix 相同：

```json
{
  "instruction": "请将以下 西班牙语 文本翻译为 简体中文，只输出译文。",
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
  "text": "Instruction: Translate Spanish to Chinese (Simplified)\nInput: ...\nResponse: ..."
}
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

首次运行 COMET 会下载评测模型权重，需可访问外网；BLEU 仅本地计算。

## HY-MT1.5 与 vLLM

若 `serve_vllm_hunyuan_mt.sh` 启动失败，可能是架构与 vLLM 不兼容；需改用 Transformers 批推理并仍通过 ModelScope 本地路径加载（可自行接同一套 `hypotheses.jsonl` 流程）。

## 训练

见 [docs/training_4x5090.md](docs/training_4x5090.md) 与 `training/llamafactory_ccmatrix_sft.placeholder.yaml`。
