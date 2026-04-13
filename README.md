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

## COMET 与网络

首次运行 COMET 会下载评测模型权重，需可访问外网；BLEU 仅本地计算。

## HY-MT1.5 与 vLLM

若 `serve_vllm_hunyuan_mt.sh` 启动失败，可能是架构与 vLLM 不兼容；需改用 Transformers 批推理并仍通过 ModelScope 本地路径加载（可自行接同一套 `hypotheses.jsonl` 流程）。

## 训练

见 [docs/training_4x5090.md](docs/training_4x5090.md) 与 `training/llamafactory_ccmatrix_sft.placeholder.yaml`。
