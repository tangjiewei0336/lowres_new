# 4× RTX 5090 + LLaMA Factory 训练建议（占位）

以下仅为起点配置，**务必以实际显存占用与 loss 曲线微调**。模型与训练数据请**仅通过 ModelScope** 获取后，再在 LLaMA Factory 中指向本地路径或已同步的数据集 ID。

## 环境与 venv

在仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 另装 LLaMA Factory 与其依赖（按官方文档）
```

## 数据格式（ccMatrix / NLLB）

格式化nllb

python scripts/prepare_nllb_for_llamafactory.py --export-from-config --pairs-config training/ccmatrix_pair_limits.json

- LLaMA Factory 支持 Alpaca / ShareGPT 等格式，需在 `data/dataset_info.json` 中注册。
- **推荐 Alpaca 翻译样本**：
  - `instruction`: 固定如「请将以下 {src} 文本翻译为 {tgt}，只输出译文。」
  - `input`: 源句
  - `output`: 目标句
- 若数据集仅在 ModelScope 上：先用 `snapshot_download` 或 `MsDataset` 导出为 `jsonl`，再在 `dataset_info.json` 里用本地 `file_name` 指向该文件（**不要默认走 HuggingFace Hub**）。

## 4× 5090（约 32GB×4）起步超参

| 项 | 建议 |
|----|------|
| 精度 | `bf16`（或 Factory 支持下的 FP8，需硬件/框架支持） |
| 并行 | DeepSpeed ZeRO-2 或 FSDP；4 卡 DDP |
| cutoff_len | 256–512 起试（翻译句对通常较短；过长浪费显存） |
| per_device_train_batch_size | 1–4 起试，OOM 则减 |
| gradient_accumulation_steps | 使 **有效全局 batch** 达到 64–256（按数据量调） |
| learning_rate | `1e-5` ~ `3e-5`（LoRA/SFT 常见范围） |
| warmup_ratio | 0.03–0.1 |
| logging_steps | 10–50；save_steps 按磁盘与验证需求 |

## 输出目录命名

与评估约定一致：`models/{基座简写}_{YYYYMMDD_HHMMSS}/`，例如 `models/Qwen3-4B_20250407_153012/`。

## 参考

- [LLaMA Factory Data Preparation](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html)
