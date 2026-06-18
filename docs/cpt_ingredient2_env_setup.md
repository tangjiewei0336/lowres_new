# ingredient2 继续预训练环境配置

本文档对应脚本：

```bash
scripts/run/run_cpt_ingredient2_4x5090.sh
scripts/train/continue_pretrain_ds_indexed.py
training/deepspeed_zero2_cpt_4x5090.json
```

目标：在 4 卡 RTX 5090 上，从 300000 步 HF checkpoint 继续预训练，并读取 `ingredient2` 下的 DeepSpeed/Megatron indexed dataset：

```bash
/share/aidev/data/slm_dataset_tokenized_docshuffled_qwen3/dolma3_dolmino_mix-100B-1125-tokQwen3/ingredient2
```

## 1. 激活环境

如果集群上已有环境，先激活它：

```bash
conda activate YOUR_ENV_NAME
cd /share/airesearch/slm/lowres_new
```

如果是在仓库路径不同的位置，进入实际的 `lowres_new` 目录即可。

## 2. 确认基础依赖

DeepSpeed 已经有的话，先确认 PyTorch、Transformers、Accelerate：

```bash
python - <<'PY'
import torch
import transformers
import accelerate
import deepspeed

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("deepspeed:", deepspeed.__version__)
print("gpu count:", torch.cuda.device_count())
PY
```

建议版本：

- `transformers >= 4.45`
- `accelerate >= 0.34`
- `torch` 使用集群 CUDA 对应版本
- `deepspeed` 已安装且能正常 import

缺少 Transformers/Accelerate 时：

```bash
pip install "transformers>=4.45" "accelerate>=0.34" sentencepiece numpy
```

## 3. 安装或挂载 Megatron indexed dataset reader

这个继续预训练脚本直接读取 `.ds/.ds.index` 二进制数据。运行时需要能 import 下面任意一个模块：

```python
megatron.core.datasets.indexed_dataset
megatron.data.indexed_dataset
deepspeed.runtime.data_pipeline.data_routing.indexed_dataset
```

先检查：

```bash
python - <<'PY'
import importlib

mods = [
    "megatron.core.datasets.indexed_dataset",
    "megatron.data.indexed_dataset",
    "deepspeed.runtime.data_pipeline.data_routing.indexed_dataset",
]

ok = False
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("OK:", m, "->", mod.__file__)
        ok = True
    except Exception as e:
        print("MISS:", m, "->", e)

raise SystemExit(0 if ok else 1)
PY
```

如果至少有一个 `OK`，可以直接训练。

如果全部 `MISS`，需要安装或挂载 Megatron-LM/Megatron-Core。常见做法是把集群已有的 Megatron 代码加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
```

然后重新执行上面的检查。

如果需要临时安装 Megatron-Core，可按集群可用源安装：

```bash
pip install megatron-core
```

注意：不同集群镜像里包名和 CUDA/PyTorch 编译环境可能不同。如果 `pip install megatron-core` 不可用，优先使用集群已有的 Megatron-LM 源码路径加 `PYTHONPATH`。

## 4. 确认数据与模型路径

```bash
ls -lh /share/airesearch/slm/checkpoints/slm_msxf_1b/checkpoints/ckp-2026-06-11-fullset-v1.5-13lang-nosynth-gbs9M-cpt-hf/300000

find /share/aidev/data/slm_dataset_tokenized_docshuffled_qwen3/dolma3_dolmino_mix-100B-1125-tokQwen3/ingredient2 \
  -name 'train.c*.ds' | head
```

每个 `.ds` 旁边应有对应 `.ds.index`，例如：

```bash
train.c0.ds
train.c0.ds.index
```

脚本会在输出目录下自动创建只读符号链接，把 `.ds/.ds.index` 映射为 Megatron 常见的 `.bin/.idx`，不会修改原始数据。

## 5. 先做一次小步冒烟测试

建议先跑 5 步，确认数据读取、模型加载、DeepSpeed 初始化都正常：

```bash
MAX_STEPS=5 SAVE_STEPS=5 LOGGING_STEPS=1 \
bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

如果这里正常，再跑正式训练。

## 6. 正式训练

默认 4 卡：

```bash
bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

默认输出：

```bash
models/slm_msxf_1b_ingredient2_cpt_4x5090
```

常用覆盖参数：

```bash
SEQ_LEN=4096 \
MAX_STEPS=30000 \
LEARNING_RATE=1e-5 \
GRADIENT_ACCUMULATION_STEPS=8 \
bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

如果 4096 OOM：

```bash
SEQ_LEN=2048 GRADIENT_ACCUMULATION_STEPS=16 \
bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

## 7. 常见问题

### 找不到 indexed dataset reader

报错类似：

```text
Cannot open DeepSpeed/Megatron indexed dataset
```

说明环境里没有 Megatron indexed dataset reader。解决方式：

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
```

或安装 `megatron-core`。

### 找不到 `.idx`

原始数据是 `.ds.index`，脚本会自动创建 `.idx` 符号链接。若输出目录无写权限，改用可写输出目录：

```bash
OUTPUT_DIR=/share/airesearch/slm/models/slm_msxf_1b_ingredient2_cpt_4x5090 \
bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

### 4 卡通信异常

可尝试：

```bash
NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

脚本里默认已经设置了这两个环境变量。

### 显存不足

优先降低序列长度：

```bash
SEQ_LEN=2048 bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

如果还不够：

```bash
SEQ_LEN=1024 GRADIENT_ACCUMULATION_STEPS=32 bash scripts/run/run_cpt_ingredient2_4x5090.sh
```

