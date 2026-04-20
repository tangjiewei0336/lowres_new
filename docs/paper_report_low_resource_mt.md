# 大模型在低资源语言翻译中的能力边界与改进路径

## 0. 报告定位

本报告围绕低资源语言机器翻译展开，核心问题是：

> 现代大模型已经具备很强的通用语言能力，但这种能力在低资源语言翻译上是否稳定？如果不稳定，问题主要来自数据覆盖不足，还是学习机制与训练目标不匹配？

报告分为两部分：

1. **经验分析大模型在低资源语言上的翻译能力**
2. **Dual Perspective: Data & Learning**

建议将报告主线组织为：

> 低资源翻译不是单纯的“模型不够大”问题，而是数据分布、语言覆盖、对齐方式、生成学习目标和推理约束共同作用的结果。经验评测先揭示现象，data perspective 解释“模型看到了什么”，learning perspective 解释“模型如何学、如何用”。

---

## 1. 研究背景与动机

### 1.1 为什么关注低资源语言翻译

高资源语言，如英语、中文、西班牙语，在网络语料、平行语料、指令数据和评测基准中覆盖充分。相比之下，低资源语言常见问题包括：

- 单语语料少，质量不稳定。
- 平行语料稀缺，领域偏窄。
- 语言标准化程度不足，存在多方言、多拼写、多脚本。
- tokenizer 覆盖差，常被切成过碎的子词甚至字节级片段。
- 评测集规模小，自动指标波动较大。

因此，大模型在低资源语言上的翻译能力不能简单从英文或中英翻译能力外推。

### 1.2 核心研究问题

可以把问题拆成三个层次：

1. **能力是否存在**

   大模型是否能完成低资源语言与高资源语言之间的基本翻译？

2. **能力是否稳定**

   不同语言方向、不同源语言、不同目标语言之间是否存在明显不对称？

3. **能力如何改进**

   应该优先补数据、调训练方式、改 tokenizer、加检索，还是设计专家模型/路由机制？

---

## 2. Part I: 经验分析大模型在低资源语言上的翻译能力

### 2.1 实验对象

本项目可以把模型分成四类基线：

1. **通用大模型**

   例如 Qwen 系列、SmolLM、Hunyuan-MT 等，通过 prompt 方式执行翻译。

2. **多语神经翻译模型**

   例如 NLLB-200 3.3B、NLLB-MoE 54B。它们不是 LLM，而是专门面向多语翻译的 encoder-decoder MT 模型。

3. **传统机器翻译基线**

   例如 Apertium English-Spanish。它是规则机器翻译系统，可解释性强，但覆盖语向有限。

4. **项目训练模型**

   例如基于 NLLB/ccMatrix/FineWeb/dictionary 数据训练的 LoRA、single SFT、pair-level MoE expert。

### 2.2 评测语言与方向

当前实验配置覆盖：

- `eng_Latn`
- `zho_Hans`
- `spa_Latn`
- `ind_Latn`
- `vie_Latn`
- `tha_Thai`
- `tgl_Latn`

评测重点不是所有语言两两组合，而是围绕英语/中文与低资源语言的跨组翻译方向。

可以重点观察：

- `eng_Latn -> low-resource`
- `low-resource -> eng_Latn`
- `zho_Hans -> low-resource`
- `low-resource -> zho_Hans`
- `eng_Latn <-> zho_Hans` 作为高资源参考方向

### 2.3 数据集与指标

评测数据：

- **FLORES-200**：多语标准评测集，覆盖 NLLB/FLORES 语言代码体系。
- **NTREX**：英语中心多语评测补充。

评测指标：

- **BLEU / spBLEU**：衡量 n-gram 匹配，适合观察表层翻译质量。
- **COMET**：基于源句、译文、参考译文的 learned metric，更接近语义质量。

报告中建议强调：

> BLEU 与 COMET 衡量角度不同。低资源语言中，BLEU 容易受分词、脚本、形态变化影响；COMET 更偏语义，但它本身也可能存在语言覆盖偏差。

### 2.4 经验现象一：方向不对称

一个常见现象是：

> 低资源语言翻译到英语，往往比英语翻译到低资源语言更容易。

可能原因：

- 目标语言为英语时，模型生成空间更熟悉。
- 英语目标端语言模型能力强，输出更流畅。
- 低资源目标语言生成需要掌握形态、拼写、语序和脚本，错误空间更大。
- 训练数据中 “X -> English” 的监督可能比 “English -> X” 更稳定。

报告可以用热力图呈现：

- 行：源语言
- 列：目标语言
- 单元格：BLEU 或 COMET

重点观察矩阵是否对称。

### 2.5 经验现象二：中文作为 pivot 的不稳定性

当前项目关心中文与低资源语言互译。通常会看到：

> 英语与低资源语言之间的表现好于中文与低资源语言之间的表现。

可能原因：

- 许多低资源平行语料以英语为中心。
- NLLB、ccMatrix 等多语资源中，非英语方向常通过英语间接连接。
- 中文与部分低资源语言的直接平行数据更少。
- 中文和低资源语言在语序、分词、形态、脚本上差异大。

可分析的问题：

- `low-resource -> zho_Hans` 是否比 `zho_Hans -> low-resource` 更好？
- 中文方向错误是语义错、漏译，还是目标语不自然？
- 模型是否倾向先生成英文中间结构，再转成中文？

### 2.6 经验现象三：语言间差异显著

即使都被称为低资源语言，不同语言的难度也不同。

可能观察：

- 西班牙语由于资源丰富，通常表现最好。
- 印尼语、越南语可能处于中间。
- 泰语因无空格分词、脚本特殊，BLEU 更敏感。
- 他加禄语可能受数据覆盖和方言/拼写影响。

报告中应避免把所有低资源语言混为一类。更合理的说法是：

> Low-resource is not a single condition, but a spectrum of resource availability, script complexity, typological distance, and benchmark reliability.

### 2.7 经验现象四：大模型翻译的典型错误

可以把错误分成几类：

1. **漏译**

   长句、从句、专有名词或数字被省略。

2. **过度意译**

   语义大致相关，但不忠实于源句。

3. **语言混杂**

   输出夹杂英语、中文或错误目标语言。

4. **专名错误**

   地名、人名、组织名被音译错或翻译成普通词。

5. **形态和词序错误**

   目标语言语法不自然，尤其在形态丰富语言中明显。

6. **脚本错误**

   输出错误文字系统，或混用拉丁化拼写与本土脚本。

### 2.8 经验结论

可以总结为：

> 大模型在低资源翻译上具备一定跨语言迁移能力，但这种能力不稳定。模型规模提升能缓解部分问题，但不能消除语言覆盖、目标端生成能力、平行数据稀缺和训练目标不匹配带来的系统性差异。

---

## 3. Part II: Dual Perspective: Data & Learning

第二部分可以从两个视角解释第一部分的经验现象。

## 3.1 Data Perspective

### 3.1.1 数据覆盖决定能力上限

低资源语言翻译的关键瓶颈首先是数据。

数据可以分成：

- 单语数据
- 平行数据
- 字典/术语数据
- 合成翻译数据
- 指令翻译数据
- 领域数据

不同数据影响不同能力：

| 数据类型 | 主要贡献 | 局限 |
|---|---|---|
| 单语数据 | 目标语言流畅度、词汇覆盖 | 不直接提供翻译对齐 |
| 平行数据 | 翻译映射、跨语言对齐 | 低资源语向稀缺 |
| 字典数据 | 术语、实体、常见词锚点 | 不提供完整句法和语境 |
| 合成数据 | 扩大覆盖、补齐方向 | 可能继承教师模型偏差 |
| 指令数据 | 让 LLM 学会按 prompt 翻译 | 可能牺牲严格忠实性 |

### 3.1.2 数据质量比数据规模更关键

低资源场景中，简单增加数据并不一定有效。

常见噪声：

- 源文和译文不对齐。
- 机器翻译伪数据质量低。
- 语言识别错误。
- 目标语言混入英语。
- 重复数据过多。
- 领域过窄。

因此报告可以提出：

> Low-resource MT benefits more from targeted, direction-aware, high-quality data than from indiscriminate scaling.

### 3.1.3 英语中心数据带来的结构性偏差

很多多语数据集是英语中心的：

```text
low-resource <-> English
Chinese <-> English
```

但项目真正关心的可能是：

```text
low-resource <-> Chinese
```

这会导致一个问题：

> 模型学到的是以英语为 hub 的跨语言映射，而不是任意语言对之间的直接翻译。

解决思路：

- 构造中文与低资源语言的直接平行数据。
- 用英语作为 pivot 生成中文方向伪平行数据。
- 使用 multilingual teacher 做数据增强。
- 对 `zho_Hans <-> low-resource` 方向单独训练 expert。

### 3.1.4 字典数据的作用

词典数据不能替代句级平行语料，但可以作为低资源方向的词汇锚点。

适合补：

- 高频词
- 专有名词
- 地名
- 术语
- 形态变化较小的短语

不适合单独优化：

- 长句翻译
- 上下文消歧
- 自然语序
- 篇章一致性

因此它更适合作为混合训练中的小比例补充，例如 `5%` 或 `10%` 的词典样本。

### 3.1.5 数据视角总结

从数据角度看，低资源翻译的改进路径是：

1. 补齐目标语向的直接平行数据。
2. 过滤噪声和语言识别错误。
3. 增强中文-低资源方向，而不是只依赖英语 pivot。
4. 结合句级数据和词典锚点数据。
5. 针对每个语言对设计不同数据配比。

---

## 3.2 Learning Perspective

### 3.2.1 大模型翻译不是标准 MT 训练目标

通用 LLM 的主要目标通常是 next-token prediction 或 instruction following。

翻译任务要求的是：

- 忠实性
- 完整性
- 目标语言自然度
- 格式约束
- 不解释、不扩写、不省略

这和开放式生成目标并不完全一致。

因此，大模型可能出现：

- 解释源句而不是翻译。
- 自动补充背景信息。
- 过度压缩。
- 输出混合语言。
- 对低资源目标语言生成不稳定。

### 3.2.2 Prompting 的能力边界

Prompt 可以激活模型已有的翻译能力，但难以创造模型没有学到的语言知识。

有效 prompt 能改善：

- 输出格式
- 是否只输出译文
- 是否避免解释
- 是否保留数字和实体

但 prompt 难以解决：

- 目标语言词汇缺失
- 平行映射缺失
- tokenizer 表示差
- 语言模型目标端能力弱

### 3.2.3 SFT 与方向专门化

对低资源翻译，SFT 的作用是把通用生成模型拉向翻译任务分布。

训练方式可以比较：

1. **single SFT**

   所有方向混在一起训练一个 adapter。

   优点：

   - 简单。
   - 能共享跨语言知识。
   - 部署方便。

   缺点：

   - 方向之间可能互相干扰。
   - 高资源方向容易压制低资源方向。
   - 目标语言分布不均衡。

2. **pair-level expert / MoE**

   每个语言对或方向训练独立 expert。

   优点：

   - 方向专门化强。
   - 容易控制数据配比。
   - 低资源方向不会被高资源方向完全淹没。

   缺点：

   - 训练和部署复杂。
   - 需要路由。
   - 数据太少时 expert 容易过拟合。

### 3.2.4 多任务混合与负迁移

多语训练的核心矛盾是：

> 共享参数带来迁移，混合过度带来干扰。

正迁移可能来自：

- 共享词汇或脚本。
- 相似语序。
- 相同语系。
- 共同的英语 pivot 表示。

负迁移可能来自：

- 语言间脚本差异大。
- 高资源语言主导梯度。
- 目标语言混淆。
- 方向标签不够明确。

因此需要观察：

- single SFT 是否提升所有方向，还是只提升高资源方向？
- pair expert 是否显著提升低资源方向？
- 中文方向是否需要单独 expert？

### 3.2.5 Draft-Refine 学习视角

可以把翻译分成两个阶段：

```text
source sentence -> draft translation -> refined translation
```

这种方式的潜在价值：

- draft 阶段保证语义覆盖。
- refine 阶段提高目标语言自然度。
- 对长句和中文方向可能更稳。

风险：

- refine 可能改变原意。
- 训练数据构造成本更高。
- 如果 teacher 本身有偏差，refine 会放大偏差。

### 3.2.6 Learning 视角总结

从学习角度看，低资源翻译的改进路径是：

1. 用 SFT 把 LLM 从开放生成拉到翻译分布。
2. 用方向标签和清晰指令降低语言混淆。
3. 用 pair-level expert 减少方向间干扰。
4. 用数据配比控制高资源语言主导问题。
5. 用 draft-refine 或 reranking 改善忠实性与流畅度平衡。

---

## 4. Data & Learning 的统一框架

可以把整个研究框架总结为：

```text
Evaluation
  -> Identify capability gaps

Data Perspective
  -> Which language pairs lack evidence?
  -> Which target languages lack fluent text?
  -> Which terms/entities are missing?

Learning Perspective
  -> How should the model absorb the data?
  -> Shared model or pair expert?
  -> Direct translation or draft-refine?

Ablation
  -> Verify which factor actually improves BLEU/COMET
```

核心假设：

> 低资源语言翻译质量由“数据可见性”和“学习可用性”共同决定。前者决定模型能看到什么，后者决定模型是否能把看到的东西稳定地转化成翻译能力。

---

## 5. 建议实验设计

### 5.1 Baseline

建议至少比较：

1. 通用 LLM prompt baseline
2. NLLB-200 3.3B
3. NLLB-MoE 54B
4. single SFT
5. pair-level LoRA expert / MoE
6. dictionary-mixed variant

### 5.2 Ablation

可以设计如下消融：

| 实验 | 目的 |
|---|---|
| no SFT | 测试原始大模型翻译能力 |
| NLLB data only | 测试句级平行数据作用 |
| NLLB + FineWeb synthetic | 测试合成数据作用 |
| NLLB + dictionary 5% | 测试词汇锚点作用 |
| single SFT | 测试共享 adapter |
| pair expert | 测试方向专门化 |
| draft-refine | 测试两阶段学习 |

### 5.3 主要图表

建议报告中放：

1. BLEU heatmap
2. COMET heatmap
3. Delta heatmap：模型 A - baseline
4. 方向分组柱状图：

   - English -> low-resource
   - low-resource -> English
   - Chinese -> low-resource
   - low-resource -> Chinese

5. 语言平均表现表
6. 典型错误案例表

### 5.4 结果解读方式

不要只报告 overall average。更重要的是：

- 哪些语言方向提升最大？
- 哪些方向退化？
- BLEU 提升但 COMET 不提升，说明什么？
- COMET 提升但 BLEU 不提升，说明什么？
- 中文方向和英语方向差异是否缩小？
- pair expert 是否比 single SFT 更适合低资源方向？

---

## 6. 可讲的核心结论

可以把最终结论组织成四点：

1. **大模型具备低资源翻译能力，但能力高度不均衡。**

   语言方向、目标语言、脚本和数据覆盖都会造成明显差异。

2. **低资源翻译不能只依赖模型规模。**

   NLLB-MoE 这类大规模专用 MT 模型可以作为强基线，但项目模型仍需要针对目标方向优化数据和学习方式。

3. **Data perspective 解释能力来源。**

   平行数据、合成数据、词典数据分别提供句级映射、覆盖扩展和词汇锚点。

4. **Learning perspective 解释能力如何稳定落地。**

   single SFT、pair expert、MoE、draft-refine 分别对应不同的共享与专门化策略。

最终主张：

> 面向低资源语言翻译，最佳策略不是简单扩大模型，而是以评测驱动发现薄弱方向，再从 data 和 learning 两个视角共同修复：用高质量方向数据补证据，用专门化学习机制减少跨语言干扰。

---

## 7. 报告结构建议

如果做 20 分钟报告，可以这样安排：

1. Motivation and Problem Setup：2 分钟
2. Evaluation Protocol：3 分钟
3. Empirical Findings：6 分钟
4. Data Perspective：4 分钟
5. Learning Perspective：4 分钟
6. Takeaways and Next Steps：1 分钟

如果写论文草稿，可以对应为：

1. Introduction
2. Related Work
3. Experimental Setup
4. Empirical Analysis
5. Data Perspective
6. Learning Perspective
7. Discussion
8. Conclusion

---

## 8. 下一步可补充内容

后续可以继续补三类内容：

1. **定量结果**

   把 `metrics.csv`、热力图和 delta heatmap 填入第 2 部分。

2. **错误案例**

   从 `hypotheses.jsonl` 中挑每个语向 2-3 个代表错误。

3. **消融实验**

   对比 single SFT、pair expert、dictionary mix、NLLB baseline 的差异。

