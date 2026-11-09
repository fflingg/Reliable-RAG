# Selective RAG baseline：论文 §5.1 的项目适配

## 任务分析与实现范围

截图中的“文章第五节”对应仓库 `doc/SUFFICIENT CONTEXT A NEW LENS ON RETRIEVAL.pdf`：
Joren et al., *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems*, ICLR 2025。
原文：https://arxiv.org/abs/2411.06037 。

§5.1 将上下文充分性二元标签和回答模型自评置信度组合，通过逻辑回归控制回答/拒答，
比较选择性准确率与覆盖率。§5.2 是微调实验，不属于本目录 baseline。
这里检验的假设是：在相同原回答上，充分性信号是否比仅使用置信度带来更好的取舍。
即使没有提升，也是可以报告的结果；不能预先声称有效。

课程 handout 要求合适的 baseline、有动机的修改或比较、测试集与开发分离、失败分析、
可复现代码和 AI Use Statement。因此 baseline 只是项目的一部分，后续仍需由团队确定并验证
自己的修改；本目录不把论文已有方法写成团队创新。

实施顺序：审计已有结果 → 冻结共同划分 → 采集自评置信度 → 训练/验证门控器 → 测试集对照与曲线。

## 已核对的项目状态（2026-11-09）

当前冻结样本 459 条，其中 169 sufficient、290 insufficient。充分性来自 Qwen 自动标注，
并非人类真值。`musique_valid_no_abstain_hint` 下逐条核对回答及判断指纹：

| 回答模型 | 有效分类 | 未分类 | correct / abstain / hallucinate |
| --- | ---: | ---: | --- |
| Qwen | 376 | 83（81 生成无效、2 评估无效） | 305 / 38 / 33 |
| Llama | 393 | 66（65 生成无效、1 评估无效） | 148 / 86 / 159 |
| Mistral | 459 | 0 | 71 / 218 / 170 |

这是已有行为实验的状态，不是新 baseline 的成绩。默认拒绝使用不完整模型结果。
可先跑完整的 Mistral；其他两个模型若不能完成，只能显式 `--allow-partial` 并报告选择偏差。
不同模型的 partial 子集不同，不能据此做公平的跨模型优劣排名。

## 方法与原文的对应关系

- 复用现有原回答，门控只决定保留回答或改成拒答，不重新选答案。
- 对每个原本尝试回答的样本，用**同一个回答模型与同一个模型 commit** 采样 20 个回答，
  temperature=1、top_p=1、top_k=0，提示词复用冻结的原生成提示词。
- 用这些回答作 brainstorm，自评原回答是否正确。默认 5 次，每次确定性打乱 brainstorm 顺序。
  模型只看问题、原上下文、未验证的采样回答与原回答，不看参考答案、充分性标签或答案 evaluator 的判断。
- 自评采用 A=正确、B=错误的聊天提示词，读取下一个 token 的
  `P(A)/(P(A)+P(B))`，对 5 次取均值。这是 **P(True) 的具体项目适配**，不是投票正确率，
  也不是答案 evaluator 的置信度。另存 A/B 在全词表中的概率质量，以便发现模型不遵循格式的问题。
- 上下文充分性直接复用现有 Qwen 二元标签，不重新调用 FLAMe。
- 逻辑回归训练 `P(correct)`，高分保留、低分拒答；等价于对错误风险作反向排序。
  原生拒答不参与 correct/incorrect 的二分类训练，也不会被门控变成回答。
- 每个逻辑回归对照均用相同随机种子抽取 100 个对数均匀的 C（1e-4 到 1e4），
  在 train 拟合，以 validation log loss 选参数；选中模型不再重拟合。

论文对开源模型使用 P(True) 的 20 次采样、5 次评估，对闭源模型使用 P(Correct)。
本实现保留开源路线和采样次数，但自评聊天模板、5 次打乱的聚合、模型、数据和充分性标注器
均有项目适配，**不能写成逐字/逐设置完全复现**。自评参考 Kadavath et al. §4.2/A.5 的
brainstorm + 判断指定答案方式：https://arxiv.org/abs/2207.05221 。本实现未复制其 few-shot 示例。

对照方法：

| 方法 | 含义 |
| --- | --- |
| native_fixed | 原模型，不新增拒答 |
| sufficiency_only_fixed | 仅保留 sufficient 上下文中的原非拒答回答 |
| confidence_raw | 直接按 P(True) 阈值拒答 |
| confidence_lr | 单特征逻辑回归，控制逻辑回归校准的影响 |
| combined_lr | 两特征逻辑回归，即 §5.1 的核心 baseline |

核心比较是 `combined_lr` 与 `confidence_lr`。不能只与原模型比较，因为单纯大量拒答也会抬高准确率。

## 划分和指标

先对完整冻结样本按规范化问题分组，60%/20%/20% 划分 train/validation/test，同一问题的
多个上下文不能跨组。固定 seed=42 后为 **276/91/92 条**，所有模型共用。
划分不读取回答正确性。partial 模式只在同一划分里排除无效样本。

这批 MuSiQue dev 样本此前已用于行为分析，新的内部划分不能消除已经查看过数据的事实。
报告应称为“固定子集上的内部 held-out 评估”；如需严格的新测试证据，应另取未用于开发的样本。

- coverage = 保留的实质回答数 / 该 split 全部样本数。
- selective_accuracy = 保留的正确回答数 / 保留的实质回答数。
- selective_risk = 保留的错误回答数 / 保留的实质回答数。
- hallucination_rate_all = 保留的错误回答数 / 该 split 全部样本数。
- 零回答时 selective_accuracy/risk 为 null，不能填成 100% 或 0%。
- 原生拒答始终算拒答；只在 non-abstain 子集算分母会虚增 coverage。
- 所有覆盖率阈值由 validation 分数选定，随后直接用于 test；不强行让 test 达到目标覆盖率。
  同分样本一起保留或拒绝，不按 test 正确性打破 ties。
- `accuracy_coverage.csv` 用 validation 分数的唯一阈值生成 test 上的曲线，
  是冻结阈值族的描述；不应在看完 test 后挑选“最好阈值”作为可部署成绩。

这里的 hallucinate 沿用现有 evaluator 的“错误回答”定义，不是逐句证据忠实度检测。
Qwen 同时参与充分性标注、答案评估和回答生成，有自评/同家族偏差，需配合已有人工复核表。

## 运行

所有命令均从项目根目录执行。CPU 的 audit/split 不需要 GPU 或第三方依赖。
推理沿用现有 TC1 V100、`.venv313`、`.model_cache` 和模型访问权限，不调用付费 API。

```bash
python3 src/selective_rag/selective_rag.py audit
python3 src/selective_rag/selective_rag.py split
uv pip install --python .venv313/bin/python -r src/selective_rag/requirements.txt
```

先用独立目录做两条 smoke；只采集置信度，不拟合，不能作为实验结论：

```bash
python3 src/selective_rag/selective_rag.py split --output-dir results/selective_rag/smoke2
SELECTIVE_STAGE=confidence sbatch --time=01:00:00 src/selective_rag/run_selective_tc1.sh mistral \
  --limit 2 --samples 2 --self-evaluations 2 --output-dir results/selective_rag/smoke2
```

正式运行 Mistral（默认 20 次采样、5 次自评、100 次参数搜索）：

```bash
sbatch src/selective_rag/run_selective_tc1.sh mistral
```

原始非拒答样本有 241 条，因此 Mistral 正式运行约需 **4820 次采样生成 + 1205 次自评前向**。
不是 241 次简单打分，具体 GPU 用时要以 smoke 实测为准。12 小时只是脚本请求的时限，
不保证队列允许或能一次完成；可按集群配额修改时限，同一命令支持恢复已完成的采样/自评。

Qwen/Llama 先处理现有生成输出触顶和无效评估。现有 greedy 输出触顶，原预算原参数重试很可能
得到相同结果；不能把它们当成错误回答或拒答。增加预算会改变实验配置，应在新的行为目录重跑，
再用 `--behavior-dir 新目录 --output-dir 新baseline目录` 对应运行。

如确需仅研究现有有效子集：

```bash
sbatch src/selective_rag/run_selective_tc1.sh llama31 --allow-partial --output-dir results/selective_rag/exploratory_valid_only
```

CPU 重做训练/绘图，无需再次加载回答模型：

```bash
.venv313/bin/python src/selective_rag/selective_rag.py fit --model mistral
.venv313/bin/python src/selective_rag/selective_rag.py plot --model mistral
```

**先在服务器生成共同 split 一次，再提交多个模型任务**，避免多个任务同时创建同一划分文件。
不要在同一模型的相同输出目录同时启动多个任务。改变模型、输入、标签、提示词、采样预算、
搜索设置或阈值目标时，冻结配置检查会报错，应使用新的输出目录。
旧 `src/rag_behavior`、`src/autorater` 与已有实验结果均不修改。

## 输出

```text
results/selective_rag/musique_valid/
  audit.json / split.json
  mistral/
    cohort.json / excluded.jsonl
    confidence_config.json / confidence_environment.json
    samples.jsonl / self_evaluations.jsonl / confidence.jsonl
    fit_config.json
    confidence_lr.json / combined_lr.json  # 可读系数、截距、C 和全部验证搜索记录
    predictions.jsonl                      # 各 split 的分数和标签
    operating_points.csv                   # 验证阈值在 test 上的实际表现
    accuracy_coverage.csv / .png / .pdf
    summary.json
```

检查 `operating_points.csv` 中实际 test coverage 相近的点、两种 LR 的曲线，以及被新增拒答的
correct/incorrect 示例。小测试集的高准确率可能只来自少数回答，应同时报告 answered 数量，
不能只摘最高准确率。bootstrap 置信区间、多种子评估和独立新方法仍可作为后续扩展。

## 验证与 AI 使用

```bash
.venv313/bin/python -m unittest discover -s src/selective_rag -p 'test_*.py' -v
.venv313/bin/python -m unittest discover -s src/rag_behavior -p 'test_*.py' -v
bash -n src/selective_rag/run_selective_tc1.sh
```

测试含模拟推理中断恢复、提示词泄漏检查、重复问题隔离、错误/过期结果拒绝、
原生拒答和零覆盖率分母，以及合成数据 CPU 完整训练/绘图。
特别检查：只翻转 test 的正确/错误标签，拟合系数、参数选择和验证阈值必须不变。
合成测试结果不会写进正式实验目录。

本目录代码由 Codex 辅助实现。团队需审阅、在实际 GPU 上验证并自行解读结果，
并将工具用途、核验方式与自己的决策写入最终 AI Use Statement。
本地代码测试通过不代表真实模型推理已运行，也不能据此填报 baseline 的提升数值。
