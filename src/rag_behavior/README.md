# Categorizing RAG Responses: Sufficient vs Insufficient Context

## 论文风格饼图

```bash
python src/rag_behavior/plot_pies.py --allow-partial
```

按模型分行，按 With RAG / Sufficient / Insufficient 分列，输出
`response_categories_pies.png`、`.pdf`、`.svg` 到当前实验结果目录。
没有 Without RAG 实验，因此不绘制该列。当前 incomplete 数据必须显式传入
`--allow-partial`，图中标明每个分母与未分类排除情况；不能作为完整实验图引用。

## 2026-11-04：去除明确拒答指令的对照实验

当前回答提示词已删除 `If you cannot determine the answer, use "The answer is: I don't know".`。
其余提示词、示例、数据、生成预算（512）、评估预算（256）和分类规则保持不变，以隔离这一处改动。
这不是完全复现论文的提示词，只是去除明确拒答引导的对照。

新默认输出为 `results/rag_behavior/musique_valid_no_abstain_hint/`；两条 smoke 输出为
`results/rag_behavior/smoke2_no_abstain_hint/`，旧 `musique_valid/` 结果保留。
`plot_results.py` 默认读取新目录。查看旧汇总可显式传入 `--results-dir results/rag_behavior/musique_valid`。

上传更新后的文件夹后，仍使用下面的三个 sbatch 命令提交，新实验会从头生成回答。
不要将旧目录作为本轮 `--results-dir`，冻结配置检查会拒绝不同提示词混用。
后文有关旧拒答提示和旧结果路径的说明描述首轮实验；当前实验以本节为准。

比较 `Qwen/Qwen3.8-27B`、`meta-llama/Llama-3.1-8B-Instruct`、`mistralai/Mistral-7B-Instruct-v0.3`。
一个任务只运行一个回答生成模型，随后评估该模型的回答。支持相同命令断点续跑。
此目录不修改 `parta.ipynb`，也不覆盖前面的 autorater / MuSiQue 标注结果。

## 数据与定义

`prepare` 从现有 `labelled_dataset.jsonl` 选取 `rater_status=ok` 且标签为整数 0/1 的所有记录。
当前冻结数据 459 条：169 sufficient、290 insufficient，排除 41 invalid（已恢复的三个字段空格问题计入有效数据）。
不平衡采样、不再次截断、不重新检索，不把原始 answerable 当作充分性真值。
排除 invalid 可能引入选择偏差；结论适用于该有效子集，不能写成原始全部 500 条的结果。
`manifest.json` 和 `excluded.jsonl` 保留筛选依据；冻结后不随源标注变化自动扩充。

- correct：最终回答与答案或别名规范化后相同；其余回答由语义 evaluator 判为 perfect / acceptable。
- abstain：没有回答，或明确表示不知道、无法判断。对带有实质回答的模糊免责声明交给 evaluator 判断。
- hallucinate：evaluator 判为 incorrect，沿用论文的“错误回答”口径；不是独立的逐句事实性或证据忠实度检测。
- 生成错误、达到输出上限、评估输出格式无效：unclassified，不能伪装为以上任何一类。

上下文 insufficient 时答对仍属于 correct；分组标签不提供给生成模型或答案 evaluator。
两个分组各以本组已分类数量为百分比分母，另外报告 total、classified、unclassified、complete。
正式比较应要求三个模型的全部样本均已分类；未完成结果只可视为阶段性结果。

## 与论文的关系

参考本项目 `doc/SUFFICIENT CONTEXT A NEW LENS ON RETRIEVAL.pdf` §4.2、Appendix C.3/C.4。
使用统一的简短说明 + `The answer is:` 格式，带一个固定且不重叠的 QA 示例。
为使拒答可明确解析，增加“无法判断时回答 I don't know”的统一指令；不是论文逐字复现。
不按参考答案子串直接匹配整段解释，以避免只提及答案却实际答错的假阳性。
不确定匹配由 Qwen3.8-27B NF4/FP16 进行零样本语义判断：perfect/acceptable → correct；missing → abstain；incorrect → hallucinate。
评估器可以看目标标准答案及别名，生成模型只能看目标问题与上下文。
Qwen 同时是上下文标注器、答案 evaluator，并参与生成比较，存在同家族/自评偏差；需要报告该限制并进行人工复核。
每个模型生成同一组随机 50 条的人工评估表，保留人工编辑；不覆盖现有表格。

## 模型与硬件

| 参数别名 | 模型 | 权重 / 计算精度 |
|---|---|---|
| qwen38 | Qwen/Qwen3.8-27B | NF4 / FP16 |
| llama31 | meta-llama/Llama-3.1-8B-Instruct | FP16 / FP16 |
| mistral | mistralai/Mistral-7B-Instruct-v0.3 | FP16 / FP16 |

针对 TC1 V100 32GB，batch size 1，使用 SDPA；不要求 FlashAttention。
Qwen 模型类为 Qwen3_5ForConditionalGeneration，其余为 AutoModelForCausalLM。
使用各自官方聊天模板，greedy，Qwen non-thinking；所有模型看到相同的正文和示例。
生成最多 512 tokens，评估最多 256 tokens；完整提示词及输出预算不得超过 16384，不静默截断上下文。
Qwen 与其余模型的规模和量化精度不同，比较的是此部署设置的行为，不是隔离模型家族的实验。
首次运行解析并冻结模型实际 commit，续跑沿用；evaluator 默认固定为此前使用的 Qwen commit。

## 本地已经准备的数据

```text
data/rag_behavior/musique_valid/
  dataset.jsonl
  manifest.json
  excluded.jsonl
```

需要重新创建独立数据集时：

```bash
python src/rag_behavior/rag_behavior.py prepare
```

不要在服务器重新按旧版 MuSiQue 合并结果生成数据，直接上传本地已冻结数据，以保证三个模型使用完全相同的 459 条。

## 上传与服务器运行

在本地 WSL 项目根目录执行：

```bash
ssh n2602689b@10.96.189.11 'mkdir -p ~/sc4001_project/data/rag_behavior'
scp -r src/rag_behavior n2602689b@10.96.189.11:~/sc4001_project/src/
scp -r data/rag_behavior/musique_valid n2602689b@10.96.189.11:~/sc4001_project/data/rag_behavior/
```

沿用 `.venv313` 及 `.model_cache`，现有 autorater 依赖一般已满足。需要安装时：

```bash
cd ~/sc4001_project
uv pip install --python .venv313/bin/python -r src/rag_behavior/requirements.txt
```

Llama 权重下载前，须在官方页面 https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
接受协议并获得访问权限，再交互登录 Hugging Face（不要将 token 写进脚本）：

```bash
.venv313/bin/hf auth login
```

先在独立目录测试两条，任务内会生成回答并评估：

```bash
sbatch src/rag_behavior/run_rag_tc1.sh qwen38 --limit 2
sbatch src/rag_behavior/run_rag_tc1.sh llama31 --limit 2
# 等有提交名额后再提交第三个（账号 MaxSubmit=2）
sbatch src/rag_behavior/run_rag_tc1.sh mistral --limit 2
```

正式运行，不带 limit：

```bash
sbatch src/rag_behavior/run_rag_tc1.sh qwen38
sbatch src/rag_behavior/run_rag_tc1.sh llama31
# 等有提交名额后再提交
sbatch src/rag_behavior/run_rag_tc1.sh mistral
```

每个任务先加载自己的回答模型，完成生成后释放显存，再加载 Qwen evaluator。
超时后重新提交同一命令；完成的生成和有效评估跳过，生成失败/截断或运行错误会重试。
评估 JSON 无效默认保留，可在相同命令后加 `--retry-invalid` 重试；相同 greedy 参数可能仍产生相同无效输出。
更改预算、提示词或数据时必须使用另一个 `--results-dir`，避免混淆实验设置。

单独阶段（generate/evaluate 只能在 GPU 分配内执行）：

```bash
.venv313/bin/python src/rag_behavior/rag_behavior.py generate --model llama31
.venv313/bin/python src/rag_behavior/rag_behavior.py evaluate --model llama31
# report 不加载模型，可在登录节点汇总完整实验
.venv313/bin/python src/rag_behavior/rag_behavior.py report
```

## 输出与下载

```text
results/rag_behavior/musique_valid/
  metrics.csv                    # 三种模型 × all/sufficient/insufficient 的数量和比例
  summary.json                   # 同一张表的 JSON
  qwen38/                        # llama31/、mistral/ 结构相同
    generation_config.json       # 数据、提示词、精度、模型 commit
    generation_environment.json
    responses.jsonl              # 追加写入；原始回答、token 数、状态、耗时
    evaluation_config.json
    evaluation_environment.json  # 实际加载 evaluator 后生成
    judgments.jsonl              # 决策、类别、解释、原始评估输出、回答指纹
    categorized.jsonl            # 每个问题的上下文组别、回答与最终类别
    human_review_blind.csv       # 随机抽查；人工类别留空
```

Smoke 结果独立位于 `results/rag_behavior/smoke2/`，不覆盖正式输出。
结果尚未生成前不要填写论文中的数值；当前本地只完成数据冻结与代码验证，没有执行 GPU 推理。

本地 WSL 下载：

```bash
scp -r n2602689b@10.96.189.11:~/sc4001_project/results/rag_behavior ./results/
```

三种模型都完成后可生成与论文 Figure 3 类似的两行三列图（上：sufficient，下：insufficient）：

```bash
.venv313/bin/python src/rag_behavior/rag_behavior.py report
.venv313/bin/python src/rag_behavior/plot_results.py
```

生成 `response_categories.png` / `response_categories.pdf`。脚本拒绝为未完成的比较绘制正式图。
