# Baseline 运行指南

本指南适用于本项目 `src/selective_rag/` 中的 baseline。命令按顺序执行，并区分 **Mac 本地终端** 与 **服务器终端**。

建议先运行 **Mistral**。截至 2026-11-09，本地保存的 Mistral 行为实验有完整的 459 条分类结果；Qwen 和 Llama 分别仍有 83、66 条未分类，默认无法用于完整 baseline。

目前已经完成代码及本地测试，**尚未完成真实 GPU 置信度采集，没有新的 baseline 实验成绩**。本指南中的服务器地址、目录和环境需要在实际登录后核实。

## 1. 这次运行的内容

Baseline 对应论文 *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems* 的 §5.1。

它复用已有回答，采集回答模型的自评置信度，再比较“只用置信度”和“置信度＋上下文充分性”的逻辑回归门控器，决定保留回答还是拒答。

运行流程：

```text
检查已有数据和回答结果
        ↓
固定训练 / 验证 / 测试划分
        ↓
在 GPU 上进行回答采样和自评
        ↓
训练小型逻辑回归，使用验证集选择参数和阈值
        ↓
测试集评估，输出表格与 accuracy–coverage 曲线
```

回答模型的权重保持不变。这里训练的是小型逻辑回归，不是微调 Mistral。

## 2. 校外先连接 VPN，再登录服务器

**操作位置：Mac 本地。**

项目文档中的服务器地址为 `10.96.189.11`，属于内网地址。在校外连接时，通常需要先连接 NTU VPN；如果课程提供了其他入口或跳板机，使用课程说明中的入口。

1. 打开 GlobalProtect，连接 NTU VPN，确认显示 `Connected`。
2. 打开终端，将下列命令中的 `你的服务器账号` 替换为课程分配给你的账号：

```bash
ssh -o ConnectTimeout=10 你的服务器账号@10.96.189.11
```

不要直接使用其他同学的账号。之前示例中的 `n2602689b` 来自项目文档，不代表一定是你的账号。

首次连接可能出现服务器指纹确认，按课程提供的主机信息核对；随后按提示登录。输入密码时终端通常不显示字符，这是正常现象。

### SSH 一直不动怎么办

如果尚未出现密码提示，先按 `Ctrl+C` 结束等待，检查 VPN，再测试 SSH 端口：

```bash
nc -vz -G 10 10.96.189.11 22
```

此命令在 Mac 上使用，最多等待约 10 秒。

| 输出 | 含义与下一步 |
| --- | --- |
| `succeeded` | SSH 端口可达，继续尝试登录 |
| `timed out` | 网络路径未通；检查 VPN、访问权限或服务器状态 |
| `Connection refused` | 目标有响应，但 SSH 端口未接受连接 |

端口可达但 SSH 仍失败时，可运行详细诊断，并把末尾的报错交给队友或管理员检查：

```bash
ssh -v -o ConnectTimeout=10 你的服务器账号@10.96.189.11
```

## 3. 找到服务器上的项目目录

**操作位置：登录后的服务器终端。**

先查看当前位置和主目录：

```bash
pwd
ls ~
```

如果看到了 `sc4001_project`：

```bash
cd ~/sc4001_project
pwd
ls -a
```

预期项目根目录下有 `src/`、`data/`、`results/` 等内容。`pwd` 输出的完整路径就是后续上传时需要的服务器项目路径。

如果找不到项目，可以在主目录中搜索已有脚本：

```bash
find ~ -maxdepth 6 -type f -name 'rag_behavior.py' 2>/dev/null
```

例如找到：

```text
/home/你的账号/sc4001_project/src/rag_behavior/rag_behavior.py
```

则项目根目录是 `/home/你的账号/sc4001_project`。如果没有找到，项目可能在共享目录或队友的账号下，应向队友确认真实路径和访问权限。

**后续所有服务器命令，都先进入确认好的项目根目录。**

## 4. 上传新增代码，确认输入文件齐全

**操作位置：新开一个 Mac 本地终端，保留已登录的服务器终端。**

先设置上传目标。请修改引号内的账号和完整路径，不要原样复制占位内容：

```bash
RAG_SERVER='你的服务器账号@10.96.189.11'
RAG_REMOTE_ROOT='/实际/项目/完整路径'
cd /Users/syh/Reliable-RAG
```

确认服务器已经有项目的 `src/` 目录后，上传新增代码：

```bash
scp -r src/selective_rag "$RAG_SERVER:$RAG_REMOTE_ROOT/src/"
```

回到**服务器终端**，进入项目根目录后，检查以下输入：

```bash
ls src/selective_rag/
ls src/rag_behavior/rag_behavior.py
ls src/rag_behavior/requirements.txt
ls src/autorater/requirements-autoraters.txt
ls data/rag_behavior/musique_valid/
ls results/rag_behavior/musique_valid_no_abstain_hint/
```

需要的结构如下：

```text
项目根目录/
  src/
    selective_rag/                         # 本次新增
    rag_behavior/
      rag_behavior.py
      requirements.txt
    autorater/
      requirements-autoraters.txt
  data/rag_behavior/musique_valid/
    dataset.jsonl
    manifest.json
  results/rag_behavior/musique_valid_no_abstain_hint/
    qwen38/
    llama31/
    mistral/
```

每个模型的行为结果目录需要有 `generation_config.json`、`evaluation_config.json`、`responses.jsonl` 和 `judgments.jsonl`。`audit` 会检查三个模型；正式跑 Mistral 时使用其中的 Mistral 输入。

如果服务器尚未上传冻结数据和行为结果，可在 **Mac 本地终端**执行：

```bash
ssh "$RAG_SERVER" "mkdir -p '$RAG_REMOTE_ROOT/data/rag_behavior' '$RAG_REMOTE_ROOT/results/rag_behavior'"

scp -r data/rag_behavior/musique_valid \
  "$RAG_SERVER:$RAG_REMOTE_ROOT/data/rag_behavior/"

scp -r results/rag_behavior/musique_valid_no_abstain_hint \
  "$RAG_SERVER:$RAG_REMOTE_ROOT/results/rag_behavior/"
```

如果服务器已有更新的实验结果，先确认应使用哪一版，避免用本地旧结果覆盖。已有行为代码也应与对应实验配置保持一致。

## 5. 检查 Python 环境并安装依赖

**操作位置：服务器终端、项目根目录。**

脚本默认使用项目中的 `.venv313` 环境和 `.model_cache` 模型缓存：

```bash
ls -l .venv313/bin/python
.venv313/bin/python --version
```

存在这个环境时，安装依赖：

```bash
uv pip install --python .venv313/bin/python \
  -r src/selective_rag/requirements.txt
```

依赖文件沿用现有推理依赖，并增加 scikit-learn 和 NumPy；绘图依赖通过原有 requirements 引入。

若没有 `uv`，且该环境已经有 pip，可以使用：

```bash
.venv313/bin/python -m pip install -r src/selective_rag/requirements.txt
```

若 `.venv313` 不存在，先确认队友已有的 CUDA/PyTorch 推理环境，使用它的 Python 安装依赖。不要把 Mac 上的虚拟环境复制到服务器。

使用其他环境时，提交任务前设置真实的环境目录，例如：

```bash
export AUTORATER_VENV='/实际/环境目录'
```

此后本指南中的 `.venv313/bin/python` 也应替换成该环境的 Python 完整路径。脚本会读取 `AUTORATER_VENV`。如模型缓存不在项目的 `.model_cache` 中，可设置：

```bash
export HF_HOME='/实际/模型缓存目录'
```

沿用已经能运行 Mistral 行为实验的环境和缓存最直接。首次缺少权重时，任务还需要下载模型；模型访问权限或网络问题会反映在任务日志中。

## 6. 审计输入并固定数据划分

**操作位置：服务器终端、项目根目录。无需 GPU。**

```bash
.venv313/bin/python src/selective_rag/selective_rag.py audit
```

按当前本地输入，Mistral 应显示：

```text
total: 459
usable: 459
complete: true
```

服务器使用其他结果版本时，以实际审计输出为准。如果提示缺少文件、指纹不一致或数据不匹配，先解决输入问题。

然后固定正式实验划分：

```bash
.venv313/bin/python src/selective_rag/selective_rag.py split
```

当前冻结数据和默认 seed=42 对应：

```text
train: 276
validation: 91
test: 92
```

同一问题的不同上下文不会跨组，所有模型共用划分。正式划分保存在 `results/selective_rag/musique_valid/split.json`。

## 7. 先运行两条小测试

**操作位置：服务器终端、项目根目录。**

小测试使用独立结果目录，先生成该目录下的划分：

```bash
.venv313/bin/python src/selective_rag/selective_rag.py split \
  --output-dir results/selective_rag/smoke2
```

提交 GPU 小测试：

```bash
SELECTIVE_STAGE=confidence sbatch --time=01:00:00 \
  src/selective_rag/run_selective_tc1.sh mistral \
  --limit 2 \
  --samples 2 \
  --self-evaluations 2 \
  --output-dir results/selective_rag/smoke2
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `SELECTIVE_STAGE=confidence` | 只执行置信度采集 |
| `mistral` | 使用 Mistral 回答模型 |
| `--limit 2` | 只处理前两条有效记录 |
| `--samples 2` | 每个需要采集置信度的样本生成 2 个回答 |
| `--self-evaluations 2` | 对原回答做 2 次自评 |
| `--output-dir .../smoke2` | 与正式结果分开保存 |

小测试检查模型加载、采样、自评和保存是否正常，不拟合逻辑回归，也不会生成正式评估曲线。

提交后会显示类似 `Submitted batch job 123456`。这是任务编号，不表示任务已经开始执行。

### 查看队列与日志

```bash
squeue -u "$USER"
```

常见状态：`PD` 表示排队，`R` 表示运行。任务不在队列后，还要查看日志或任务记录，确认是成功完成还是失败。

将下面的 `123456` 替换为实际任务编号：

```bash
tail -f selective_123456.out
```

按 `Ctrl+C` 退出日志查看；它不会取消服务器任务。

```bash
tail -n 80 selective_123456.err
sacct -j 123456 --format=JobID,State,ExitCode,Elapsed
```

`sacct` 是否可用取决于集群配置。正常完成应没有 Python traceback，任务状态为 `COMPLETED`、退出码为 `0:0`。`.err` 中可能包含正常的模型加载进度或警告，不能只凭文件非空判断失败。

检查小测试输出：

```bash
cat results/selective_rag/smoke2/mistral/confidence.jsonl
```

当前完整输入的两条小测试成功后，应有两条 `status: ok` 的置信度记录。确认成功后再正式提交。

## 8. 提交正式 Mistral baseline

**操作位置：服务器终端、项目根目录。**

```bash
sbatch src/selective_rag/run_selective_tc1.sh mistral
```

脚本自动依次执行划分检查、置信度采集、训练、评估和绘图。默认设置为：

- 每个原非拒答样本采样 20 次，自评 5 次。
- 逻辑回归每种对照搜索 100 个正则化参数。
- 申请 1 张 GPU、64 GB 内存，任务时限 12 小时。

12 小时是请求的上限，并非预计耗时或集群保证。如果集群只允许 6 小时，可覆盖时限：

```bash
sbatch --time=06:00:00 src/selective_rag/run_selective_tc1.sh mistral
```

当前 Mistral 有 241 条非拒答回答，对应约 **4820 次采样生成＋1205 次自评前向计算**。实际用时以服务器实测为准。

### 中断或超时后的恢复

任务已结束后，重新提交与之前相同的命令。程序复用已保存的采样、自评和有效置信度记录。

不要同时向同一个模型的同一个输出目录提交多个任务。需要主动取消时：

```bash
scancel 123456
```

更改采样数、提示词、输出预算、数据、模型配置或训练设置时，应使用新输出目录；不要删除冻结配置文件来绕过检查。

## 9. 查看正式结果

**操作位置：服务器终端、项目根目录。**

```bash
ls results/selective_rag/musique_valid/mistral/
```

主要文件：

| 文件 | 用途 |
| --- | --- |
| `confidence.jsonl` | 每条样本的自评置信度；原生拒答记录的置信度为空 |
| `samples.jsonl` | 采样回答及状态 |
| `self_evaluations.jsonl` | 自评概率及 A/B 标签概率质量 |
| `confidence_lr.json` | 仅置信度逻辑回归参数及搜索记录 |
| `combined_lr.json` | 置信度＋充分性逻辑回归参数及搜索记录 |
| `operating_points.csv` | 验证阈值在测试集上的表现 |
| `accuracy_coverage.csv` | 曲线数据 |
| `accuracy_coverage.png` / `.pdf` | 结果曲线 |
| `summary.json` | 汇总、数据数量与局限说明 |

核心比较是 `combined_lr` 与 `confidence_lr`：在实际覆盖率相近的情况下，组合信号能否提高准确率。还会输出原始回答、仅充分性门控和原始置信度门控作为参照。

看结果时同时报告 `answered`、`coverage` 和 `selective_accuracy`。目标覆盖率由验证集选阈值，实际测试覆盖率可能不同；不能只挑准确率最高的点。

## 10. 下载结果到 Mac

**操作位置：Mac 本地终端。保持 VPN 连通。**

如果换了新的本地终端，重新设置第 4 步中的 `RAG_SERVER`、`RAG_REMOTE_ROOT`，然后执行：

```bash
cd /Users/syh/Reliable-RAG
mkdir -p results/selective_rag

scp -r "$RAG_SERVER:$RAG_REMOTE_ROOT/results/selective_rag/musique_valid" \
  results/selective_rag/
```

下载后即可在本地查看图片和表格。

## 11. 常见问题

| 问题 | 处理方法 |
| --- | --- |
| Mac 提示 `command not found: sbatch` | `sbatch` 在学校服务器执行，先 SSH 登录 |
| SSH 超时 | 按第 2 步检查 VPN、端口和服务器入口 |
| `No such file or directory` | 用 `pwd` 核对位置，检查是否上传了文件 |
| 找不到 `.venv313/bin/python` | 使用真实推理环境，设置 `AUTORATER_VENV` |
| `ModuleNotFoundError` | 在实际运行任务的 Python 环境中安装 requirements |
| 模型下载或鉴权失败 | 检查任务日志中的下载错误、缓存目录和模型访问权限 |
| 排队一直是 `PD` | 查看 `squeue` 的等待原因；这不等于代码卡住 |
| 提交时提示时限超限 | 按集群限制使用 `sbatch --time=...` 覆盖 |
| `unclassified rows` | 现有行为输入不完整；当前建议先使用完整 Mistral |
| `Frozen configuration differs` | 配置或输入变了，使用新的输出目录 |
| 采样 `empty/capped sample` | 检查日志；触顶时可在新输出目录增大 `--sample-max-new-tokens`，不要当作正常样本跳过 |
| `no silent context truncation` / 输入预算超限 | 检查提示长度和模型可支持的窗口，不能直接丢弃上下文后仍称为同一实验 |
| 只有置信度文件，没有曲线 | 小测试只采集置信度；正式任务也可能尚未进入训练/绘图阶段 |

## 12. 单独重做训练或绘图

置信度采集完整后，可以在服务器项目根目录运行 CPU 阶段，不需要再次加载 Mistral：

```bash
.venv313/bin/python src/selective_rag/selective_rag.py fit --model mistral
.venv313/bin/python src/selective_rag/selective_rag.py plot --model mistral
```

如果正式实验用了自定义 `--output-dir`，这里也必须传同一个目录。缺少置信度记录时，应恢复 GPU 采集。

方法细节、论文适配差异和评估限制见 `src/selective_rag/README.md`。只有真实模型任务成功完成后，才能将输出作为 baseline 实验结果写入报告。
