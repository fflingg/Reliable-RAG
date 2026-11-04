# MuSiQue dataset preparation

This stage follows the Sufficient Context paper's **MuSiQue-Ans dev, random 500 questions, original 20 candidate paragraphs, 6000-token context** setup. It substitutes the selected **Qwen3.8-27B, NF4/FP16, 1-shot** autorater, so this is a reproduction of the workflow rather than an exact replication of the paper's labels.

Source: https://github.com/StonyBrookNLP/musique (CC BY 4.0). The download helper uses the Google Drive archive ID in the official download_data.sh and extracts only musique_ans_v1.0_dev.jsonl, avoiding arbitrary ZIP extraction.

## Frozen choices

- Sample 500 unique questions with seed 42 after sorting source IDs; do not balance by the predicted label or answer correctness.
- Concatenate **all provided** title/text paragraphs in original array order (normally 20). Keep distractors; never select only is_supporting paragraphs. The actual 500-question sample has 498 records with 20 paragraphs, one with 19 and one with 18; preserve them without dropping or padding, and record the distribution.
- Truncate that text to at most 6000 Qwen tokenizer tokens, then verify the saved text's re-encoded length. Save exact contexts/hashes and truncated flags.
- Pin the actual Hugging Face model/tokenizer snapshot SHA during preparation; use the same SHA during labeling and on resume.
- Reuse the original evaluated 1-shot demonstration and instructions stored in rater_spec.json. Qwen only sees question/context, not answers, original answerable, supporting flags, or decomposition.
- No ground-truth sufficiency labels are available under this exact prompt/context setup. Invalid outputs remain null, are exported for manual review, and are never forced to insufficient or scored as wrong using unavailable truth.
- The original answer and aliases are retained in a separate data field for later answer evaluation. Original answerable=True is metadata, not the new sufficiency label.
- Keep these exact contexts when evaluating answer generators later; do not silently re-truncate for another model.

## Server use

Upload src/musique/ and retain src/autorater/ (the shared prompt/parser is imported). The runtime defaults to .venv313 and the existing project .model_cache.

From local WSL, at the actual project root:

```bash
scp -r src/musique n2602689b@10.96.189.11:~/sc4001_project/src/
```

On the server:

```bash
cd ~/sc4001_project
uv pip install --python .venv313/bin/python -r src/musique/requirements.txt
```

For minimal environment change if the autorater dependencies are already installed, install just the new download helper instead:

```bash
uv pip install --python .venv313/bin/python 'gdown>=5.2,<6'
```

Download the official archive (no model inference):

```bash
.venv313/bin/python src/musique/prepare_dataset.py download
```

If Google Drive is unavailable, download the official ZIP manually and provide it with `download --archive /absolute/path/musique_v1.0.zip`. You can also place the official dev JSONL directly at data/musique/raw/musique_ans_v1.0_dev.jsonl.

First submit a small, separately named 2-question smoke dataset:

```bash
sbatch --time=01:00:00 src/musique/run_musique_tc1.sh --count 2
```

After it completes successfully, submit the full dataset labeling job:

```bash
sbatch src/musique/run_musique_tc1.sh
```

These jobs run `prepare` then `label`; raw data is downloaded only if missing. Do not run GPU inference on the login node. Use one GPU job at a time. Re-submit the same command to resume; valid and invalid generated labels are retained, missing/runtime-error records are retried. Invalid labels need manual handling; deterministic repeated generation is not automatically retried as a quality fix.

Individual stages (prepare loads only tokenizer, label loads the model):

```bash
.venv313/bin/python src/musique/prepare_dataset.py prepare
.venv313/bin/python src/musique/prepare_dataset.py label  # only inside a GPU allocation
.venv313/bin/python src/musique/prepare_dataset.py export
```

## Outputs

```text
data/musique/raw/musique_ans_v1.0_dev.jsonl
data/musique/processed/dev500_seed42_ctx6000/
  prepared.jsonl          # fixed Q, context, answer/aliases, source metadata
  manifest.json          # source/context hashes, IDs, tokenizer/model commit, rater spec
  sample_ids.json
results/musique/dev500_seed42_ctx6000/qwen3.8_1shot/
  config.json
  environment.json
  labels.jsonl           # append-only checkpoint with raw outputs and status
  labelled_dataset.jsonl # merged rows; invalid/missing labels stay null
  summary.json           # valid/invalid/pending counts; sufficient ratio among valid
  human_review_blind.csv # random 50 cases plus any errors present on first export
  invalid_review.jsonl   # refreshed list of ALL current invalid/runtime-error cases
```

The 2-question smoke run uses dev2_seed42_ctx6000 directories and cannot overwrite the 500-question dataset. Export does not overwrite an existing human review CSV, preserving human edits. Use invalid_review.jsonl for additional invalid cases after a resumed job.

The blind audit sheet omits gold answers and autorater labels. Random-audit IDs are selected independently of model labels. Inspect a random 50–100 target cases and report agreement with human judgments; targeted invalid/boundary cases are additional checks, not an unbiased accuracy estimate. Do not force the class ratio to match the paper's 44.6% sufficient.

To retrieve the generated data and labels from local WSL:

```bash
scp -r n2602689b@10.96.189.11:~/sc4001_project/data/musique data/
scp -r n2602689b@10.96.189.11:~/sc4001_project/results/musique results/
```

This folder prepares labels only; generator answers and correct/error/abstain evaluation are the next experiment, not outputs of this step.

## Current local preparation (2026-11-03)

The official dev file has been downloaded and schema-checked: 2,417 questions. The frozen 500-question sample is already prepared locally. Saved contexts span 1,340–5,031 tokenizer tokens (median 2,603.5); none needed truncation at the 6,000-token cap. All original answerable flags are true; these are not the new sufficiency labels. Model/tokenizer commit: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Inspect [prepare_dataset.ipynb](prepare_dataset.ipynb) for actual preparation outputs. No Qwen labeling has run locally.

To upload these already prepared files without uploading the large source ZIP, run from the local WSL project root:

```bash
ssh n2602689b@10.96.189.11 'mkdir -p ~/sc4001_project/data/musique/raw'
scp -r src/musique n2602689b@10.96.189.11:~/sc4001_project/src/
scp data/musique/raw/musique_ans_v1.0_dev.jsonl data/musique/raw/download_manifest.json \
  n2602689b@10.96.189.11:~/sc4001_project/data/musique/raw/
scp -r data/musique/processed n2602689b@10.96.189.11:~/sc4001_project/data/musique/
```

Then `sbatch src/musique/run_musique_tc1.sh` reuses the prepared full sample and the existing model cache. A new smoke dataset with --count 2 still needs its own tokenizer preparation, but uses a separate output folder.
# 修复格式与单独重试无效输出

在服务器项目根目录执行（不需要重新上传 results）：

```bash
.venv313/bin/python src/musique/prepare_dataset.py repair
sbatch src/musique/run_musique_tc1.sh --retry-invalid --retry-max-new-tokens 512
```

`repair` 不加载模型，仅对完整 JSON 中 `sufficient` 字段名前后的空格做规范化。
原始记录不会删除，修复记录追加到 `labels.jsonl`，重新运行不会重复修复。
重试只选择最新状态为 `invalid` 的记录，不重跑已有有效标签。保持问题、上下文、
提示词、模型、量化和 greedy 解码不变，将输出预算从 64 提高到 512 tokens；
每条重试记录保存预算与 attempt 信息，原始输出保留在同一日志中。
此策略不能保证所有输出恢复正常；仍输出解释文字而非完整 JSON 的记录继续保留为空标签。
512-token 重试属于补救设置，分析时应区分首轮与重试结果。

