# One autorater per SLURM job

Run from `~/sc4001_project`. The launcher defaults to `.venv313/bin/python`; override with `export AUTORATER_VENV=/absolute/path/to/venv` if needed.

Each submission must select exactly one autorater. By default it evaluates the full validation split: 264 questions / 528 contexts. All three use the same data, seed and train-only 1-shot demonstration. Metrics treat sufficient=1 as the positive class.

```bash
sbatch --job-name=qwen0 src/autorater/run_autoraters_tc1.sh qwen_0shot
sbatch --job-name=qwen1 src/autorater/run_autoraters_tc1.sh qwen_1shot
sbatch --job-name=truenli src/autorater/run_autoraters_tc1.sh true_nli
```

The account has MaxSubmit=2 and a total GPU allocation of 1. **Submit and finish one job at a time**; do not submit all three at once. The three lines above are separate alternatives to use in sequence. The defaults request normal QoS, UGGPU-TC1, 4 CPUs, 1 GPU, 32G memory and 4 hours. You can request up to your allowed 6 hours with `sbatch --time=06:00:00 ...`.

Outputs are isolated per method:

```text
results/autoraters/validation/
  qwen_0shot/   # config.json, metrics.csv/json, qwen_0shot.jsonl, environment/model metadata
  qwen_1shot/   # same structure for this method
  true_nli/    # same structure for this method
```

To resume, submit the same command. Successful predictions are retained. Do not run two jobs for the same method/output directory simultaneously. Existing combined smoke files are not deleted or mixed into these new directories.

## Invalid output policy

Empty output, whitespace-only output and other generated responses that cannot be parsed as a legal label are counted as incorrect judgments, and the next sample is processed. Their raw output is preserved; `prediction` remains null and `status` is `invalid` (no invented model label). For penalized Precision/Recall/F1, an invalid output with true label 1 contributes FN; one with true label 0 contributes FP. Accuracy uses the full processed sample count. Metrics also record `invalid_outputs`, `invalid_positive` and `invalid_negative`, so these penalties can be separated from actual 0/1 predictions. Report this scoring policy alongside the metrics.

Legacy parse-error checkpoints with generated output are recognized under the same policy when resuming. Already processed valid and invalid outputs are skipped; missing samples continue. Model-loading failures, OOM and over-budget inputs still stop as runtime errors and are not counted as model judgments. Final metrics remain withheld until all samples have been processed.

Optional smoke test of one method only:

```bash
sbatch --time=00:30:00 src/autorater/run_autoraters_tc1.sh qwen_0shot \
  --limit-questions 2 --output results/autoraters/smoke_qwen0
```

Direct Python invocation inside an allocated GPU compute job:

```bash
.venv313/bin/python src/autorater/autorater_benchmark.py --method true_nli
```

There is no multi-method default anymore; `--method` is required. `--split test` selects the full held-out test split only after the protocol is frozen. `--limit-questions` is for explicitly requested smoke runs.

Download `results/autoraters/validation/` to the same local path and run the final `src/parta.ipynb` cell for the combined comparison. No model inference was performed as part of this code change.
