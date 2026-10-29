#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=nli-diagnostic
#SBATCH --output=nli_diagnostic_%j.out
#SBATCH --error=nli_diagnostic_%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch; do not run model diagnostics on the login node.' >&2
    exit 1
fi
project_root="${AUTORATER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
project_root="$(cd "$project_root" && pwd -P)"
while [[ ! -f "$project_root/src/autorater/diagnose_true_nli.py" || ! -d "$project_root/data/sufficiencybench_v1.1" ]]; do
    if [[ "$project_root" == / ]]; then
        echo 'Project not found; submit from the project directory or a subdirectory.' >&2
        exit 1
    fi
    project_root="$(dirname "$project_root")"
done
cd "$project_root"
venv_dir="${AUTORATER_VENV:-$project_root/.venv313}"
export HF_HOME="${HF_HOME:-$project_root/.model_cache}"
export TOKENIZERS_PARALLELISM=false
# FP32 reference runs on the allocated compute node CPU, with no GPU requested.
"$venv_dir/bin/python" -u src/autorater/diagnose_true_nli.py --root "$project_root" "$@"
