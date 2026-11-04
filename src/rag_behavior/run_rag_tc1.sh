#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=rag-behavior
#SBATCH --output=rag_%j.out
#SBATCH --error=rag_%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch; GPU inference must not run on the login node.' >&2
    exit 1
fi
model="${1:?First argument: qwen38, llama31, or mistral}"
shift
case "$model" in qwen38|llama31|mistral) ;; *) echo 'Unknown model' >&2; exit 1 ;; esac
project_root="${AUTORATER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
project_root="$(cd "$project_root" && pwd -P)"
while [[ ! -f "$project_root/src/rag_behavior/rag_behavior.py" ]]; do
    if [[ "$project_root" == / ]]; then echo 'Project not found' >&2; exit 1; fi
    project_root="$(dirname "$project_root")"
done
cd "$project_root"
venv_dir="${AUTORATER_VENV:-$project_root/.venv313}"
export HF_HOME="${HF_HOME:-$project_root/.model_cache}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
"$venv_dir/bin/python" -u src/rag_behavior/rag_behavior.py run --root "$project_root" --model "$model" "$@"
