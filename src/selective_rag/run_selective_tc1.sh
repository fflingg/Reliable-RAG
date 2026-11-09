#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=selective-rag
#SBATCH --output=selective_%j.out
#SBATCH --error=selective_%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch; inference requires an allocated GPU.' >&2
    exit 1
fi
model="${1:?First argument: qwen38, llama31, or mistral}"
shift
case "$model" in qwen38|llama31|mistral) ;; *) echo 'Unknown model' >&2; exit 1 ;; esac
project_root="${AUTORATER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
project_root="$(cd "$project_root" && pwd -P)"
while [[ ! -f "$project_root/src/selective_rag/selective_rag.py" ]]; do
    if [[ "$project_root" == / ]]; then echo 'Project not found' >&2; exit 1; fi
    project_root="$(dirname "$project_root")"
done
cd "$project_root"
venv_dir="${AUTORATER_VENV:-$project_root/.venv313}"
export HF_HOME="${HF_HOME:-$project_root/.model_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
nvidia-smi
stage="${SELECTIVE_STAGE:-run}"
case "$stage" in run|confidence) ;; *) echo 'SELECTIVE_STAGE must be run or confidence' >&2; exit 1 ;; esac
"$venv_dir/bin/python" -u src/selective_rag/selective_rag.py "$stage" --model "$model" "$@"
