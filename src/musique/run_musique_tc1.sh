#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=musique-label
#SBATCH --output=musique_%j.out
#SBATCH --error=musique_%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit using sbatch; do not run models on the login node.' >&2
    exit 1
fi
project_root="${AUTORATER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
project_root="$(cd "$project_root" && pwd -P)"
while [[ ! -f "$project_root/src/musique/prepare_dataset.py" || ! -d "$project_root/src/autorater" ]]; do
    if [[ "$project_root" == / ]]; then echo 'Project not found.' >&2; exit 1; fi
    project_root="$(dirname "$project_root")"
done
cd "$project_root"
venv_dir="${AUTORATER_VENV:-$project_root/.venv313}"
export HF_HOME="${HF_HOME:-$project_root/.model_cache}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
# "all" downloads only if raw data is missing, then freezes contexts and labels.
"$venv_dir/bin/python" -u src/musique/prepare_dataset.py all --root "$project_root" "$@"
