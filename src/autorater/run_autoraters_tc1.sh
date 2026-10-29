#!/bin/bash
# Submit through SLURM; never run this as `bash ...` on the login node.
# Account allocation verified with MyTCinfo on 2026-10-28:
# normal, UGGPU-TC1; maximum 20 CPUs, 1 GPU, 64G RAM, 6 hours.
# Command-line sbatch options can override the defaults below for smoke tests.
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=autoraters
#SBATCH --output=autoraters_%j.out
#SBATCH --error=autoraters_%j.err
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo 'Usage: sbatch run_autoraters_tc1.sh {qwen_0shot|qwen_1shot|true_nli} [options]' >&2
    exit 1
fi
autorater="$1"
shift
case "$autorater" in
    qwen_0shot|qwen_1shot|true_nli) ;;
    *) echo "Unknown autorater: $autorater" >&2; exit 1 ;;
esac
# Keep the positional selection unambiguous; pass all other Python flags through.
for option in "$@"; do
    case "$option" in
        --method|--method=*|--methods|--methods=*)
            echo 'Select the autorater only using the first positional argument.' >&2
            exit 1 ;;
    esac
done
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit this script with sbatch on an allocated compute node.' >&2
    exit 1
fi
# SLURM copies this script into its spool directory, so BASH_SOURCE cannot
# locate the uploaded project. Search upwards from the submission directory.
project_root="${AUTORATER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
project_root="$(cd "$project_root" && pwd -P)"
while [[ ! -f "$project_root/src/autorater/autorater_benchmark.py" || ! -d "$project_root/data/sufficiencybench_v1.1" ]]; do
    if [[ "$project_root" == / ]]; then
        echo 'Project not found. Submit from the project directory or a subdirectory, or set AUTORATER_PROJECT_ROOT.' >&2
        exit 1
    fi
    project_root="$(dirname "$project_root")"
done
cd "$project_root"
for data_file in train.json val.json test.json; do
    if [[ ! -f "$project_root/data/sufficiencybench_v1.1/$data_file" ]]; then
        echo "Missing dataset file: data/sufficiencybench_v1.1/$data_file" >&2
        exit 1
    fi
done
venv_dir="${AUTORATER_VENV:-$project_root/.venv313}"
venv_python="$venv_dir/bin/python"
if [[ ! -x "$venv_python" ]]; then
    echo "Virtual environment not found: $venv_python" >&2
    echo 'Create .venv313 on the server or set AUTORATER_VENV to an existing venv.' >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$project_root/.model_cache}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
echo "Autorater: $autorater; default scope: full validation split (528 contexts)"
"$venv_python" -u "$project_root/src/autorater/autorater_benchmark.py" --root "$project_root" --method "$autorater" "$@"
