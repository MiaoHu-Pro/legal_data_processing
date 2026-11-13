#!/bin/bash
#SBATCH --job-name=odrl-sft-prepare
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=odrl-sft-prepare-%j.out
set -euo pipefail

# Submit from the legal_data_process root, or set DATA_PROCESS_ROOT explicitly.
DATA_PROCESS_ROOT="${DATA_PROCESS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
TRAINING_PROJECT_ROOT="${TRAINING_PROJECT_ROOT:-${HOME}/scratch/dips_project/lex-legal-post-training}"
ODRL_PREPARED_ROOT="${ODRL_PREPARED_ROOT:-${DATA_PROCESS_ROOT}/datasets/ODRL_SFT-processed-Qwen3}"
ODRL_TOKENIZER_PATH="${ODRL_TOKENIZER_PATH:-${HOME}/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lex_legal_rl_post_training_env}"

cd "${DATA_PROCESS_ROOT}"
for required in datasets_processing/prepare_odrl_sft.py "${TRAINING_PROJECT_ROOT}/src/training/sft_data.py" "${ODRL_TOKENIZER_PATH}/tokenizer_config.json"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Prefer the environment's C++ runtime to inherited older module libraries.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
unset _PYTHON_SYSCONFIGDATA_NAME

echo "Job: ${SLURM_JOB_ID:-manual}; node: $(hostname); started: $(date -Is)"
echo "Source project: ${DATA_PROCESS_ROOT}"
echo "Training helpers: ${TRAINING_PROJECT_ROOT}"
echo "Tokenizer: ${ODRL_TOKENIZER_PATH}"
echo "Default output: ${ODRL_PREPARED_ROOT}"
printf 'Additional preparation arguments: '; printf '%q ' "$@"; printf '\n'
python -c 'import pyarrow, datasets, transformers; print("Preparation imports: OK")'

# New output only: no overwrite/removal of existing datasets. Extra CLI options
# override defaults, e.g. --output-dir ... --target-format structured.
srun --cpu-bind=none python -u datasets_processing/prepare_odrl_sft.py \
    --input-dir "${DATA_PROCESS_ROOT}/datasets/SFT-datasets/odrl_data" \
    --output-dir "${ODRL_PREPARED_ROOT}" \
    --tokenizer-path "${ODRL_TOKENIZER_PATH}" \
    --training-project "${TRAINING_PROJECT_ROOT}" \
    --max-length 4096 "$@"
echo "Preparation finished: $(date -Is)"
