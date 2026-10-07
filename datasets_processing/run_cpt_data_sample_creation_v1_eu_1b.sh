#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=amd
#SBATCH --time=24:00:00
#SBATCH --job-name=cpt-v1-eu-1b
#SBATCH --output=cpt-v1-eu-1b-%j.out

# This preprocessing job does not use a GPU. The standard AMD compute
# partition provides sufficient memory for the 64 GB request.

set -euo pipefail

# Slurm may copy this script to /var/spool/slurm before executing it, so
# BASH_SOURCE then points at the spool copy rather than the repository. Prefer
# the directory from which `sbatch` was invoked. Outside Slurm, derive the
# repository from the script's real location.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
   [[ -f "${SLURM_SUBMIT_DIR}/datasets_processing/cpt_data_sample_creation_v1_eu_1b.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SOURCE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd -- "${SOURCE_SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${PROJECT_ROOT}/datasets_processing"
CONDA_ENV_NAME="legal-data-process"

# These locations can be overridden at submission time by exporting the
# corresponding environment variables with `sbatch --export=ALL,...`.
LEGAL_DATA_DIR="${LEGAL_DATA_DIR:-${PROJECT_ROOT}/datasets/Multi_Legal_Pile_Commercial}"
REPLAY_DATA_DIR="${REPLAY_DATA_DIR:-${PROJECT_ROOT}/datasets/SlimPajama-1B/row_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/datasets/CPT-V1-EU-1B}"

# `conda activate` is not initialized automatically in a Slurm batch shell.
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available in PATH" >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Hugging Face datasets creates large Arrow fingerprint files. Prefer the
# compute node's local temporary storage rather than the home filesystem.
JOB_TEMP_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_TEMP_DIR="${JOB_TEMP_BASE}/cpt-v1-eu-1b-${SLURM_JOB_ID:-manual}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

cd "${PROJECT_ROOT}"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not-active}"
echo "Legal input: ${LEGAL_DATA_DIR}"
echo "Replay input: ${REPLAY_DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Hugging Face cache: ${HF_DATASETS_CACHE}"
echo "Additional Python arguments: $*"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-none}" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/cpt_data_sample_creation_v1_eu_1b.py" ]]; then
    echo "CPT creation script is missing from ${SCRIPT_DIR}" >&2
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/src/text-dedup/src/text_dedup" ]]; then
    echo "text-dedup submodule is missing; run: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ! -d "${LEGAL_DATA_DIR}" ]]; then
    echo "Legal dataset directory is missing: ${LEGAL_DATA_DIR}" >&2
    exit 1
fi
if [[ ! -d "${REPLAY_DATA_DIR}" ]]; then
    echo "SlimPajama replay directory is missing: ${REPLAY_DATA_DIR}" >&2
    exit 1
fi

python --version
python - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required; found {sys.version}")

import datasets
import polars
import polars_grouper
import text_dedup
import legal_data_process

print("datasets:", datasets.__version__)
print("polars:", polars.__version__)
print("Runtime imports: OK")
PY

echo "Allocated resources:"
echo "  CPUs: ${SLURM_CPUS_PER_TASK:-4}"
echo "  Memory requested: ${SLURM_MEM_PER_NODE:-unknown} MB"
free -h || true
df -h "${PROJECT_ROOT}" "${JOB_TEMP_BASE}" || true

# Extra command-line arguments are passed through to the Python program. For
# example, continue from transferred preprocessing files with:
#   sbatch .../run_cpt_data_sample_creation_v1_eu_1b.sh --resume-from-preprocessed
srun python -u "${SCRIPT_DIR}/cpt_data_sample_creation_v1_eu_1b.py" \
    --legal-dir "${LEGAL_DATA_DIR}" \
    --replay-dir "${REPLAY_DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --replay-target-tokens 140000000 \
    --skip-perplexity \
    --near-dedup text-dedup \
    --dedup-threshold 0.5 \
    --dedup-processes "${SLURM_CPUS_PER_TASK:-4}" \
    --shard-target-tokens 200000000 \
    "$@"

if [[ ! -s "${OUTPUT_DIR}/manifest.json" ]]; then
    echo "CPT creation finished without producing manifest.json" >&2
    exit 1
fi
if [[ ! -s "${OUTPUT_DIR}/statistics.json" ]]; then
    echo "CPT creation finished without producing statistics.json" >&2
    exit 1
fi
if ! compgen -G "${OUTPUT_DIR}/cpt-v1-eu-1b-*.jsonl" >/dev/null; then
    echo "CPT creation finished without producing final JSONL shards" >&2
    exit 1
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "Final artifacts:"
du -sh "${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"/manifest.json "${OUTPUT_DIR}"/statistics.json
ls -lh "${OUTPUT_DIR}"/cpt-v1-eu-1b-*.jsonl

# The standard amd nodes have 256 GB RAM, so requesting 64 GB is appropriate. amd_serial
 #  is unnecessary for this four-process job.
 #
 #  Submit with:
 #
 #  sbatch datasets_processing/run_cpt_data_sample_creation_v1_eu_1b.sh
 #
 #  Or resume transferred preprocessing:
 #
#   sbatch datasets_processing/run_cpt_data_sample_creation_v1_eu_1b.sh \
#     --resume-from-preprocessed
