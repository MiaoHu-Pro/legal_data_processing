#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=amd
#SBATCH --time=24:00:00
#SBATCH --job-name=cpt-proportional
#SBATCH --output=cpt-proportional-%j.out

# Post-dedup proportional CPT creation. --mem=0 requests all allocatable
# memory on one AMD node (about 256 GB). This job does not use a GPU.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
   [[ -f "${SLURM_SUBMIT_DIR}/datasets_processing/cpt_data_sample_creation_proportional.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SOURCE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd -- "${SOURCE_SCRIPT_DIR}/.." && pwd)"
fi
PYTHON_SCRIPT="${PROJECT_ROOT}/datasets_processing/cpt_data_sample_creation_proportional.py"
CONDA_ENV_NAME="legal-data-process"

HAS_TARGET=false
for argument in "$@"; do
    if [[ "${argument}" == "--total-target-tokens" || "${argument}" == --total-target-tokens=* ]]; then
        HAS_TARGET=true
        break
    fi
done
if [[ "${HAS_TARGET}" != true ]]; then
    echo "Missing required argument: --total-target-tokens" >&2
    echo "Example: sbatch $0 --total-target-tokens 5000000000" >&2
    exit 2
fi

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

JOB_TEMP_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_TEMP_DIR="${JOB_TEMP_BASE}/cpt-proportional-${SLURM_JOB_ID:-manual}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

cd "${PROJECT_ROOT}"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not-active}"
echo "Hugging Face cache: ${HF_DATASETS_CACHE}"
echo "Proportional CPT arguments: $*"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-none}" >&2
    exit 1
fi
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "CPT creation script is missing: ${PYTHON_SCRIPT}" >&2
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/src/text-dedup/src/text_dedup" ]]; then
    echo "text-dedup submodule is missing; run: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/datasets/Multi_Legal_Pile_Commercial" ]]; then
    echo "Legal dataset directory is missing" >&2
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/datasets/SlimPajama-1B/row_data" ]]; then
    echo "SlimPajama replay directory is missing" >&2
    exit 1
fi

python --version
python - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required; found {sys.version}")

import datasets
import legal_data_process
import polars
import polars_grouper
import pyarrow
import text_dedup

print("datasets:", datasets.__version__)
print("polars:", polars.__version__)
print("pyarrow:", pyarrow.__version__)
print("Runtime imports: OK")
PY

echo "Allocated resources:"
echo "  CPUs: ${SLURM_CPUS_PER_TASK:-4}"
echo "  Memory requested: all allocatable node memory"
free -h || true
df -h "${PROJECT_ROOT}" "${JOB_TEMP_BASE}" || true

# KenLM perplexity filtering remains intentionally deferred. Other arguments,
# including the total target and optional output naming, pass through verbatim.
srun python -u "${PYTHON_SCRIPT}" \
    --legal-share 0.9 \
    --candidate-oversample-factor 1.5 \
    --skip-perplexity \
    --dedup-processes "${SLURM_CPUS_PER_TASK:-4}" \
    --shard-target-tokens 200000000 \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"

# Created the new standalone Slurm launcher:
  #
  #  datasets_processing/run_cpt_data_sample_creation_proportional.sh
  #
  #  The existing run_cpt_data_sample_creation.sh remains unchanged.
  #
  #  Submit a 5B dataset:
  #
  #  cd /home/mh1f25/scratch/legal_data_process
  #
  #  sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh \
  #    --total-target-tokens 5000000000
  #
  #  This uses:
  #
  #  - 4.5B post-dedup legal tokens
  #  - 0.5B post-dedup SlimPajama tokens
  #  - 1.5× candidate oversampling
  #  - Four CPU workers
  #  - Entire AMD node memory
  #  - 200M-token output shards
  #  - No GPU
  #  - Perplexity filtering skipped
  #
  #  To specify the output:
  #
  #  sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh \
  #    --total-target-tokens 5000000000 \
  #    --dataset-name CPT-Proportional-5B \
  #    --output-prefix cpt-proportional-5b \
  #    --output-dir datasets/CPT-Proportional-5B
  #
  #  Safe resume after completed preprocessing:
  #
  #  sbatch datasets_processing/run_cpt_data_sample_creation_proportional.sh \
  #    --total-target-tokens 5000000000 \
  #    --resume-from-preprocessed
  #
  #  The log filename will be:
  #
  #  cpt-proportional-<job-id>.out