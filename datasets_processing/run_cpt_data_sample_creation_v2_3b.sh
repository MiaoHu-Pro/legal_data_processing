#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=amd
#SBATCH --time=24:00:00
#SBATCH --job-name=cpt-v2-legislation-3b
#SBATCH --output=cpt-v2-legislation-3b-%j.out

# CPT-V2: all legislation jurisdictions plus a deterministic 10% SlimPajama
# replay sample. --mem=0 requests all allocatable memory on one AMD node.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
   [[ -f "${SLURM_SUBMIT_DIR}/datasets_processing/cpt_data_sample_creation_v2_3b.py" ]]; then
    PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SOURCE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd -- "${SOURCE_SCRIPT_DIR}/.." && pwd)"
fi
PYTHON_SCRIPT="${PROJECT_ROOT}/datasets_processing/cpt_data_sample_creation_v2_3b.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/datasets/CPT-V2-Legislation-3B}"
CONDA_ENV_NAME="legal-data-process"

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
JOB_TEMP_DIR="${JOB_TEMP_BASE}/cpt-v2-legislation-3b-${SLURM_JOB_ID:-manual}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

cd "${PROJECT_ROOT}"

echo "Slurm job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not-active}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Hugging Face cache: ${HF_DATASETS_CACHE}"
echo "Additional Python arguments: $*"

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
import text_dedup

print("datasets:", datasets.__version__)
print("polars:", polars.__version__)
print("Runtime imports: OK")
PY

echo "Allocated resources:"
echo "  CPUs: ${SLURM_CPUS_PER_TASK:-4}"
echo "  Memory requested: all allocatable node memory"
free -h || true
df -h "${PROJECT_ROOT}" "${JOB_TEMP_BASE}" || true

srun python -u "${PYTHON_SCRIPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --skip-perplexity \
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
if ! compgen -G "${OUTPUT_DIR}/cpt-v2-legislation-3b-*.jsonl" >/dev/null; then
    echo "CPT creation finished without producing final JSONL shards" >&2
    exit 1
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "Final artifacts:"
du -sh "${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"/manifest.json "${OUTPUT_DIR}"/statistics.json
ls -lh "${OUTPUT_DIR}"/cpt-v2-legislation-3b-*.jsonl
