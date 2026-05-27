#!/bin/bash
#SBATCH --job-name="reextract_hc"
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -t 02:00:00
#SBATCH --qos <QOS>
#SBATCH --account <ACCOUNT>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${REPO_ROOT}/logs"

# source "${REPO_ROOT}/use_venv.sh"

python "${REPO_ROOT}/evaluation/extractor.py" \
  --in-dir "${REPO_ROOT}/results" \
  --out-dir "${REPO_ROOT}/results/raw/base_and_instruct" \
  --num-workers ${SLURM_CPUS_PER_TASK:-32}
