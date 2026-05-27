#!/bin/bash
#SBATCH --job-name="eval_constrained"
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -t 02:00:00
#SBATCH --qos=REPLACE_WITH_QOS
#SBATCH --account=REPLACE_WITH_ACCOUNT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Set these from the environment when you submit the job.
# Example:
#   HF_MODELS=/path/to/hf-models RESULTS_BASE=results/raw/base_and_instruct sbatch --account=YOUR_ACCOUNT run_eval_constrained.sh
HF_MODELS="${HF_MODELS:-/path/to/hf-models}"
RESULTS_BASE="${RESULTS_BASE:-${REPO_ROOT}/results/raw/base_and_instruct}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results_constrained}"

# source "${REPO_ROOT}/use_venv.sh"
mkdir -p "$OUT_DIR" "${REPO_ROOT}/logs"

# MODEL_NAME=gemma-3-4b-pt
MODEL_NAME=Qwen3-1.7B-Base
# MODEL_NAME=Meta-Llama-3.2-3B

MODEL_PATH="${HF_MODELS}/${MODEL_NAME}"
RESULTS_DIR="${RESULTS_BASE}/$(echo "$MODEL_PATH" | sed 's|/|__|g')"

echo "=== $(date) | Starting ${MODEL_NAME} ==="
python "${REPO_ROOT}/evaluation/run_eval_constrained.py" \
    --results_dir "$RESULTS_DIR" \
    --model "$MODEL_PATH" \
    --out_dir "$OUT_DIR" \
    --max_new_tokens 16
echo "=== $(date) | Done ${MODEL_NAME} ==="
