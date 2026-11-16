#!/bin/bash
#SBATCH --job-name="eval_gemma3_4b_pt_reeval"
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=9
#SBATCH --ntasks=35
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -t 02:00:00
#SBATCH --qos REPLACE_WITH_QOS
#SBATCH --account REPLACE_WITH_ACCOUNT

# Single job, 9 nodes x 4 GPUs = 36 GPU slots (35 used) across the 7
# (lang, variant) pairs x 5 configs (greedy seeds 123/2024, sampled@temp0.7
# seeds 42/123/2024) re-evaluation. One srun task per GPU; each task
# picks its own (config, pair) from $SLURM_PROCID -- see
# run_eval_reeval_gemma3_4b_pt_task.sh.

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${REPO_ROOT}/logs"

MODEL="$1"

# source "${REPO_ROOT}/use_venv.sh"

echo "=== $(date) | Launching ${SLURM_NTASKS} tasks across ${SLURM_NNODES} nodes ==="
srun --ntasks="${SLURM_NTASKS}" bash "${SCRIPT_DIR}/run_eval_reeval_gemma3_4b_pt_task.sh" "$MODEL" "$REPO_ROOT"
echo "=== $(date) | All tasks done ==="
