#!/bin/bash
#SBATCH --job-name="steering_vectors"
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -t 02:00:00
#SBATCH --qos=REPLACE_WITH_QOS
#SBATCH --account=REPLACE_WITH_ACCOUNT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# source "${REPO_ROOT}/use_venv.sh"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/steering_outputs" "${REPO_ROOT}/steering_outputs_eq_pos"

# Each run launches one srun step on its own node (backgrounded).
# Logs go to per-config files so outputs don't interleave.
run() {
    local name="$1"; shift
    local log="${REPO_ROOT}/logs/${name}_${SLURM_JOB_ID}"
    echo "=== $(date) | Launching ${name} ==="
    srun --nodes=1 --ntasks=1 --gres=gpu:1 \
        python -u "${REPO_ROOT}/analysis/steering/steering_vectors.py" \
        --output_dir "${REPO_ROOT}/steering_outputs" \
        --vec_dir "${REPO_ROOT}/steering_outputs" \
        "$@" \
        > "${log}.out" 2> "${log}.err" &
}

run_eq() {
    local name="$1"; shift
    local log="${REPO_ROOT}/logs/${name}_${SLURM_JOB_ID}"
    echo "=== $(date) | Launching ${name} (eq-pos) ==="
    srun --nodes=1 --ntasks=1 --gres=gpu:1 \
        python -u "${REPO_ROOT}/analysis/steering/steering_vectors_eq_pos.py" \
        --output_dir "${REPO_ROOT}/steering_outputs_eq_pos" \
        --vec_dir "${REPO_ROOT}/steering_outputs" \
        "$@" \
        > "${log}.out" 2> "${log}.err" &
}



# -----------------------------
# Example (all positions)
# -----------------------------
# These examples shows steering applied at all token positions.
LAYERS_GEMMA="2,6,10,14,16,18,20,22,24,26,28,30,32,34"
run "en_es_eval_es" \
    --models "gemma-3-4b-pt" \
    --common_features "prompt=answer" \
    --positive_features "language=en,answer_type=ANSWER-C" \
    --negative_features "language=es,answer_type=NO-ANSWER" \
    --eval_side negative \
    --coeffs "1" \
    --layers "$LAYERS_GEMMA"

run "ca_ar_eval_es" \
    --models "gemma-3-4b-pt" \
    --common_features "prompt=answer" \
    --positive_features "language=ca,answer_type=ANSWER-C" \
    --negative_features "language=ar,answer_type=NO-ANSWER" \
    --eval_features "language=es,prompt=answer,answer_type=NO-ANSWER" \
    --eval_side negative \
    --coeffs "1" \
    --layers "$LAYERS_GEMMA"

# -----------------------------
# Example (eq position)
# -----------------------------
# Steering applied at the '=' token position during pre-fill (eq-pos).
LAYERS_LLAMA="2,6,10,14,16,18,20,22,24,26,28"
run_eq "en_ru_eval_ru" \
    --models "Meta-Llama-3.2-3B" \
    --common_features "prompt=answer" \
    --positive_features "language=en,answer_type=ANSWER-C" \
    --negative_features "language=ru,answer_type=NO-ANSWER" \
    --eval_side negative \
    --coeffs "1" \
    --layers "$LAYERS_LLAMA"

run_eq "ca_ar_eval_es" \
    --models "gemma-3-4b-pt" \
    --common_features "prompt=answer" \
    --positive_features "language=ca,answer_type=ANSWER-C" \
    --negative_features "language=ar,answer_type=NO-ANSWER" \
    --eval_features "language=es,prompt=answer,answer_type=NO-ANSWER" \
    --eval_side negative \
    --coeffs "1" \
    --layers "$LAYERS_GEMMA"
