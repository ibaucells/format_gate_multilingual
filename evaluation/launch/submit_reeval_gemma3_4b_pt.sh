#!/bin/bash
# Submits ONE sbatch job spanning 9 nodes (36 GPUs, 35 used) that re-evaluates
# gemma-3-4b-pt 0-shot on 7 (lang, variant) pairs x 5 configs in parallel,
# one task per GPU (seed-robustness appendix experiment):
#   greedy decoding, seeds 123 and 2024
#   sampling (temperature=0.7, top_p=1), seeds 42, 123 and 2024
#
# Override defaults via environment, e.g.:
#   HF_MODELS=/path/to/hf-models QOS=<qos> ACCOUNT=<account> ./submit_reeval_gemma3_4b_pt.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run sbatch from here so $SLURM_SUBMIT_DIR (used by the job script to find REPO_ROOT) is correct.
cd "$SCRIPT_DIR"

: "${HF_MODELS:?set HF_MODELS}"
: "${QOS:?set QOS}"
: "${ACCOUNT:?set ACCOUNT}"

MODEL_NAME="gemma-3-4b-pt"
MODEL_PATH="${HF_MODELS}/${MODEL_NAME}"
JOB_SCRIPT="${SCRIPT_DIR}/run_eval_reeval_gemma3_4b_pt_job.sh"

sbatch --qos="$QOS" --account="$ACCOUNT" "$JOB_SCRIPT" "$MODEL_PATH"
