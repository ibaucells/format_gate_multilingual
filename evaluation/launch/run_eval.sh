#!/bin/bash
#SBATCH --job-name="eval_llm"
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes 1
#SBATCH -t 02:00:00
#SBATCH --qos REPLACE_WITH_QOS
#SBATCH --account REPLACE_WITH_ACCOUNT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${REPO_ROOT}/logs"

MODEL="$1"
LANGS="$2"
VARS="$3"

# source "${REPO_ROOT}/use_venv.sh"

# Example zero-shot evaluation of a base or instructed model
python "${REPO_ROOT}/evaluation/run_eval.py" \
  --dataset "${REPO_ROOT}/data/simple_math_dataset.jsonl" \
  --model "$MODEL" \
  --out_dir "${REPO_ROOT}/results" \
  --max_new_tokens 512 \
  --few_shot 0 \
  --languages "$LANGS" \
  --variants "$VARS" \
  --disable_thinking \
  --fewshot_jsonl "${REPO_ROOT}/data/simple_math_fewshot.jsonl"

# Example few-shot evaluation of a base or instructed model
python "${REPO_ROOT}/evaluation/run_eval.py" \
  --dataset "${REPO_ROOT}/data/simple_math_dataset.jsonl" \
  --model "$MODEL" \
  --out_dir "${REPO_ROOT}/results_fs" \
  --max_new_tokens 256 \
  --few_shot 5 \
  --languages "$LANGS" \
  --variants "$VARS" \
  --fewshot_jsonl "${REPO_ROOT}/data/simple_math_fewshot.jsonl"
