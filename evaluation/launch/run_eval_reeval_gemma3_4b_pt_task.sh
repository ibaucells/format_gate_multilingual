#!/bin/bash
# Worker invoked once per srun task (one task per GPU). Picks its (config, pair)
# from $SLURM_PROCID, pins itself to one GPU via $SLURM_LOCALID, and runs
# run_eval_reeval_gemma3_4b_pt.py for just that (config, pair).
set -euo pipefail

if [ -n "${VENV_ACTIVATE:-}" ]; then source "${VENV_ACTIVATE}"; fi

MODEL="$1"
REPO_ROOT="$2"

PAIRS=(
  "de:long_numeric"
  "ar:long_numeric"
  "zh:solve"
  "ca:answer"
  "es:determine"
  "empty:answer"
  "fake:answer"
)
# Greedy decoding is seed-invariant (checked with two seeds); sampling uses three seeds.
# The main-paper greedy run (seed 42) is results/raw/base_and_instruct/gemma-3-4b-pt.
CONFIG_NAMES=(seed123_greedy seed2024_greedy seed42_sampled_temp0.7 seed123_sampled_temp0.7 seed2024_sampled_temp0.7)
CONFIG_SEEDS=(123 2024 42 123 2024)
CONFIG_EXTRA=("" "" "--do_sample --temperature 0.7" "--do_sample --temperature 0.7" "--do_sample --temperature 0.7")

NUM_PAIRS=${#PAIRS[@]}
CONFIG_IDX=$(( SLURM_PROCID / NUM_PAIRS ))
PAIR_IDX=$(( SLURM_PROCID % NUM_PAIRS ))

if [ "$CONFIG_IDX" -ge "${#CONFIG_NAMES[@]}" ]; then
  echo "[task ${SLURM_PROCID}] no work assigned (only $((NUM_PAIRS * ${#CONFIG_NAMES[@]})) tasks needed) -- exiting."
  exit 0
fi

PAIR="${PAIRS[$PAIR_IDX]}"
SEED="${CONFIG_SEEDS[$CONFIG_IDX]}"
OUT_SUBDIR="${CONFIG_NAMES[$CONFIG_IDX]}"
EXTRA="${CONFIG_EXTRA[$CONFIG_IDX]}"

export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"

OUT_DIR="${REPO_ROOT}/results/raw/reeval_gemma-3-4b-pt/${OUT_SUBDIR}"

echo "[task ${SLURM_PROCID}] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} pair=${PAIR} seed=${SEED} extra='${EXTRA}'"

python "${REPO_ROOT}/evaluation/run_eval_reeval_gemma3_4b_pt.py" \
  --dataset "${REPO_ROOT}/data/simple_math_dataset.jsonl" \
  --model "$MODEL" \
  --out_dir "$OUT_DIR" \
  --max_new_tokens 512 \
  --few_shot 0 \
  --seed "$SEED" \
  --disable_thinking \
  --fewshot_jsonl "${REPO_ROOT}/data/simple_math_fewshot.jsonl" \
  --pairs "$PAIR" \
  $EXTRA
