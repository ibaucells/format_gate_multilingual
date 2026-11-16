#!/bin/bash
# Reproduces every steering-vector experiment reported in the paper
# (Table 4, Figure 4, and appendix Tables 28-31).
#
# This is a driver script, not an sbatch job: run it directly and it submits
# one single-GPU job per (configuration, steering mode).
#
#   ACCOUNT=<account> QOS=<qos> HF_MODELS=/path/to/hf-models \
#       bash analysis/steering/run_steering_vectors.sh [eq|all|both] [gemma|llama|both]
#
# Steering modes:
#   eq   -> steering_vectors_eq_pos.py: vector added only at the "=" position
#           (Tables 28-29, solid lines in Figure 4). Outputs: steering_outputs_eq_pos/<job>/
#   all  -> steering_vectors.py: vector added at every generation step
#           (Tables 30-31, dashed line in Figure 4). Outputs: steering_outputs/<job>/
#
# Both modes share the steering-vector cache in steering_vecs/, so a vector is
# computed once (by whichever job runs first) and reused afterwards.
#
# Optional environment variables:
#   VENV_ACTIVATE  file to source before running Python (e.g. .venv/bin/activate)
#   TIME           Slurm time limit per job (default 08:00:00)
#   EXTRA_SBATCH   any additional sbatch flags (e.g. "--partition=acc")
set -euo pipefail

MODE="${1:-both}"
FAMILY="${2:-both}"

: "${ACCOUNT:?set ACCOUNT}"
: "${QOS:?set QOS}"
: "${HF_MODELS:?set HF_MODELS to the directory containing the HF model folders}"
TIME="${TIME:-08:00:00}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/steering steering_vecs steering_outputs steering_outputs_eq_pos

# Every layer of the residual stream after decoder blocks 0..N-1.
GEMMA_LAYERS="$(seq -s, 1 33)"
LLAMA_LAYERS="$(seq -s, 1 27)"

COMMON_ARGS=(
    --model_base_dir "${HF_MODELS}"
    --results_dir results/raw/base_and_instruct
    --fs_results_dir results/raw/fewshot
    --vec_dir steering_vecs
)

submit_one() {
    local script="$1" out_root="$2" name="$3"; shift 3
    local args
    args="$(printf '%q ' "${COMMON_ARGS[@]}" --output_dir "${out_root}/${name}" "$@")"
    sbatch --job-name="${name}" -D "${REPO_ROOT}" \
        --output="logs/steering/%x_%j.out" --error="logs/steering/%x_%j.err" \
        --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=20 -t "${TIME}" \
        --qos="${QOS}" --account="${ACCOUNT}" ${EXTRA_SBATCH} <<EOF
#!/bin/bash
set -euo pipefail
${VENV_ACTIVATE:+source "${VENV_ACTIVATE}"}
python -u analysis/steering/${script} ${args}
EOF
}

# submit <job-name> <args...>
submit() {
    local name="$1"; shift
    if [[ "${MODE}" == "eq" || "${MODE}" == "both" ]]; then
        submit_one steering_vectors_eq_pos.py steering_outputs_eq_pos "${name}" "$@"
    fi
    if [[ "${MODE}" == "all" || "${MODE}" == "both" ]]; then
        submit_one steering_vectors.py steering_outputs "${name}" "$@"
    fi
}

# Shorthands for the two contrast sets used by most experiments.
# Steering vector = mean(positive) - mean(negative), on 100 matched training ids.
POS_NEG_GEMMA_CA=(--common_features "prompt=answer" --positive_features "language=ca,answer_type=ANSWER-C" --negative_features "language=es,answer_type=NO-ANSWER")
POS_NEG_GEMMA_EN=(--common_features "prompt=answer" --positive_features "language=en,answer_type=ANSWER-C" --negative_features "language=es,answer_type=NO-ANSWER")
POS_NEG_LLAMA_EN=(--common_features "prompt=answer" --positive_features "language=en,answer_type=ANSWER-C" --negative_features "language=ru,answer_type=NO-ANSWER")
POS_NEG_LLAMA_ZH=(--common_features "prompt=answer" --positive_features "language=zh,answer_type=ANSWER-C" --negative_features "language=ru,answer_type=NO-ANSWER")

# ════════════════════════════════════════════════════════════════════════════
# Gemma-3-4b-pt  (Tables 28 and 30)
# ════════════════════════════════════════════════════════════════════════════
if [[ "${FAMILY}" == "gemma" || "${FAMILY}" == "both" ]]; then
    G=(--models "gemma-3-4b-pt" --layers "${GEMMA_LAYERS}")
    GIT=(--models "gemma-3-4b-it,gemma-3-4b-pt" --layers "${GEMMA_LAYERS}")

    # Base case: (ca,ans)-(es,ans) and (en,ans)-(es,ans), eval on (es,ans)
    submit gemma_base_ca-es          "${G[@]}" "${POS_NEG_GEMMA_CA[@]}" --eval_side negative --coeffs 1
    submit gemma_base_en-es          "${G[@]}" "${POS_NEG_GEMMA_EN[@]}" --eval_side negative --coeffs 1
    # Within language: (es,det)-(es,ans) and (ar,solve)-(ar,ans)
    submit gemma_within_esdet-esans  "${G[@]}" --common_features "language=es" --positive_features "prompt=determine,answer_type=ANSWER-C" --negative_features "prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit gemma_within_arsol-arans  "${G[@]}" --common_features "language=ar" --positive_features "prompt=solve,answer_type=ANSWER-C" --negative_features "prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    # Reverse direction: coeff -1 on already-correct positive-condition instances
    submit gemma_reverse_ca-es       "${G[@]}" "${POS_NEG_GEMMA_CA[@]}" --eval_side positive --coeffs -1
    submit gemma_reverse_en-es       "${G[@]}" "${POS_NEG_GEMMA_EN[@]}" --eval_side positive --coeffs -1
    # Sanity check: coeff +1 on already-correct positive-condition instances
    submit gemma_sanity_ca-es        "${G[@]}" "${POS_NEG_GEMMA_CA[@]}" --eval_side positive --coeffs 1
    submit gemma_sanity_en-es        "${G[@]}" "${POS_NEG_GEMMA_EN[@]}" --eval_side positive --coeffs 1
    # Prompt generalization: eval on (es,comp)
    submit gemma_promptgen_ca-es     "${G[@]}" "${POS_NEG_GEMMA_CA[@]}" --eval_features "language=es,prompt=compute,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit gemma_promptgen_en-es     "${G[@]}" "${POS_NEG_GEMMA_EN[@]}" --eval_features "language=es,prompt=compute,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    # Language generalization: eval on (ar,ans)
    submit gemma_langgen_ca-es       "${G[@]}" "${POS_NEG_GEMMA_CA[@]}" --eval_features "language=ar,prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit gemma_langgen_en-es       "${G[@]}" "${POS_NEG_GEMMA_EN[@]}" --eval_features "language=ar,prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    # FS vector: (fs,lang,ans)-(lang,ans)
    submit gemma_fs_es               "${G[@]}" --common_features "language=es,prompt=answer" --positive_features "few-shot,answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit gemma_fs_ar               "${G[@]}" --common_features "language=ar,prompt=answer" --positive_features "few-shot,answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    # IT vector: (it,lang,ans)-(pt,lang,ans), two-model comparison
    submit gemma_it_es               "${GIT[@]}" --common_features "language=es,prompt=answer" --positive_features "answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit gemma_it_ar               "${GIT[@]}" --common_features "language=ar,prompt=answer" --positive_features "answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
fi

# ════════════════════════════════════════════════════════════════════════════
# Meta-Llama-3.2-3B  (Tables 29 and 31)
# ════════════════════════════════════════════════════════════════════════════
if [[ "${FAMILY}" == "llama" || "${FAMILY}" == "both" ]]; then
    L=(--models "Meta-Llama-3.2-3B" --layers "${LLAMA_LAYERS}")
    LIT=(--models "Llama-3.2-3B-Instruct,Meta-Llama-3.2-3B" --layers "${LLAMA_LAYERS}")

    # Base case: (en,ans)-(ru,ans) and (zh,ans)-(ru,ans), eval on (ru,ans)
    submit llama_base_en-ru          "${L[@]}" "${POS_NEG_LLAMA_EN[@]}" --eval_side negative --coeffs 1
    submit llama_base_zh-ru          "${L[@]}" "${POS_NEG_LLAMA_ZH[@]}" --eval_side negative --coeffs 1
    # Within language: (ru,long_num)-(ru,ans) and (de,long_num)-(de,ans)
    # (fewer than 100 clean test candidates exist for these contrasts)
    submit llama_within_rulong-ruans "${L[@]}" --common_features "language=ru" --positive_features "prompt=long_numeric,answer_type=ANSWER-C" --negative_features "prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1 --allow_partial_test
    submit llama_within_delong-deans "${L[@]}" --common_features "language=de" --positive_features "prompt=long_numeric,answer_type=ANSWER-C" --negative_features "prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1 --test_size 78
    # Reverse direction
    submit llama_reverse_en-ru       "${L[@]}" "${POS_NEG_LLAMA_EN[@]}" --eval_side positive --coeffs -1
    submit llama_reverse_zh-ru       "${L[@]}" "${POS_NEG_LLAMA_ZH[@]}" --eval_side positive --coeffs -1
    # Sanity check
    submit llama_sanity_en-ru        "${L[@]}" "${POS_NEG_LLAMA_EN[@]}" --eval_side positive --coeffs 1
    submit llama_sanity_zh-ru        "${L[@]}" "${POS_NEG_LLAMA_ZH[@]}" --eval_side positive --coeffs 1
    # Prompt generalization: eval on (ru,solve)
    submit llama_promptgen_en-ru     "${L[@]}" "${POS_NEG_LLAMA_EN[@]}" --eval_features "language=ru,prompt=solve,answer_type=NO-ANSWER" --eval_side negative --coeffs 1 --allow_partial_test
    submit llama_promptgen_zh-ru     "${L[@]}" "${POS_NEG_LLAMA_ZH[@]}" --eval_features "language=ru,prompt=solve,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    # Language generalization: eval on (de,ans)
    submit llama_langgen_en-ru       "${L[@]}" "${POS_NEG_LLAMA_EN[@]}" --eval_features "language=de,prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit llama_langgen_zh-ru       "${L[@]}" "${POS_NEG_LLAMA_ZH[@]}" --eval_features "language=de,prompt=answer,answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    # FS vector
    submit llama_fs_ru               "${L[@]}" --common_features "language=ru,prompt=answer" --positive_features "few-shot,answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit llama_fs_de               "${L[@]}" --common_features "language=de,prompt=answer" --positive_features "few-shot,answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    # IT vector (two-model comparison)
    submit llama_it_ru               "${LIT[@]}" --common_features "language=ru,prompt=answer" --positive_features "answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
    submit llama_it_de               "${LIT[@]}" --common_features "language=de,prompt=answer" --positive_features "answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER" --eval_side negative --coeffs 1
fi
