"""
Variant of steering_vectors.py that applies the steering vector at the '=' token
position (during the prompt pre-fill step) instead of at the last token position.

Steering vectors are shared with steering_vectors.py (same .pt files in steering_outputs/).
Evaluation results go to steering_outputs_eq_pos/ to keep them separate.

Example usage:
python steering_vectors_eq_pos.py --models "gemma-3-4b-pt" --common_features "language=ca" --positive_features "prompt=answer" --negative_features "prompt=compute"
python steering_vectors_eq_pos.py --models "gemma-3-4b-pt" --positive_features "language=empty,prompt=answer" --negative_features "language=ca,prompt=compute"
python steering_vectors_eq_pos.py --models "gemma-3-4b-pt" --common_features "prompt=answer" --positive_features "language=ca" --negative_features "language=es"
python steering_vectors_eq_pos.py --models "gemma-3-4b-pt" --common_features "language=en,type=sum" --positive_features "prompt=answer" --negative_features "prompt=compute"
python steering_vectors_eq_pos.py --models "gemma-3-4b-it,gemma-3-4b-pt" --common_features "language=ca,prompt=determine"
python steering_vectors_eq_pos.py --models "Qwen3-0.6B,Qwen3-0.6B-Base" --common_features "language=ca,prompt=determine"
python steering_vectors_eq_pos.py --models "Llama-3.2-3B-Instruct,Meta-Llama-3.2-3B" --common_features "language=ca,prompt=determine"
python steering_vectors_eq_pos.py --models "gemma-3-4b-pt" --common_features "language=es, prompt=answer" --positive_features "type=eq" --negative_features "type=sum"
python steering_vectors_eq_pos.py --models "gemma-3-4b-pt" --common_features "language=es,prompt=answer" --positive_features "answer_type=ANSWER-C" --negative_features "answer_type=NO-ANSWER"
python steering_vectors_eq_pos.py --models "gemma-3-4b-pt" --common_features "language=es,prompt=answer" --positive_features "answer_type=ANSWER-C" --negative_features "answer_type=ANSWER-I"

answer_type values: ANSWER-C (score=1), NO-ANSWER (score=0, extractor in NO_ANSWER_EXTRACTORS), ANSWER-I (score=0, extractor not in NO_ANSWER_EXTRACTORS)
"""
import argparse
import os
import json
import functools
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import defaultdict

# Force unbuffered output (needed when stdout is piped, e.g. SLURM jobs)
print = functools.partial(print, flush=True)

from format_gate_multilingual.evaluation.extractor import (
    extract_high_confidence_only,
    _find_gold_label_token_in_output,
    _is_gold_label_spurious,
    _SPURIOUS_CHECK_EXTRACTORS,
)

NO_ANSWER_EXTRACTORS = {
    "empty",
    "no_digit_gate",
    "unfinished_operation_repetition",
    "prompt_repetition_drift",
    "unresolved_operation_loop",
    "other_repetition_drift",
}

def get_answer_type(score, extractor):
    """Return ANSWER-C, NO-ANSWER, or ANSWER-I for a single result record."""
    if score == 1:
        return "ANSWER-C"
    if extractor in NO_ANSWER_EXTRACTORS:
        return "NO-ANSWER"
    return "ANSWER-I"


def get_results_dic(base_results_dir):
    results = {}
    id_problem_type = {}
    id_problem_subtype = {}
    ids = set()
    languages = set()

    for name in os.listdir(base_results_dir):
        model_dir = os.path.join(base_results_dir, name)
        model_name = model_dir.split("__")[-1]
        results[model_name] = {}

        for language in os.listdir(model_dir):
            language_dir = os.path.join(model_dir, language)
            results[model_name][language] = {}
            languages.add(language)

            for prompt in os.listdir(language_dir):
                prompt_file = os.path.join(language_dir, prompt)
                prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]
                results[model_name][language][prompt_name] = {}

                with open(prompt_file) as f:
                    data = [json.loads(line) for line in f]
                    for instance in data:
                        id = instance["id"]
                        problem_type = instance.get("type", "")
                        problem_subtype = instance.get("subtype", "")

                        if "correct" in instance:
                            score = 1 if instance.get("correct") is True else 0
                        else:
                            contains_gold_label = instance.get("contains_gold_label")
                            if isinstance(contains_gold_label, bool):
                                score = 1 if contains_gold_label else 0
                            else:
                                cgl = str(contains_gold_label).strip().lower()
                                score = 0 if cgl in {"false", False, "yes_but_spurious"} else 1

                        extractor = instance.get("extractor", "")
                        results[model_name][language][prompt_name][id] = {"score": score,
                                                                        "extractor": extractor,
                                                                        "answer_type": get_answer_type(score, extractor),
                                                                        "expression": instance.get("expression", ""),
                                                                        "answer": instance.get("answer", ""),
                                                                        "extracted_answer": instance.get("extracted_answer", []),
                                                                        "model_output_raw": instance.get("model_output_raw", ""),
                                                                        "avg_nll_numeric": instance.get("avg_nll_numeric"),
                                                                        "avg_nll_output": instance.get("avg_nll_output"),
                                                                        "has_numeric_answer": instance.get("has_numeric_answer"),
                                                                        "prompt_text": instance.get("prompt_text", ""),
                                                                        "type": problem_type,
                                                                        "subtype": problem_subtype,
                                                                        "problem_type": problem_type
                                                                        }
                        id_problem_type[id] = problem_type
                        id_problem_subtype[id] = problem_subtype
                        ids.add(id)
    return results, ids, languages, id_problem_type, id_problem_subtype

def list_of_strings(arg):
    return arg.split(',')

def to_dict(arg):
    features = {}
    for raw_item in arg.split(','):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            features[item] = "true"
        else:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise argparse.ArgumentTypeError(
                    f"Invalid feature '{raw_item}'. Feature key cannot be empty."
                )
            features[key] = value
    return features

def set_args(parser):
    parser.add_argument("--models", type=list_of_strings, required=True)
    parser.add_argument("--common_features", type=to_dict, required=False, default={})
    parser.add_argument("--positive_features", type=to_dict, required=False, default={})
    parser.add_argument("--negative_features", type=to_dict, required=False, default={})
    parser.add_argument("--multiplier", type=int)
    parser.add_argument("--coeffs", type=lambda s: [float(x) for x in s.split(",")], default=[1],
                        help="Comma-separated list of steering coefficients. Example: '1,2,5,-1,-2'")
    parser.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=None,
                        help="Comma-separated list of layer indices to steer. If not set, uses every 3rd layer.")
    parser.add_argument("--eval_side", choices=["negative", "positive"], default="negative",
                        help="Which side to evaluate on. 'negative' (default): steer bad→good. 'positive': steer good→bad (use negative coeffs).")
    parser.add_argument("--eval_features", type=to_dict, required=False, default=None,
                        help="If set, build the test set from these features instead of the positive/negative contrast. "
                             "Example: 'language=ar,prompt=answer,answer_type=NO-ANSWER'. "
                             "The steering vector is still trained from positive/negative.")
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--allow_partial_test", action="store_true",
                        help="If set, use however many test candidates are available when fewer than --test_size exist, "
                             "instead of raising an error.")
    parser.add_argument(
        "--model_base_dir",
        default=os.environ.get("HF_MODELS", "models"),
        help="Directory containing Hugging Face model folders. Ignored when --models entries are full paths.",
    )
    parser.add_argument(
        "--results_dir",
        default=os.environ.get("RESULTS_DIR", "results/raw/base_and_instruct"),
        help="Path to extracted zero-shot/base results directory.",
    )
    parser.add_argument(
        "--fs_results_dir",
        default=os.environ.get("FS_RESULTS_DIR", "results/raw/fewshot"),
        help="Path to few-shot results dir. Used when 'few-shot' appears in positive/negative features.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("STEERING_EQ_OUTPUT_DIR", "steering_outputs_eq_pos"),
        help="Directory for eq-position steering evaluation JSONL outputs.",
    )
    parser.add_argument(
        "--vec_dir",
        default=os.environ.get("STEERING_VEC_DIR", "steering_outputs"),
        help="Directory for cached steering vector .pt files shared with steering_vectors.py.",
    )
    return parser

def load_model_and_tokenizer(model_name, model_base_dir):
    model_path = model_name if os.path.isabs(model_name) else os.path.join(model_base_dir, model_name)
    print(f"[Loading] Model: {model_name} from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    print(f"[Loaded]  Model '{model_name}' ready on device(s): {set(str(p.device) for p in model.parameters())}")
    return model, tokenizer

def summary_results_model(results_model):
    results_csv = []
    prompts = [p for p in results_model["en"]]
    for language in results_model:
        lang_scores_per_prompt = [language]
        for prompt in prompts:
            if prompt in results_model[language].keys():
                scores = [v["score"] for k,v in results_model[language][prompt].items()]
                try:
                    lang_scores_per_prompt.append(sum(scores)/len(scores))
                except Exception:
                    lang_scores_per_prompt.append("-")
            else:
                lang_scores_per_prompt.append("-")
        results_csv.append(lang_scores_per_prompt)
    return results_csv

def check_features(num_models, common_features, positive_features, negative_features):
    required = {"language", "prompt"}
    if num_models == 1:
        if not required.issubset(set(common_features) | set(positive_features)):
            missing = required - (set(common_features) | set(positive_features))
            raise ValueError(f"Incomplete (common+positive) features. Missing: {sorted(missing)}")
        if not required.issubset(set(common_features) | set(negative_features)):
            missing = required - (set(common_features) | set(negative_features))
            raise ValueError(f"Incomplete (common+negative) features. Missing: {sorted(missing)}")

    elif num_models == 2:
        if not required.issubset(set(common_features)):
            missing = required - set(common_features)
            raise ValueError(f"Incomplete common_features for 2 models. Missing: {sorted(missing)}")
    else:
        raise ValueError(f"Unsupported num_models={num_models}. Expected 1 or 2.")


_FS_FLAG = "few-shot"


def filter_examples_by_features(examples, features):
    ignored = {"language", "prompt", _FS_FLAG}
    record_filters = {k: v for k, v in features.items() if k not in ignored}
    if not record_filters:
        return examples

    filtered = {}
    for key, record in examples.items():
        keep = True
        for feature_name, feature_value in record_filters.items():
            if str(record.get(feature_name, "")) != str(feature_value):
                keep = False
                break
        if keep:
            filtered[key] = record
    return filtered


def build_eval_set(results, model, eval_features, eval_size, exclude_keys=None):
    lang = eval_features.get("language")
    prompt = eval_features.get("prompt")
    if not lang or not prompt:
        raise ValueError("--eval_features must include 'language' and 'prompt'.")
    try:
        examples = results[model][lang][prompt]
    except KeyError as e:
        raise KeyError(f"Missing results for model={model}, language={lang}, prompt={prompt}. KeyError: {e}")
    examples = filter_examples_by_features(examples, eval_features)
    exclude_keys = exclude_keys or set()
    items = []
    excluded = 0
    for key, rec in examples.items():
        if key in exclude_keys:
            excluded += 1
            continue
        items.append({
            "key": key,
            "prompt_text_negative": rec["prompt_text"],
            "prompt_text_positive": rec["prompt_text"],
            "neg_completion": rec.get("model_output_raw", ""),
            "pos_completion": rec.get("model_output_raw", ""),
            "gold_answer": rec.get("answer", ""),
            "type": rec.get("type", ""),
            "score_negative": rec.get("score"),
            "score_positive": rec.get("score"),
        })
    if len(items) < eval_size:
        print(f"[Dataset] WARNING: only {len(items)} eval examples available (requested {eval_size})")
    test = items[:eval_size]
    print(f"[Dataset] Eval set from eval_features ({eval_features}): {len(test)} examples")
    return test


def load_dataset(models, args, saved_test_keys=None):
    base_results_dir = args.results_dir
    print(f"[Dataset] Loading results from: {base_results_dir}")
    results, _, _, _, _ = get_results_dic(base_results_dir)
    print(f"[Dataset] Loaded results for {len(results)} model(s): {list(results.keys())}")

    pos_fs = _FS_FLAG in args.positive_features
    neg_fs = _FS_FLAG in args.negative_features
    fs_results = None
    if pos_fs or neg_fs:
        print(f"[Dataset] Loading few-shot results from: {args.fs_results_dir}")
        fs_results, _, _, _, _ = get_results_dic(args.fs_results_dir)
        print(f"[Dataset] Loaded few-shot results for {len(fs_results)} model(s): {list(fs_results.keys())}")

    num_models = len(models)
    print(f"[Dataset] Mode: {'1-model pos/neg' if num_models == 1 else '2-model comparison'} | Models: {models}")
    print(f"[Dataset] common={args.common_features}  positive={args.positive_features}  negative={args.negative_features}")
    check_features(num_models, args.common_features, args.positive_features, args.negative_features)

    def get_examples(model, language, prompt, use_fs=False):
        src = fs_results if use_fs else results
        try:
            return src[model][language][prompt]
        except KeyError as e:
            raise KeyError(f"Missing results for model={model}, language={language}, prompt={prompt}. KeyError: {e}")

    def split_no_leak(train_candidates, test_candidates):
        if len(train_candidates) < train_size:
            raise RuntimeError(f"Not enough train candidates: {len(train_candidates)} < {train_size}")

        train = train_candidates[:train_size]
        train_keys = {x["key"] for x in train}

        test_remaining = [x for x in test_candidates if x["key"] not in train_keys]
        if len(test_remaining) < test_size:
            if args.allow_partial_test:
                print(f"[Dataset] WARNING: only {len(test_remaining)} test candidates available (requested {test_size}) — using all of them.")
            else:
                raise RuntimeError(
                    f"Not enough test candidates after removing train overlap: {len(test_remaining)} < {test_size}. "
                    f"Use --allow_partial_test to proceed with fewer examples."
                )

        test = test_remaining[:test_size]
        reserve = test_remaining[test_size:]
        return train, test, reserve

    if num_models == 1:
        model = models[0]

        pos_lang = args.positive_features.get("language", args.common_features.get("language"))
        pos_prompt = args.positive_features.get("prompt", args.common_features.get("prompt"))
        neg_lang = args.negative_features.get("language", args.common_features.get("language"))
        neg_prompt = args.negative_features.get("prompt", args.common_features.get("prompt"))

        pos = get_examples(model, pos_lang, pos_prompt, use_fs=pos_fs)
        neg = get_examples(model, neg_lang, neg_prompt, use_fs=neg_fs)

        pos_features = dict(args.common_features)
        pos_features.update(args.positive_features)
        neg_features = dict(args.common_features)
        neg_features.update(args.negative_features)

        pos = filter_examples_by_features(pos, pos_features)
        neg = filter_examples_by_features(neg, neg_features)

        pos_map = {k: e for k,e in pos.items()}
        neg_map = {k: e for k,e in neg.items()}
        overlap = sorted(set(pos_map) & set(neg_map))
        random.Random(42).shuffle(overlap)
        print(f"[Dataset] Positive examples: {len(pos_map)}  Negative examples: {len(neg_map)}  Overlap: {len(overlap)}")
        if not overlap:
            raise RuntimeError("No overlap between positive and negative examples (by example_id/id/prompt_text).")

        train_candidates, test_candidates = [], []

        for k in overlap:
            p = pos_map[k]
            n = neg_map[k]

            item = {
                "key": k,
                "prompt_text_positive": p.get("prompt_text"),
                "prompt_text_negative": n.get("prompt_text"),
                "type": p.get("type", n.get("type")),
                "score_positive": p.get("score"),
                "score_negative": n.get("score"),
                "gold_answer": p.get("answer"),
                "pos_completion": p.get("model_output_raw"),
                "neg_completion": n.get("model_output_raw"),
                "pos_extracted_answer": p.get("extracted_answer"),
                "neg_extracted_answer": n.get("extracted_answer"),
            }

            if item["score_positive"] == 1 and item["score_negative"] == 0:
                train_candidates.append(item)
                test_candidates.append(item)

        train_size = args.train_size
        test_size = args.test_size

        print(f"[Dataset] Train candidates (pos=1, neg=0): {len(train_candidates)}  Test candidates: {len(test_candidates)}")
        train, test, reserve = split_no_leak(train_candidates, test_candidates)
        print(f"[Dataset] Final split -> train: {len(train)}  test: {len(test)}  reserve: {len(reserve)}")

        if args.eval_features is not None:
            test = build_eval_set(results, models[0], args.eval_features, args.test_size,
                                   exclude_keys={x["key"] for x in train})
            reserve = []
        elif saved_test_keys is not None:
            key_to_item = {item["key"]: item for item in test_candidates}
            test = [key_to_item[k] for k in saved_test_keys if k in key_to_item]
            print(f"[Dataset] Restored test set from saved keys: {len(test)} examples")
            reserve = []

        test_desc = (f"eval_features={args.eval_features}" if args.eval_features is not None
                     else f"language={neg_lang} prompt={neg_prompt} (same slice as negative)")
        print(f"[Dataset] === Summary ===")
        print(f"[Dataset]   Model: {model}")
        print(f"[Dataset]   Positive: language={pos_lang} prompt={pos_prompt} features={pos_features} -> {len(pos_map)} examples")
        print(f"[Dataset]   Negative: language={neg_lang} prompt={neg_prompt} features={neg_features} -> {len(neg_map)} examples")
        print(f"[Dataset]   Train: {len(train)} instances")
        print(f"[Dataset]   Test:  {len(test)} instances ({test_desc})")
        print(f"[Dataset]   Reserve: {len(reserve)} instances")

        return {"mode": "one_model_pos_vs_neg", "train": train, "test": test, "reserve": reserve}

    model_a, model_b = models[0], models[1]
    lang = args.common_features["language"]
    prompt = args.common_features["prompt"]

    a = get_examples(model_a, lang, prompt, use_fs=pos_fs)
    b = get_examples(model_b, lang, prompt, use_fs=neg_fs)

    pos_record_features = dict(args.common_features)
    pos_record_features.update(args.positive_features)
    neg_record_features = dict(args.common_features)
    neg_record_features.update(args.negative_features)
    a = filter_examples_by_features(a, pos_record_features)
    b = filter_examples_by_features(b, neg_record_features)

    a_map = {k: e for k,e in a.items()}
    b_map = {k: e for k,e in b.items()}

    overlap = sorted(set(a_map) & set(b_map))
    random.Random(42).shuffle(overlap)
    print(f"[Dataset] Model A examples: {len(a_map)}  Model B examples: {len(b_map)}  Overlap: {len(overlap)}")
    if not overlap:
        raise RuntimeError("No overlap between the two models' examples (by example_id/id/prompt_text).")

    train_candidates, test_candidates = [], []

    for k in overlap:
        ea = a_map[k]
        eb = b_map[k]

        item = {
            "key": k,
            "prompt_text_positive": ea.get("prompt_text"),
            "prompt_text_negative": eb.get("prompt_text"),
            "type": ea.get("type", eb.get("type")),
            "score_positive": ea.get("score"),
            "score_negative": eb.get("score"),
            "gold_answer": ea.get("answer"),
            "pos_completion": ea.get("model_output_raw"),
            "neg_completion": eb.get("model_output_raw"),
            "pos_extracted_answer": ea.get("extracted_answer"),
            "neg_extracted_answer": eb.get("extracted_answer"),
            }

        if item["score_positive"] == 1 and item["score_negative"] == 0:
            train_candidates.append(item)
            test_candidates.append(item)

    train_size = args.train_size
    test_size = args.test_size
    print(f"[Dataset] Train candidates (A=1, B=0): {len(train_candidates)}  Test candidates: {len(test_candidates)}")
    train, test, reserve = split_no_leak(train_candidates, test_candidates)
    print(f"[Dataset] Final split -> train: {len(train)}  test: {len(test)}  reserve: {len(reserve)}")

    if args.eval_features is not None:
        test = build_eval_set(results, model_a, args.eval_features, args.test_size,
                               exclude_keys={x["key"] for x in train})
        reserve = []
    elif saved_test_keys is not None:
        key_to_item = {item["key"]: item for item in test_candidates}
        test = [key_to_item[k] for k in saved_test_keys if k in key_to_item]
        print(f"[Dataset] Restored test set from saved keys: {len(test)} examples")
        reserve = []

    test_desc = (f"eval_features={args.eval_features}" if args.eval_features is not None
                 else f"language={lang} prompt={prompt} (same slice as model B / negative)")
    print(f"[Dataset] === Summary ===")
    print(f"[Dataset]   Model A (positive): {model_a}   Model B (negative): {model_b}")
    print(f"[Dataset]   language={lang} prompt={prompt}")
    print(f"[Dataset]   Positive features={pos_record_features} -> {len(a_map)} examples")
    print(f"[Dataset]   Negative features={neg_record_features} -> {len(b_map)} examples")
    print(f"[Dataset]   Train: {len(train)} instances")
    print(f"[Dataset]   Test:  {len(test)} instances ({test_desc})")
    print(f"[Dataset]   Reserve: {len(reserve)} instances")

    return {"mode": "two_models_compare", "train": train, "test": test, "reserve": reserve}

def to_inputs(tokenizer, model, text: str):
    return tokenizer(text, return_tensors="pt").to(model.device)

def get_hidden_states_from_inputs(model, inputs):
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states

def get_equal_token_index(tokenizer, input_ids):
    """
    Return index of the last '=' token in input_ids.
    Works best if '=' is its own token; otherwise falls back to matching decoded tokens.
    """
    eq_id = tokenizer.convert_tokens_to_ids("=")
    if eq_id is not None and eq_id != tokenizer.unk_token_id:
        eq_positions = (input_ids[0] == eq_id).nonzero(as_tuple=True)[0]
        if len(eq_positions) > 0:
            return int(eq_positions[-1].item())

    toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    eq_positions = [i for i, t in enumerate(toks) if "=" in t]
    if not eq_positions:
        preview = tokenizer.decode(input_ids[0], skip_special_tokens=True)[:80]
        print(f"[EqIdx] WARNING: no '=' token found in prompt ({input_ids.shape[1]} tokens); preview: {preview!r}")
        return None
    return eq_positions[-1]

def extract_hidden_at_index(hidden_states, token_index):
    per_layer = []
    for hs in hidden_states:
        v = hs[0, token_index, :]
        v = v.float().cpu()
        per_layer.append(v)
    return torch.stack(per_layer, dim=0)

def get_steering_vector(train_dataset, models_loaded):
    if len(models_loaded) == 1:
        model_pos, tokenizer_pos = models_loaded[0]
        model_neg, tokenizer_neg = models_loaded[0]
    elif len(models_loaded) == 2:
        model_pos, tokenizer_pos = models_loaded[0]
        model_neg, tokenizer_neg = models_loaded[1]
    else:
        raise ValueError("models_loaded must contain 1 or 2 models")

    pos_activations = defaultdict(list)
    neg_activations = defaultdict(list)

    n_train = len(train_dataset)
    print(f"[SteeringVec] Computing activations for {n_train} training examples...")
    for i, item in enumerate(train_dataset):
        p_text = item["prompt_text_positive"]
        n_text = item["prompt_text_negative"]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[SteeringVec]   example {i+1}/{n_train}")
        p_inputs = to_inputs(tokenizer_pos, model_pos, p_text)
        p_hs = get_hidden_states_from_inputs(model_pos, p_inputs)
        p_eq_idx = get_equal_token_index(tokenizer_pos, p_inputs.input_ids)
        if p_eq_idx is None:
            raise RuntimeError(
                f"[SteeringVec] No '=' token found in positive prompt for train example "
                f"{i} (key={item.get('key')}) — aborting."
            )
        p_v = extract_hidden_at_index(p_hs, p_eq_idx)
        for layer_num in range(p_v.shape[0]):
            pos_activations[layer_num].append(p_v[layer_num])

        n_inputs = to_inputs(tokenizer_neg, model_neg, n_text)
        n_hs = get_hidden_states_from_inputs(model_neg, n_inputs)
        n_eq_idx = get_equal_token_index(tokenizer_neg, n_inputs.input_ids)
        if n_eq_idx is None:
            raise RuntimeError(
                f"[SteeringVec] No '=' token found in negative prompt for train example "
                f"{i} (key={item.get('key')}) — aborting."
            )
        n_v = extract_hidden_at_index(n_hs, n_eq_idx)
        for layer_num in range(n_v.shape[0]):
            neg_activations[layer_num].append(n_v[layer_num])

    print(f"[SteeringVec] Averaging pos/neg activations across {n_train} examples per layer...")
    steering_vecs = defaultdict(list)
    for layer_num in pos_activations.keys():
        all_pos_layer = torch.stack(pos_activations[layer_num], dim=0)
        all_neg_layer = torch.stack(neg_activations[layer_num], dim=0)
        steering_vecs[layer_num] = (all_pos_layer - all_neg_layer).mean(dim=0)
    print(f"[SteeringVec] Done. Steering vectors computed for {len(steering_vecs)} layers.")
    return steering_vecs

def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print(f"[Hook] Found decoder layers at model.model.layers ({len(model.model.layers)} layers)")
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        print(f"[Hook] Found decoder layers at model.language_model.layers ({len(model.language_model.layers)} layers)")
        return model.language_model.layers
    print(f"[Hook] WARNING: could not find decoder layers. Model type: {type(model)}")
    return None

def make_caa_hook(vec, coeff, eq_idx):
    """
    Adds coeff*vec to the residual stream at the '=' token position.

    Only fires during the prompt pre-fill step (hidden.shape[1] > 1). During
    single-token generation steps the hook returns the output unchanged because
    the '=' token is already committed to the KV cache.

    If eq_idx is None (no '=' found in the prompt) the hook also skips.
    """
    fired = [False]

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
            is_tuple = True
        elif torch.is_tensor(output):
            hidden, rest = output, ()
            is_tuple = False
        else:
            print(f"[Hook] WARNING: unexpected output type {type(output)}, skipping steering")
            return output

        if hidden.ndim != 3:
            print(f"[Hook] WARNING: hidden has unexpected shape {hidden.shape}, skipping steering")
            return output

        # Only steer during pre-fill (full prompt present) and when eq_idx is valid
        if eq_idx is None or hidden.shape[1] == 1 or eq_idx >= hidden.shape[1]:
            if is_tuple:
                return (hidden,) + rest
            return hidden

        if not fired[0]:
            print(f"[Hook] Fired. output type={'tuple' if is_tuple else 'tensor'}, "
                  f"hidden shape={hidden.shape}, coeff={coeff}, eq_idx={eq_idx}")
            fired[0] = True

        v = vec.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[:, eq_idx, :] = hidden[:, eq_idx, :] + coeff * v

        if is_tuple:
            return (hidden,) + rest
        return hidden

    return hook

def register_caa_hooks(model, steering_vecs, coeff, eq_idx):
    """
    steering_vecs is keyed by hidden_states index: key k = hidden_states[k] = the
    residual stream *going into* decoder layer k (k=0 is the raw embedding output,
    before any decoder layer runs). A forward_hook on layers[i] fires *after* layer i
    runs, i.e. it sees hidden_states[i+1] — so to inject steering_vecs[k] at the point
    it was actually measured, hook layers[k-1], not layers[k].
    """
    layers = get_decoder_layers(model)
    if layers is None:
        print("[Hook] ERROR: no decoder layers found — no hooks registered, steering will have no effect")
        return []
    module_names = {id(m): name for name, m in model.named_modules()}
    handles = []
    for layer_idx, vec in steering_vecs.items():
        target_idx = layer_idx - 1
        if target_idx < 0:
            print(f"[Hook] WARNING: layer_idx=0 is the embedding output (before any decoder layer) — "
                  f"no decoder-layer hook point corresponds to it, skipping")
            continue
        if target_idx >= len(layers):
            print(f"[Hook] WARNING: layer_idx={layer_idx} out of range [1, {len(layers)}], skipping")
            continue
        layer = layers[target_idx]
        full_name = module_names.get(id(layer), f"<unknown>[{target_idx}]")
        print(f"[Hook] steering_vecs[{layer_idx}] (residual stream after decoder block {target_idx}) "
              f"-> hooking decoder block {target_idx}: {full_name} ({type(layer).__name__})")
        h = layer.register_forward_hook(make_caa_hook(vec, coeff, eq_idx))
        handles.append(h)
    print(f"[Hook] Registered {len(handles)} hook(s) for layers {list(steering_vecs.keys())} at eq_idx={eq_idx}")
    return handles

def generate_with_steering(model, tokenizer, prompt, steering_vecs, coeff, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eq_idx = get_equal_token_index(tokenizer, inputs["input_ids"]) if coeff != 0 else None
    handles = register_caa_hooks(model, steering_vecs, coeff, eq_idx) if coeff != 0 else []
    try:
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = gen_ids[0, prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text, inputs["input_ids"][0].tolist(), new_ids.tolist()
    finally:
        for h in handles:
            h.remove()


def eval_dataset(model, tokenizer, test, steering_vecs=None, coeff=0.0, eval_side="negative"):
    n_test = len(test)
    steered = steering_vecs is not None and coeff != 0.0
    prompt_key = f"prompt_text_{eval_side}"
    print(f"[Eval] Evaluating {n_test} examples | steered={steered} coeff={coeff} eval_side={eval_side}")

    active_vecs = steering_vecs if steered else {}
    active_coeff = coeff if steered else 0.0

    correct = 0
    outputs = []
    for i, doc in enumerate(test):
        prompt = doc[prompt_key]
        text, input_ids, output_ids = generate_with_steering(model, tokenizer, prompt, active_vecs, active_coeff)
        if (i + 1) % 10 == 0:
            print(f"[Eval] {i + 1}/{n_test} done")
        extracted_list, extractor, _ = extract_high_confidence_only(
            prompt_text=prompt,
            model_output_raw=text,
            expression="",
            prompt_variant="",
            problem_type=str(doc.get("type", "")),
            generation_ended_with_eos=None,
        )
        extracted_str = extracted_list[0] if extracted_list else ""

        gold_token = _find_gold_label_token_in_output(doc["gold_answer"], text)
        contains_gold = gold_token is not None
        if contains_gold and extractor in _SPURIOUS_CHECK_EXTRACTORS:
            if _is_gold_label_spurious(doc["gold_answer"], text):
                contains_gold = False
        is_correct = int(contains_gold)
        correct += is_correct

        outputs.append({
            "key": doc["key"],
            "prompt_text": prompt,
            "model_output_raw": text,
            "extracted_answer": extracted_str,
            "extractor": extractor,
            "answer": doc["gold_answer"],
            "correct": is_correct,
            "input_ids": input_ids,
            "output_ids": output_ids,
        })

    acc = correct / max(1, len(test))
    print(f"[Eval] Done. Accuracy: {acc:.4f} ({correct}/{n_test})")
    return acc, outputs

def replace_contaminated_baseline(model, tokenizer, test, base_outputs, reserve, eval_side, check_other_side=True):
    """
    After a baseline (coeff=0) run, swap out any test example that fails either check:
      - eval_side scored wrong (neg should be 0, pos should be 1)
      - other side scored wrong (only checked when check_other_side=True)
    Reserve candidates are accepted only if they pass the active check(s). If no clean
    replacement is found for a bad key, it is dropped rather than kept — every item in the
    returned test set satisfies the check(s), so the returned test set may be smaller than
    the requested size.
    Returns updated (test, base_outputs).

    check_other_side must be False in 2-model comparison mode: "other side" belongs to
    the model that was never used for generation here (main() always generates with
    model B), so requiring it to reproduce model A's recorded answer isn't a contamination
    check — model B is expected to differ from model A, that's the whole point of comparing
    them. Only the 1-model case (same model, different prompt condition) can reasonably be
    expected to reproduce its own recorded behavior.
    """
    other_side = "positive" if eval_side == "negative" else "negative"
    main_want  = 0 if eval_side == "negative" else 1
    other_want = 1 if eval_side == "negative" else 0

    main_bad = {o["key"] for o in base_outputs if o["correct"] != main_want}

    if check_other_side:
        _, other_outputs = eval_dataset(model, tokenizer, test, steering_vecs=None, coeff=0.0, eval_side=other_side)
        other_bad = {o["key"] for o in other_outputs if o["correct"] != other_want}
    else:
        other_bad = set()

    bad_keys = main_bad | other_bad

    if not bad_keys:
        if check_other_side:
            print(f"[Baseline] All examples pass both checks ({eval_side}={main_want}, {other_side}={other_want}).")
        else:
            print(f"[Baseline] All examples pass the {eval_side}={main_want} check "
                  f"(other-side check skipped: 2-model comparison).")
        return test, base_outputs

    if check_other_side:
        print(f"[Baseline] {len(bad_keys)} bad example(s): "
              f"{len(main_bad)} {eval_side}!={main_want}, {len(other_bad)} {other_side}!={other_want} "
              f"— searching reserve pool ({len(reserve)} candidates) for replacements...")
    else:
        print(f"[Baseline] {len(bad_keys)} bad example(s): {len(main_bad)} {eval_side}!={main_want} "
              f"(other-side check skipped: 2-model comparison) "
              f"— searching reserve pool ({len(reserve)} candidates) for replacements...")

    reserve_iter = iter(reserve)
    test_by_key  = {ex["key"]: ex for ex in test}
    main_out_by_key = {o["key"]: o for o in base_outputs}

    n_replaced = 0
    n_dropped = 0
    for bad_key in list(bad_keys):
        replaced = False
        for candidate in reserve_iter:
            _, cand_main = eval_dataset(model, tokenizer, [candidate], steering_vecs=None, coeff=0.0, eval_side=eval_side)
            other_ok = True
            if check_other_side:
                _, cand_other = eval_dataset(model, tokenizer, [candidate], steering_vecs=None, coeff=0.0, eval_side=other_side)
                other_ok = cand_other[0]["correct"] == other_want
            if cand_main[0]["correct"] == main_want and other_ok:
                print(f"[Baseline] Replacing key={bad_key} with key={candidate['key']}")
                test_by_key.pop(bad_key)
                main_out_by_key.pop(bad_key)
                test_by_key[candidate["key"]] = candidate
                main_out_by_key[candidate["key"]] = cand_main[0]
                replaced = True
                n_replaced += 1
                break
        if not replaced:
            print(f"[Baseline] WARNING: no clean replacement found for key={bad_key} — dropping it from the test set.")
            test_by_key.pop(bad_key, None)
            main_out_by_key.pop(bad_key, None)
            n_dropped += 1

    print(f"[Baseline] Done: {n_replaced} replaced, {n_dropped} dropped. Final test set size: {len(test_by_key)}.")
    return list(test_by_key.values()), list(main_out_by_key.values())


def make_output_path(args):
    def features_str(d):
        return "_".join(f"{k}-{v}" for k, v in sorted(d.items())) or "none"
    model_str = "+".join(args.models)
    eval_feat_str = features_str(args.eval_features) if args.eval_features else "same-as-neg"
    parts = [
        model_str,
        "common", features_str(args.common_features),
        "pos",    features_str(args.positive_features),
        "neg",    features_str(args.negative_features),
        "eval",   eval_feat_str,
        "side",   args.eval_side,
    ]
    name = "__".join(parts) + ".jsonl"
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def make_vec_path(args):
    """Steering vectors depend only on training data — shared with steering_vectors.py."""
    def features_str(d):
        return "_".join(f"{k}-{v}" for k, v in sorted(d.items())) or "none"
    model_str = "+".join(args.models)
    parts = [
        model_str,
        "common", features_str(args.common_features),
        "pos",    features_str(args.positive_features),
        "neg",    features_str(args.negative_features),
    ]
    name = "__".join(parts) + ".steering_vecs.pt"
    out_dir = args.vec_dir
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)

def save_outputs(path, outputs, layer, coeff):
    with open(path, "a") as f:
        for rec in outputs:
            row = dict(rec, layer=layer, coeff=coeff)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_done_pairs(output_path):
    pair_counts = defaultdict(int)
    if not os.path.exists(output_path):
        return pair_counts
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                pair_counts[(rec.get("layer"), rec.get("coeff"))] += 1
            except json.JSONDecodeError:
                pass
    return pair_counts


def load_saved_test_keys(output_path):
    if not os.path.exists(output_path):
        return None
    keys = []
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("coeff") == 0 and rec.get("layer") is None:
                    keys.append(rec["key"])
            except json.JSONDecodeError:
                pass
    return keys if keys else None


def main():
    parser = argparse.ArgumentParser()
    parser = set_args(parser)
    args = parser.parse_args()
    print(f"[Main] Args: {args}")

    output_path = make_output_path(args)
    vec_path = make_vec_path(args)
    print(f"[Main] Output path: {output_path}")
    print(f"[Main] Steering vec path: {vec_path}")

    pair_counts = load_done_pairs(output_path)
    if pair_counts:
        print(f"[Main] Found existing output with {sum(pair_counts.values())} records "
              f"across {len(pair_counts)} (layer, coeff) pair(s) — will resume.")

    saved_test_keys = load_saved_test_keys(output_path)
    if saved_test_keys:
        print(f"[Main] Found {len(saved_test_keys)} saved test keys — will reuse same test set.")

    print("\n[Main] === Step 1/4: Loading dataset ===")
    data_dic = load_dataset(args.models, args, saved_test_keys=saved_test_keys)

    n_test = len(data_dic["test"])
    done_pairs = {k for k, cnt in pair_counts.items() if n_test > 0 and cnt >= n_test}
    if done_pairs:
        baseline_done = (None, 0) in done_pairs
        layer_done = sorted((l, c) for l, c in done_pairs if l is not None)
        print(f"[Main] Already-complete (n_test={n_test}): baseline={baseline_done}, layers={layer_done}")

    print("\n[Main] === Step 2/4: Loading model(s) ===")
    models_loaded = {}
    for i, model_name in enumerate(args.models):
        model, tokenizer = load_model_and_tokenizer(model_name, args.model_base_dir)
        models_loaded[i] = (model, tokenizer)

    print("\n[Main] === Step 3/4: Computing steering vectors ===")
    if os.path.exists(vec_path):
        print(f"[Main] Loading steering vectors from {vec_path}")
        steering_vecs = torch.load(vec_path, weights_only=False)
    else:
        steering_vecs = get_steering_vector(data_dic["train"], models_loaded)
        torch.save(steering_vecs, vec_path)
        print(f"[Main] Steering vectors saved to {vec_path}")

    print("\n[Main] === Step 4/4: Evaluating with steering vectors ===")
    print(f"[Main] Saving outputs to: {output_path}")
    model, tokenizer = models_loaded[1] if 1 in models_loaded.keys() else models_loaded[0]

    eval_side = args.eval_side
    print(f"[Main] Eval side: {eval_side} ({'steer bad→good' if eval_side == 'negative' else 'steer good→bad'})")

    print("\n[Main] --- Baseline (no steering) ---")
    if (None, 0) in done_pairs:
        print("[Main] Baseline already complete, skipping.")
    else:
        base_acc, base_outputs = eval_dataset(model, tokenizer, data_dic["test"], steering_vecs=None, coeff=0.0, eval_side=eval_side)
        print(f"[Main] Baseline accuracy: {base_acc}")
        contaminated = (eval_side == "negative" and base_acc > 0) or (eval_side == "positive" and base_acc < 1.0)
        if contaminated:
            print(f"[Main] WARNING: baseline accuracy is {base_acc} (expected {'0' if eval_side == 'negative' else '1'} for eval_side={eval_side}) — attempting to replace contaminated examples...")
            data_dic["test"], base_outputs = replace_contaminated_baseline(
                model, tokenizer, data_dic["test"], base_outputs, data_dic.get("reserve", []), eval_side,
                check_other_side=(len(models_loaded) == 1 and args.eval_features is None),
            )
            base_acc = sum(o["correct"] for o in base_outputs) / max(1, len(base_outputs))
            print(f"[Main] Baseline accuracy after replacement: {base_acc}")
            if not data_dic["test"]:
                raise RuntimeError(
                    "[Main] All test examples were dropped as contaminated and no clean reserve "
                    "replacements were found — nothing left to evaluate. Aborting."
                )
        save_outputs(output_path, base_outputs, layer=None, coeff=0)

    coeffs = args.coeffs
    all_layers = list(steering_vecs.keys())
    layers_to_inspect = args.layers if args.layers is not None else list(range(0, len(all_layers), 3))
    print(f"[Main] Layers to inspect: {layers_to_inspect}  Coefficients: {coeffs}")
    for num_layer in layers_to_inspect:
        if num_layer not in steering_vecs or not isinstance(steering_vecs[num_layer], torch.Tensor):
            print(f"[Main] WARNING: layer {num_layer} not in steering_vecs (model has {len(steering_vecs)} layers: {sorted(steering_vecs.keys())}), skipping")
            continue
        for coeff in coeffs:
            if (num_layer, coeff) in done_pairs:
                print(f"[Main] Skipping layer={num_layer} coeff={coeff} (already complete, {n_test} records found)")
                continue
            one_layer_steering_vecs = {num_layer: steering_vecs[num_layer]}
            model, tokenizer = models_loaded[1] if 1 in models_loaded.keys() else models_loaded[0]
            print(f"\n[Main] --- steering_vecs[{num_layer}] (residual stream after decoder block {num_layer - 1}) "
                  f"| coeff={coeff} ---")
            acc, outputs = eval_dataset(model, tokenizer, data_dic["test"], steering_vecs=one_layer_steering_vecs, coeff=coeff, eval_side=eval_side)
            save_outputs(output_path, outputs, layer=num_layer, coeff=coeff)
            print("###############################")
            print(f"Layer: {num_layer} - Coeff: {coeff} - Acc: {acc}")
            print("###############################\n\n")
            print(f"[Main] INFO: layer={num_layer} coeff={coeff} acc={acc:.4f} — results saved to {output_path}")


if __name__ == "__main__":
    main()
