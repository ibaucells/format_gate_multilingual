import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import jinja2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
random.seed(42)


def is_instruct_model(model_path: str) -> bool:
    """Infer whether a model should use chat formatting from its name/path."""
    model_name = Path(model_path).name.lower()
    return any(marker in model_name for marker in ("instruct", "-it", "_it"))


# -------------------------
# Prompts (4 short + 1 long numeric)
# -------------------------
def fake_text(n):
    scripts = [
        (0x0041, 0x007A),  # Latin (A–Z, a–z)
        (0x0410, 0x044F),  # Cyrillic
        (0x0621, 0x064A),  # Arabic
        (0x4E00, 0x9FFF),  # CJK (Chinese)
    ]
    # Probabilities: Latin 6/9, others 1/9 each
    weights = [6, 1, 1, 1]
    out = []
    for _ in range(n):
        lo, hi = random.choices(scripts, weights=weights, k=1)[0]
        out.append(chr(random.randint(lo, hi)))
    return "".join(out)

PROMPTS: Dict[str, Dict[str, str]] = {
    "en": {
        "solve": "Solve",
        "compute": "Compute",
        "determine": "Determine",
        "answer": "Answer",
        "long_numeric": "Provide the numeric answer to the following operation",
    },
    "es": {
        "solve": "Resuelve",
        "compute": "Calcula",
        "determine": "Determina",
        "answer": "Responde",
        "long_numeric": "Proporciona la respuesta numérica de la siguiente operación",
    },
    "ca": {
        "solve": "Resol",
        "compute": "Calcula",
        "determine": "Determina",
        "answer": "Respon",
        "long_numeric": "Proporciona la resposta numèrica de la següent operació",
    },
    "fr": {
        "solve": "Résous",
        "compute": "Calcule",
        "determine": "Détermine",
        "answer": "Réponds",
        "long_numeric": "Donne la réponse numérique à l’opération suivante",
    },
    "de": {
        "solve": "Löse",
        "compute": "Berechne",
        "determine": "Bestimme",
        "answer": "Antworte",
        "long_numeric": "Gib die numerische Antwort für die folgende Operation an",
    },
    "eu": {
        "solve": "Ebatzi",
        "compute": "Kalkulatu",
        "determine": "Zehaztu",
        "answer": "Erantzun",
        "long_numeric": "Eman hurrengo eragiketaren emaitza zenbakizkoa",
    },
    "ru": {
        "solve": "Реши",
        "compute": "Вычисли",
        "determine": "Определи",
        "answer": "Ответь",
        "long_numeric": "Дай числовой ответ для следующей операции",
    },
    "zh": {
        "solve": "解",
        "compute": "计算",
        "determine": "求",
        "answer": "回答",
        "long_numeric": "给出下列运算的数值结果",
    },
    "ar": {
        "solve": "حل",
        "compute": "احسب",
        "determine": "حدد",
        "answer": "أجب",
        "long_numeric": "أعطِ الناتج العددي للعملية التالية",
    },
    "fake": {
        "solve": fake_text(2),
        "compute": fake_text(3),
        "determine": fake_text(4),
        "answer": fake_text(5),
        "long_numeric": fake_text(6),
    },
    "empty": {
        "answer": ""
    },
}

# Punctuation conventions
COLON_BY_LANGUAGE = {
    "zh": "：",   # fullwidth colon
    "fr": " :",   # French spacing before colon
}
DEFAULT_COLON = ":"

TEUKEN_CHAT_TEMPLATE_MAP = {
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "de": "DE",
    # fallbacks:
    "ca": "EN",
    "eu": "EN",
    "ru": "EN",
    "zh": "EN",
    "ar": "EN"
}

def load_fewshot_index(path: str) -> Dict[Tuple[str, str], List[dict]]:
    """
    Returns dict keyed by (type, subtype) -> list of exemplar records (dicts),
    in file order (deterministic).
    """
    exs = load_jsonl(path)
    idx: Dict[Tuple[str, str], List[dict]] = {}
    for rec in exs:
        key = (rec["type"], rec["subtype"])
        idx.setdefault(key, []).append(rec)
    return idx


def build_few_shot_prefix_from_index(
    fewshot_index: Dict[Tuple[str, str], List[dict]],
    k: int,
    lang: str,
    variant: str,
    match_type: str,
    match_subtype: str,
) -> str:
    if k <= 0:
        return ""

    key = (match_type, match_subtype)
    pool = fewshot_index.get(key, [])
    if not pool:
        raise ValueError(f"No few-shot exemplars found for key={key}")

    verb = PROMPTS[lang][variant]
    colon = COLON_BY_LANGUAGE.get(lang, DEFAULT_COLON)

    if lang == "empty":
        shots = [f"{rec['expression']} = {rec['answer']}" for rec in pool[:k]]
    else:
        shots = [
            f"{verb}{colon} {rec['expression']} = {rec['answer']}"
            for rec in pool[:k]
        ]

    if len(shots) < k:
        raise ValueError(
            f"Few-shot dataset has only {len(shots)} examples for key={key}, need {k}"
        )

    return "\n".join(shots) + "\n"


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def render_prompt(language: str, verb_phrase: str, expression: str) -> str:
    colon = COLON_BY_LANGUAGE.get(language, DEFAULT_COLON)
    # Keep prompt structure constant across languages: [instruction]{colon} {expr} =
    if language == "empty":
        prompt = f"{expression} ="
    else:
        prompt = f"{verb_phrase}{colon} {expression} ="
    return prompt


def build_model_inputs(tokenizer, user_prompt=None, lang=None, device=None, system_prompt=None, model_path=None, enable_thinking=True):
    # Decide if we SHOULD use chat formatting (not just if it exists)
    use_chat = is_instruct_model(model_path or "")

    # Only use chat template for instruct models (and only if tokenizer has one)
    if use_chat and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        chat_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
        if not enable_thinking:
            chat_kwargs["enable_thinking"] = False

        try:
            try:
                input_ids = tokenizer.apply_chat_template(messages, **chat_kwargs).to(device)
            except TypeError:
                # model does not support enable_thinking; retry without it
                chat_kwargs.pop("enable_thinking", None)
                input_ids = tokenizer.apply_chat_template(messages, **chat_kwargs).to(device)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, device=device)}

        except (ValueError, jinja2.exceptions.TemplateError) as e:
            msg = str(e)
            if "Roles must alternate" in msg or "multiple chat templates" in msg:
                merged = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
                messages2 = [{"role": "user", "content": merged}]
                template_name = TEUKEN_CHAT_TEMPLATE_MAP.get(lang, "EN")
                teuken_kwargs = dict(chat_template=template_name, add_generation_prompt=True, return_tensors="pt")
                if not enable_thinking:
                    teuken_kwargs["enable_thinking"] = False
                try:
                    input_ids = tokenizer.apply_chat_template(messages2, **teuken_kwargs).to(device)
                except TypeError:
                    teuken_kwargs.pop("enable_thinking", None)
                    input_ids = tokenizer.apply_chat_template(messages2, **teuken_kwargs).to(device)
                return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, device=device)}
            raise
    # Base models (or no template): plain tokenization
    return tokenizer(user_prompt, return_tensors="pt").to(device)


def ensure_padding(tokenizer):
    # Many decoder-only chat models have no pad token set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to input JSONL math dataset")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--out_dir", default="results", help="Output directory root")
    ap.add_argument("--languages", default="en,es,ca,fr,eu,de,ru,zh,ar", help="Comma-separated languages to run")
    ap.add_argument("--variants", default="solve,compute,determine,answer,long_numeric",
                    help="Comma-separated prompt variants to run")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--few_shot", type=int, default=0, choices=[0, 2, 5], help="Number of few-shot examples to prepend (0, 2, or 5)")
    ap.add_argument("--fewshot_jsonl", type=str, default=None, help="Path to fixed few-shot exemplars JSONL. If set, we draw few-shot examples from this file.")
    ap.add_argument("--disable_thinking", action="store_true", default=False,
                    help="Pass enable_thinking=False to apply_chat_template (for Qwen3-style models)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    # Ensure a pad token id exists for generate()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    data = load_jsonl(args.dataset)
    print(f"Loaded {len(data)} problems from {args.dataset}")

    fewshot_index = None
    if args.fewshot_jsonl:
        fewshot_index = load_fewshot_index(args.fewshot_jsonl)


    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    model_slug = model_path.replace("/", "__")
    base_out = Path(args.out_dir) / model_slug
    ensure_dir(base_out)

    # Generation config: greedy decoding
    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for lang in languages:
        if lang not in PROMPTS:
            raise ValueError(f"Unknown language '{lang}'. Available: {list(PROMPTS.keys())}")

        lang_dir = base_out / lang
        ensure_dir(lang_dir)

        for variant in variants:
            if variant not in PROMPTS[lang]:
                raise ValueError(f"Unknown variant '{variant}' for {lang}. Available: {list(PROMPTS[lang].keys())}")

            out_path = lang_dir / f"{variant}.jsonl"
            print(f"Writing: {out_path}")

            verb = PROMPTS[lang][variant]

            with open(out_path, "w", encoding="utf-8") as fout:
                for ex in data:
                    # Few-shot prefix (optional)
                    few_shot_prefix = ""
                    if args.few_shot > 0:
                        if fewshot_index is None:
                            raise ValueError("--few_shot > 0 requires --fewshot_jsonl")
                        few_shot_prefix = build_few_shot_prefix_from_index(
                            fewshot_index=fewshot_index,
                            k=args.few_shot,
                            lang=lang,
                            variant=variant,
                            match_type=ex["type"],
                            match_subtype=ex["subtype"],
                        )

                    prompt = few_shot_prefix + render_prompt(lang, verb, ex["expression"])
                    inputs = build_model_inputs(tokenizer,
                                       user_prompt=prompt,
                                       lang=lang,
                                       device=device,
                                       system_prompt=None,
                                       model_path=model_path,
                                       enable_thinking=not args.disable_thinking,
                                       )
                    
                    # inputs = build_model_inputs(
                    #     tokenizer,
                    #     user_prompt=prompt,
                    #     lang=lang,
                    #     system_prompt=None,
                    #     device=device
                    # )
                    prompt_len = inputs["input_ids"].shape[1]
                    inputs.pop("token_type_ids", None)
                    # 1) Generate output (greedy)
                    with torch.no_grad():
                        gen_ids = model.generate(**inputs, **gen_kwargs)

                    # Extract only newly generated ids
                    full_gen_ids = gen_ids[0].tolist()
                    output_ids = full_gen_ids[prompt_len:]
                    # Decode output in a stable way (no cleanup)
                    output_text_raw = tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    record = {
                        **ex,
                        "model": model_path,
                        "language": lang,
                        "prompt_variant": variant,
                        "prompt_text": prompt,
                        "generation_kwargs": gen_kwargs,
                        "input_ids": inputs["input_ids"][0].tolist(),
                        "output_ids": output_ids,
                        "model_output_raw": output_text_raw,
                        "few_shot": args.few_shot,
                    }

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
