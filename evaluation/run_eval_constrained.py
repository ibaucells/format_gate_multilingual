"""
Constrained vocabulary evaluation.

Instead of open-ended generation, the model's output vocabulary is restricted
to a small set of tokens: space, negative sign, integers (single- or
multi-digit, e.g. "5", "146", " 42"), and EOS.

Prompts are read directly from an existing results folder
(results/raw/base_and_instruct/<model>/<lang>/<variant>.jsonl)
via the `prompt_text` field, so no prompt re-generation is needed.

Only the constrained output is saved, but correctness is not assessed here.

Example:
python run_eval_constrained.py --results_dir results/raw/base_and_instruct/model --model /path/to/model --out_dir results_constrained --max_new_tokens 16
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import jinja2
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


def is_instruct_model(model_path: str) -> bool:
    """Infer whether a model should use chat formatting from its name/path."""
    model_name = Path(model_path).name.lower()
    return any(marker in model_name for marker in ("instruct", "-it", "_it"))


TEUKEN_CHAT_TEMPLATE_MAP = {
    "en": "EN", "es": "ES", "fr": "FR", "de": "DE",
    "ca": "EN", "eu": "EN", "ru": "EN", "zh": "EN", "ar": "EN",
}


# ---------------------------------------------------------------------------
# Constrained vocabulary
# ---------------------------------------------------------------------------

# Matches any token that decodes to an integer (single- or multi-digit),
# optionally preceded by a space and/or a minus sign.
# Examples: "5", " 5", "146", " 146", "-3", " -42"
_NUMBER_TOKEN_RE = re.compile(r"^ ?-?\d+$")

# Non-digit punctuation tokens that are always allowed.
_ALLOWED_PUNCT = {" ", "-", " -"}


def get_constrained_token_ids(tokenizer) -> List[int]:
    """
    Scan the full vocabulary and return every token ID whose decoded surface
    form matches _ALLOWED_PUNCT or _NUMBER_TOKEN_RE, plus the EOS token.

    tokenizer.decode() is used so that byte-level BPE representations
    (Ġ, ▁, Ċ …) are correctly mapped to actual characters before comparison.
    """
    allowed_ids: List[int] = []
    vocab: Dict[str, int] = tokenizer.get_vocab()

    for _token_str, token_id in vocab.items():
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded in _ALLOWED_PUNCT or _NUMBER_TOKEN_RE.match(decoded):
            allowed_ids.append(token_id)

    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in allowed_ids:
        allowed_ids.append(tokenizer.eos_token_id)

    return allowed_ids


class ConstrainedVocabProcessor(LogitsProcessor):
    """
    At every decoding step, set the logit of every token NOT in `allowed_ids`
    to -inf so that the model can only sample from the constrained set.
    The additive mask is built lazily on the first call (device unknown before then).
    """

    def __init__(self, allowed_ids: List[int]):
        self._allowed_ids = allowed_ids
        self._mask: Optional[torch.Tensor] = None

    def _build_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((vocab_size,), float("-inf"), device=device)
        for tid in self._allowed_ids:
            if tid < vocab_size:
                mask[tid] = 0.0
        return mask

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._mask is None or self._mask.device != scores.device:
            self._mask = self._build_mask(scores.shape[-1], scores.device)
        return scores + self._mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_model_inputs(tokenizer, user_prompt: str, lang: str, device: str,
                       model_path: str, system_prompt=None):
    """Apply chat template for instruct models; plain tokenisation for base models."""
    use_chat = is_instruct_model(model_path)

    if use_chat and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            return {"input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids, device=device)}

        except (ValueError, jinja2.exceptions.TemplateError) as e:
            msg = str(e)
            if "Roles must alternate" in msg or "multiple chat templates" in msg:
                merged = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
                template_name = TEUKEN_CHAT_TEMPLATE_MAP.get(lang, "EN")
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": merged}],
                    chat_template=template_name,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
                return {"input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids, device=device)}
            raise

    return tokenizer(user_prompt, return_tensors="pt").to(device)


def compute_finished(output_ids, eos_token_id, pad_token_id=None) -> Optional[bool]:
    if not output_ids or eos_token_id is None:
        return None
    ids = list(output_ids)
    if pad_token_id is not None:
        while ids and ids[-1] == pad_token_id:
            ids.pop()
    if not ids:
        return None
    return ids[-1] == eos_token_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Constrained vocabulary evaluation. "
            "Reads prompt_text from an existing results folder and re-runs "
            "generation with vocabulary restricted to: space, minus, integers (single- or multi-digit), EOS."
        )
    )
    ap.add_argument(
        "--results_dir", required=True,
        help=(
            "Path to a single model's results folder "
            "(e.g. results/raw/base_and_instruct/<model>). "
            "Expected structure: <results_dir>/<lang>/<variant>.jsonl"
        ),
    )
    ap.add_argument("--model", required=True, help="HF model name or local path to run")
    ap.add_argument("--out_dir", default="results_constrained", help="Output directory root")
    ap.add_argument(
        "--languages", default=None,
        help="Comma-separated languages to process (default: all subfolders found)",
    )
    ap.add_argument(
        "--variants", default=None,
        help="Comma-separated variants to process (default: all .jsonl files found)",
    )
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--print_allowed_tokens", action="store_true",
        help="Print allowed token IDs and their decoded strings, then exit",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model
    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    allowed_ids = get_constrained_token_ids(tokenizer)
    print(f"Constrained vocabulary: {len(allowed_ids)} token(s) allowed (including EOS)")

    if args.print_allowed_tokens:
        for tid in sorted(allowed_ids):
            decoded = tokenizer.decode(
                [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            eos_tag = " [EOS]" if tid == tokenizer.eos_token_id else ""
            print(f"  id={tid:6d}  repr={repr(decoded)}{eos_tag}")
        return

    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Discover source files
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"--results_dir not found: {results_dir}")

    lang_filter  = set(args.languages.split(",")) if args.languages else None
    var_filter   = set(args.variants.split(","))  if args.variants  else None

    source_files: List[tuple] = []  # (lang, variant, Path)
    for lang_dir in sorted(results_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if lang_filter and lang not in lang_filter:
            continue
        for jl in sorted(lang_dir.glob("*.jsonl")):
            variant = jl.stem
            if var_filter and variant not in var_filter:
                continue
            source_files.append((lang, variant, jl))

    if not source_files:
        raise RuntimeError(
            f"No matching .jsonl files found under {results_dir} "
            f"(languages={args.languages}, variants={args.variants})"
        )

    print(f"Found {len(source_files)} source file(s) to process")

    # Output tree mirrors the source tree but under out_dir/<model_slug>/
    model_slug = model_path.replace("/", "__")
    base_out = Path(args.out_dir) / model_slug
    ensure_dir(base_out)

    logits_processor = LogitsProcessorList([ConstrainedVocabProcessor(allowed_ids)])
    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
    )

    for lang, variant, src_path in source_files:
        records = load_jsonl(src_path)

        out_lang_dir = base_out / lang
        ensure_dir(out_lang_dir)
        out_path = out_lang_dir / f"{variant}.jsonl"

        # Skip if output already contains all records.
        if out_path.exists():
            existing_lines = sum(1 for _ in out_path.open(encoding="utf-8"))
            if existing_lines >= len(records):
                print(f"  {lang}/{variant}.jsonl  SKIP (already {existing_lines}/{len(records)} records)")
                continue

        print(f"  {lang}/{variant}.jsonl  ({len(records)} records) -> {out_path}")

        with open(out_path, "w", encoding="utf-8") as fout:
            for rec in records:
                prompt_text = rec["prompt_text"]

                inputs = build_model_inputs(
                    tokenizer,
                    user_prompt=prompt_text,
                    lang=lang,
                    device=device,
                    model_path=model_path,
                )
                prompt_len = inputs["input_ids"].shape[1]
                inputs.pop("token_type_ids", None)

                with torch.no_grad():
                    gen_ids = model.generate(**inputs, **gen_kwargs)

                output_ids  = gen_ids[0].tolist()[prompt_len:]
                output_text = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                finished = compute_finished(
                    output_ids, tokenizer.eos_token_id, tokenizer.pad_token_id
                )

                out_rec = {
                    # Original problem fields
                    "id":         rec.get("id"),
                    "type":       rec.get("type"),
                    "subtype":    rec.get("subtype"),
                    "expression": rec.get("expression"),
                    "answer":     rec.get("answer"),
                    "meta":       rec.get("meta"),

                    # Provenance
                    "source_model":   rec.get("model"),
                    "language":       lang,
                    "prompt_variant": variant,
                    "prompt_text":    prompt_text,

                    # Constrained model
                    "model": model_path,
                    "generation_kwargs": {
                        k: v for k, v in gen_kwargs.items()
                        if k != "logits_processor"  # not JSON-serialisable
                    },
                    "allowed_token_ids": allowed_ids,

                    # Token ids
                    "input_ids":  inputs["input_ids"][0].tolist(),
                    "output_ids": output_ids,

                    # Constrained output (correctness not assessed here)
                    "model_output_raw": output_text,
                    "finished":         finished,
                }

                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
