#!/usr/bin/env python3
"""
Logit-lens analysis: for each transformer layer, project the hidden state
at the last '=' token through the final LayerNorm + LM head, and record
the top-K most probable tokens.

Output: CSV with one row per item.
Columns: item_id, gold, target_pos,
         L{i}_tok{r}, L{i}_prob{r}  for each layer i and rank r=1..top_k.

Reads input_ids directly from the JSONL records produced by the evaluation
pipeline (results_dir / model_folder / lang / prompt.jsonl).

Usage:
  python logit_lens.py \
    --model-path /path/to/Qwen3-0.6B-Base \
    --results-dir ../results/raw/base_and_instruct \
    --lang en \
    --prompt answer \
    --output logit_lens.csv \
    [--item-id math_001]          # specific item; default: all items
    [--n-items 3]                 # limit to first N items
    [--top-k 5]                   # tokens to record per layer (default: 5)
    [--layers 0,5,10,15,20]       # specific layers; default: all
    [--append-space]              # probe space after '=' instead of '=' itself

Options:
--model-path: Path to the HuggingFace model directory.
--results-dir: Directory containing the results (JSONL files).
--lang: Language code (e.g., en).
--prompt: Prompt variant (e.g., answer).
--item-id: Specific item ID to process (optional).
--n-items: Limit to the first N items (default: all items).
--top-k: Number of top tokens to record per layer (default: 5).
--layers: Comma-separated layer indices (default: all layers).
--append: String to append after the = token before probing.
--last-layer-only: Analyze only the final layer (faster, narrower output).
--output: Path to the output CSV file.

Usage examples:

# All items, all layers, top-5 tokens → CSV
python logit_lens.py \
  --model-path /path/to/hf-models/Qwen3-0.6B-Base \
  --results-dir ../results/raw/base_and_instruct \
  --lang en --prompt answer \
  --output logit_lens_en_answer.csv

# Specific item, all layers
python logit_lens.py \
  --model-path /path/to/hf-models/gemma-3-4b-pt \
  --results-dir results/raw/base_and_instruct \
  --lang ca --prompt answer \
  --item-id mix_0068 \
  --output logit_lens_mix0068.csv

# Subset: first 50 items, selected layers, top-3 tokens
python logit_lens.py \
  --model-path /path/to/hf-models/Qwen3-0.6B-Base \
  --results-dir ../results/raw/base_and_instruct \
  --lang ar --prompt long_numeric \
  --n-items 50 --top-k 3 --layers 0,5,10,15,20,25

python logit_lens.py \
  --model-path /path/to/hf-models/gemma-3-4b-pt \
  --results-dir  results/raw/base_and_instruct \
  --lang ar \
  --prompt answer \
  --top-k 1 \
  --output logit_lens/gemma-3-4b-pt_ar_answer_top1.csv

python logit_lens.py \
  --model-path /path/to/hf-models/Meta-Llama-3.2-3B \
  --results-dir  results/raw/base_and_instruct \
  --lang ar \
  --prompt answer \
  --top-k 1 \
  --output logit_lens/Meta-Llama-3.2-3B_ar_answer_top1.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Helpers shared with compute_cosine_similarities.py
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> dict:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def find_model_dir(results_dir: Path, model_name: str) -> Path:
    candidates = [d for d in results_dir.iterdir()
                  if d.is_dir() and model_name in d.name]
    if not candidates:
        raise FileNotFoundError(
            f"No folder matching '{model_name}' in {results_dir}"
        )
    return candidates[0]


def get_equal_token_index(tokenizer, input_ids: torch.Tensor) -> Optional[int]:
    """Return index of the last '=' token in input_ids (1-D or 2-D batch=1)."""
    ids_1d = input_ids[0] if input_ids.dim() == 2 else input_ids
    eq_id = tokenizer.convert_tokens_to_ids("=")
    if eq_id is not None and eq_id != tokenizer.unk_token_id:
        positions = (ids_1d == eq_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return int(positions[-1].item())
    toks = tokenizer.convert_ids_to_tokens(ids_1d.tolist())
    positions = [i for i, t in enumerate(toks) if t and "=" in t]
    return positions[-1] if positions else None


# ---------------------------------------------------------------------------
# Logit-lens
# ---------------------------------------------------------------------------

def get_final_norm_and_head(model):
    """
    Return (final_layernorm, lm_head) from a HuggingFace CausalLM.
    Works for Llama / Qwen / Gemma / Mistral / Gemma3 style models.

    Strategy:
    1. Try common attribute names directly on model.model.
    2. Walk model.named_modules() to find the last norm-like module
       whose FULL path contains no digit (excludes per-layer norms like
       layers.0.input_layernorm).
    """
    inner = model.model
    # 1. Direct attribute lookup
    for attr in ("norm", "ln_f", "final_layernorm", "final_norm"):
        if hasattr(inner, attr):
            return getattr(inner, attr), model.lm_head

    # 2. Walk all named modules, keep the last one that looks like a
    #    top-level norm: name ends with a norm-like word AND the full path
    #    contains no digit (excludes per-layer norms like layers.0.input_layernorm).
    import re
    candidate = None
    for name, module in model.named_modules():
        last_part = name.split(".")[-1]
        if re.search(r"norm|ln_f|layer_norm", last_part, re.IGNORECASE):
            if not re.search(r"\d", name):   # check full path, not just last_part
                candidate = module

    if candidate is not None:
        return candidate, model.lm_head

    raise AttributeError(
        "Cannot find final LayerNorm. "
        f"Available top-level attrs on model.model: {[n for n,_ in inner.named_children()]}"
    )


def _escape_token(s: str) -> str:
    """Replace whitespace characters with visible escape sequences."""
    return (s
            .replace("\r\n", "\\r\\n")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace(" ", "<sp>"))


@torch.no_grad()
def logit_lens_item(
    model,
    tokenizer,
    input_ids_list: List[int],
    top_k: int = 5,
    layer_indices: Optional[List[int]] = None,
    append_str: Optional[str] = None,
    last_layer_only: bool = False,
) -> Optional[list]:
    """
    Run logit lens on one item.

    By default, reads the hidden state at the last '=' token, which predicts
    the first generated token.

    With append_str set (e.g. ' ', '\n\n'), those characters are tokenized and
    appended after '='. The logit lens then probes the last appended token,
    predicting whatever comes after that string.

    Returns a list of dicts:
      [{"layer": int, "target_pos": int, "top": [(token_str, prob), ...]}, ...]
    or None if '=' is not found in the sequence.
    """
    norm, lm_head = get_final_norm_and_head(model)

    # Confirm '=' is present
    ids_check = torch.tensor([input_ids_list], dtype=torch.long, device=model.device)
    eq_idx = get_equal_token_index(tokenizer, ids_check)
    if eq_idx is None:
        return None

    if append_str is not None:
        # Tokenize the append string in context ("= <append_str>") and take
        # only the tokens that come after the '=' token.
        context_ids  = tokenizer.encode(f"= {append_str}", add_special_tokens=False)
        eq_ctx_ids   = tokenizer.encode("=",               add_special_tokens=False)
        suffix_ids   = context_ids[len(eq_ctx_ids):]
        if not suffix_ids:
            # Fallback: tokenize the string alone
            suffix_ids = tokenizer.encode(append_str, add_special_tokens=False)
        extended_ids = input_ids_list + suffix_ids
        target_pos   = len(extended_ids) - 1
    else:
        # Use '=' itself: predicts the first generated token.
        extended_ids = input_ids_list
        target_pos   = eq_idx

    # Debug: show sequence length, all '=' positions, and context around target
    all_toks     = tokenizer.convert_ids_to_tokens(extended_ids)
    all_eq_pos   = [i for i, t in enumerate(all_toks) if t and "=" in t]
    context_ids  = extended_ids[max(0, target_pos - 3): target_pos + 2]
    context_str  = tokenizer.convert_ids_to_tokens(context_ids)
    target_tok   = all_toks[target_pos]
    print(f"    [logit-lens] seq_len={len(extended_ids)}  all_eq_positions={all_eq_pos}  "
          f"eq_idx={eq_idx}  target_pos={target_pos}  "
          f"target_tok={target_tok!r}  context={context_str}",
          file=sys.stderr)

    input_ids = torch.tensor([extended_ids], dtype=torch.long, device=model.device)

    if last_layer_only:
        # Skip hidden states — just read logits directly (faster, less memory).
        outputs = model(input_ids=input_ids, output_hidden_states=False)
        logits  = outputs.logits[0, target_pos, :].float()
        probs   = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        top_tokens = [
            (_escape_token(tokenizer.decode([tid.item()])), float(p.item()))
            for tid, p in zip(top_ids, top_probs)
        ]
        # Use -1 as a sentinel layer index meaning "final layer"
        return [{"layer": -1, "target_pos": target_pos, "top": top_tokens}]

    outputs = model(input_ids=input_ids, output_hidden_states=True)
    # hidden_states: tuple of (1, seq_len, hidden_dim), len = num_layers + 1
    hidden_states = outputs.hidden_states

    last_layer_i = len(hidden_states) - 1
    results = []
    for layer_i, hs in enumerate(hidden_states):
        if layer_indices is not None and layer_i not in layer_indices:
            continue

        if layer_i == last_layer_i:
            # Use the model's own logits directly — architecture-agnostic,
            # captures any post-lm_head transforms.
            logits = outputs.logits[0, target_pos, :].float()
        else:
            h = hs[0, target_pos, :].unsqueeze(0)  # (1, hidden_dim), model dtype
            h_normed = norm(h)                     # (1, hidden_dim)
            logits = lm_head(h_normed)[0].float()  # (vocab_size,) — cast after lm_head
        probs  = F.softmax(logits, dim=-1)

        top_probs, top_ids = probs.topk(top_k)
        top_tokens = [
            (_escape_token(tokenizer.decode([tid.item()])), float(p.item()))
            for tid, p in zip(top_ids, top_probs)
        ]
        results.append({
            "layer":      layer_i,
            "target_pos": target_pos,
            "top":        top_tokens,
        })

    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def make_csv_fieldnames(layer_results: list, top_k: int) -> List[str]:
    """Build the CSV column names from the first item's layer results."""
    fields = ["item_id", "gold", "is_correct", "extraction_rule", "problem_type", "output_preview", "target_pos",
              "first_match_layer", "stays_matched", "stable_from_layer"]
    for r in layer_results:
        i = r["layer"]
        for rank in range(1, top_k + 1):
            fields.append(f"L{i}_tok{rank}")
            fields.append(f"L{i}_prob{rank}")
    return fields


def sanitize_field(value):
    """Replace raw commas in a field with '<comma>' to avoid CSV issues."""
    if isinstance(value, str):
        # Replace commas and ensure all whitespace characters are escaped
        return (value.replace(",", "<comma>")
                    .replace("，", "<comma>")  # also replace full-width comma
                    .replace("\r\n", "\\r\\n")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                    .replace(" ", "<sp>"))
    elif isinstance(value, (int, float)):
        # Convert numeric values to strings and sanitize
        return str(value).replace(",", "<comma>")
    return value

def layer_results_to_row(
    item_id: str, record: dict, layer_results: list, top_k: int,
    output_preview: str,
) -> dict:
    """Flatten logit-lens results for one item into a CSV row dict."""
    gold = record.get("gold_label", record.get("answer", ""))

    # Sort by layer index so iteration order is guaranteed
    sorted_results = sorted(layer_results, key=lambda r: r["layer"])
    final_top1 = sorted_results[-1]["top"][0][0]

    # First layer whose top-1 matches the final layer's top-1
    first_match_layer = next(
        (r["layer"] for r in sorted_results if r["top"][0][0] == final_top1), None
    )
    # Does it stay matched from first_match_layer to the end?
    if first_match_layer is None:
        stays_matched = 0
    else:
        stays_matched = int(all(
            r["top"][0][0] == final_top1
            for r in sorted_results
            if r["layer"] >= first_match_layer
        ))

    # Earliest layer from which the top-1 prediction never changes again through
    # to the final layer (the prediction has "stabilized").  Walk backwards: find
    # the longest unbroken suffix of layers that all agree on the same top-1 token.
    stable_from_layer = None
    if sorted_results:
        stable_tok = sorted_results[-1]["top"][0][0]
        stable_idx = len(sorted_results) - 1
        while stable_idx > 0 and sorted_results[stable_idx - 1]["top"][0][0] == stable_tok:
            stable_idx -= 1
        stable_from_layer = sorted_results[stable_idx]["layer"]

    # Extract problem_type from the `type` field in the JSONL record
    problem_type = record.get("type", "unknown")

    row = {
        "item_id":           sanitize_field(item_id),
        "gold":              sanitize_field(gold),
        "output_preview":    sanitize_field(output_preview),
        "target_pos":        sanitize_field(sorted_results[0]["target_pos"] if sorted_results else ""),
        "first_match_layer": sanitize_field(first_match_layer if first_match_layer is not None else ""),
        "stays_matched":     sanitize_field(stays_matched),
        "stable_from_layer": sanitize_field(stable_from_layer if stable_from_layer is not None else ""),
        "is_correct":       sanitize_field(int(record.get("contains_gold_label") is True)),
        "extraction_rule":  sanitize_field(record.get("extractor", "")),
        "problem_type":     sanitize_field(problem_type),
    }
    for r in sorted_results:
        i = r["layer"]
        for rank, (tok, prob) in enumerate(r["top"], start=1):
            row[f"L{i}_tok{rank}"]  = sanitize_field(tok)
            row[f"L{i}_prob{rank}"] = sanitize_field(f"{prob:.6f}")
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to HF model directory")
    parser.add_argument("--results-dir", required=True,
                        help="Root results directory (model_folder/lang/prompt.jsonl)")
    parser.add_argument("--lang",   required=True, help="Language code (e.g. en)")
    parser.add_argument("--prompt", required=True,
                        help="Prompt variant (e.g. answer, compute, solve, ...)")
    parser.add_argument("--item-id", default=None,
                        help="Specific item ID to process. Overrides --n-items.")
    parser.add_argument("--n-items", type=int, default=None,
                        help="Limit to first N items (default: all items)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top tokens to record per layer (default: 5)")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: all). "
                             "Layer 0 = embedding, layer N = last transformer block.")
    parser.add_argument("--append", default=None, metavar="STR",
                        help="String to tokenize and append after '=' before probing. "
                             "E.g. --append ' ' (space), --append $'\\n\\n' (two newlines). "
                             "Default: probe '=' directly (predicts 1st generated token).")
    parser.add_argument("--last-layer-only", action="store_true",
                        help="Only compute top-k for the final layer (uses outputs.logits "
                             "directly — faster and narrower CSV output).")
    parser.add_argument("--output", default=None,
                        help="Write CSV to this file instead of stdout")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model_name  = Path(args.model_path).name
    model_dir   = find_model_dir(results_dir, model_name)

    jsonl_path = model_dir / args.lang / f"{args.prompt}.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(jsonl_path)
    if not records:
        print("ERROR: no records found", file=sys.stderr)
        sys.exit(1)

    # Select items
    if args.item_id:
        if args.item_id not in records:
            print(f"ERROR: item '{args.item_id}' not found in {jsonl_path}",
                  file=sys.stderr)
            sys.exit(1)
        item_ids = [args.item_id]
    else:
        item_ids = sorted(records.keys())
        if args.n_items is not None:
            item_ids = item_ids[: args.n_items]

    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    print(f"Loading model: {args.model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Processing {len(item_ids)} item(s)...\n", file=sys.stderr)

    out = open(args.output, "w", encoding="utf-8", newline="") if args.output else sys.stdout
    writer = None  # initialised after first successful item (layer count unknown until then)
    n_written = 0
    n_skipped = 0

    try:
        for idx, item_id in enumerate(item_ids):
            print(f"  [{idx+1}/{len(item_ids)}] {item_id}", file=sys.stderr)
            rec = records[item_id]
            ids = rec.get("input_ids")
            if not ids:
                print(f"    SKIP: no input_ids", file=sys.stderr)
                n_skipped += 1
                continue

            layer_results = logit_lens_item(
                model, tokenizer, ids,
                top_k=args.top_k,
                layer_indices=layer_indices,
                append_str=args.append,
                last_layer_only=args.last_layer_only,
            )
            if layer_results is None:
                print(f"    SKIP: '=' token not found", file=sys.stderr)
                n_skipped += 1
                continue

            out_ids = rec.get("output_ids") or []
            output_preview = _escape_token(
                tokenizer.decode(out_ids[:10], skip_special_tokens=False)
            )
            row = layer_results_to_row(item_id, rec, layer_results, args.top_k,
                                       output_preview)

            if writer is None:
                fieldnames = make_csv_fieldnames(layer_results, args.top_k)
                writer = csv.DictWriter(out, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(row)
            out.flush()
            n_written += 1

    finally:
        if args.output:
            out.close()

    print(f"\nDone. {n_written} rows written, {n_skipped} skipped.", file=sys.stderr)
    if args.output:
        print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
