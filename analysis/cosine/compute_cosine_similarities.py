#!/usr/bin/env python3
"""
For a given model, compute pairwise cosine similarities between hidden states
at the last '=' token position across all (lang, prompt) condition pairs,
for each of the 600 math items.

For every unordered pair of conditions (lang1,prompt1) × (lang2,prompt2) and
each item, we extract the hidden state vector (all layers) at the last '='
token, compute cosine similarity per layer, and write one row per
(item, pair, layer):

  item_id | lang1 | lang2 | prompt1 | prompt2 | layer | cos_sim | behavior1 | behavior2

behavior ∈ {ANSWER-RIGHT, ANSWER-WRONG, NO-ANSWER, UNKNOWN}
  derived from reextracted JSONL records (contains_gold_label, extracted_answer, extractor).

Output: a single .parquet file (written incrementally via pyarrow).

Usage:
  python compute_cosine_similarities.py \\
    --model-path /path/to/hf-models/Qwen3-0.6B-Base \\
    --results-dir results/raw/base_and_instruct \\
    --output cosine_sims_Qwen3-0.6B-Base.parquet

Requires: transformers torch pandas pyarrow tqdm
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGS   = ["ar", "ca", "es", "zh", "ru", "eu", "fr", "de", "en", "fake", "empty"]
PROMPTS = ["answer", "compute", "determine", "solve", "long_numeric"]

PARQUET_SCHEMA = pa.schema([
    pa.field("item_id",    pa.string()),
    pa.field("lang1",      pa.dictionary(pa.int8(), pa.string())),
    pa.field("lang2",      pa.dictionary(pa.int8(), pa.string())),
    pa.field("prompt1",    pa.dictionary(pa.int8(), pa.string())),
    pa.field("prompt2",    pa.dictionary(pa.int8(), pa.string())),
    pa.field("layer",      pa.int16()),
    pa.field("cos_sim",    pa.float32()),
    pa.field("behavior1",  pa.dictionary(pa.int8(), pa.string())),
    pa.field("behavior2",  pa.dictionary(pa.int8(), pa.string())),
    pa.field("extractor1", pa.dictionary(pa.int8(), pa.string())),
    pa.field("extractor2", pa.dictionary(pa.int8(), pa.string())),
])

# ---------------------------------------------------------------------------
# Behavior classification (same logic as summarize_results.py)
# ---------------------------------------------------------------------------

def classify_answer(record: Dict[str, Any]) -> str:
    """Return one of: ANSWER-RIGHT, ANSWER-WRONG, NO-ANSWER, UNKNOWN."""
    if record.get("contains_gold_label") or record.get("contains_gold_number"):
        return "ANSWER-RIGHT"
    if record.get("extracted_answer"):
        return "ANSWER-WRONG"
    extractor = record.get("extractor", "")
    if extractor in ("", "unknown"):
        return "UNKNOWN"
    return "NO-ANSWER"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> Dict[str, dict]:
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
    candidates = [d for d in results_dir.iterdir() if d.is_dir() and model_name in d.name]
    if not candidates:
        raise FileNotFoundError(
            f"No folder matching '{model_name}' in {results_dir}"
        )
    return candidates[0]

# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def get_equal_token_index(tokenizer, input_ids: torch.Tensor) -> Optional[int]:
    """Return index of the last '=' token in input_ids (1-D or 2-D batch=1)."""
    ids_1d = input_ids[0] if input_ids.dim() == 2 else input_ids

    eq_id = tokenizer.convert_tokens_to_ids("=")
    if eq_id is not None and eq_id != tokenizer.unk_token_id:
        positions = (ids_1d == eq_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return int(positions[-1].item())

    # Fallback: scan decoded tokens for any containing '='
    toks = tokenizer.convert_ids_to_tokens(ids_1d.tolist())
    positions = [i for i, t in enumerate(toks) if t and "=" in t]
    return positions[-1] if positions else None


@torch.no_grad()
def hidden_at_eq(
    model,
    tokenizer,
    input_ids_list: List[int],
) -> Optional[torch.Tensor]:
    """
    Run a forward pass using the stored input_ids (from the JSONL record).
    Returns tensor of shape (num_layers, hidden_dim) at the last '=' position,
    or None if '=' is not found.
    """
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=model.device)
    eq_idx = get_equal_token_index(tokenizer, input_ids)
    if eq_idx is None:
        return None

    outputs = model(input_ids=input_ids, output_hidden_states=True)
    # hidden_states: tuple len = num_layers+1 (embedding + each transformer layer)
    # each element shape: (1, seq_len, hidden_dim)
    per_layer = [hs[0, eq_idx, :].float().cpu() for hs in outputs.hidden_states]
    return torch.stack(per_layer, dim=0)  # (num_layers, hidden_dim)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to HF model directory")
    parser.add_argument("--results-dir", required=True,
                        help="Root of reextracted results "
                             "(model_folder/lang/prompt.jsonl)")
    parser.add_argument("--output", required=True,
                        help="Output .parquet file path")
    parser.add_argument("--langs",   default=",".join(LANGS),
                        help="Comma-separated languages (default: all 10)")
    parser.add_argument("--prompts", default=",".join(PROMPTS),
                        help="Comma-separated prompt variants (default: all 5)")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="Write to parquet after this many items (default: 50)")
    args = parser.parse_args()

    langs   = [l.strip() for l in args.langs.split(",")   if l.strip()]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    results_dir = Path(args.results_dir)
    model_name  = Path(args.model_path).name
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_dir = find_model_dir(results_dir, model_name)
    print(f"Results dir : {model_dir}")

    # ------------------------------------------------------------------
    # Load all records: by_id[item_id][(lang, prompt)] = record
    # ------------------------------------------------------------------
    conditions: List[Tuple[str, str]] = []
    by_id: Dict[str, Dict[Tuple[str, str], dict]] = defaultdict(dict)

    for lang in langs:
        for prompt in prompts:
            path = model_dir / lang / f"{prompt}.jsonl"
            if not path.exists():
                print(f"  [WARN] missing: {path}")
                continue
            recs = load_jsonl(path)
            for item_id, rec in recs.items():
                by_id[item_id][(lang, prompt)] = rec
            if (lang, prompt) not in conditions:
                conditions.append((lang, prompt))

    all_item_ids   = sorted(by_id.keys())
    condition_pairs = list(combinations(range(len(conditions)), 2))
    n_items  = len(all_item_ids)
    n_pairs  = len(condition_pairs)

    print(f"Items       : {n_items}")
    print(f"Conditions  : {len(conditions)}")
    print(f"Pairs/item  : {n_pairs}")
    print(f"Est. rows   : {n_items * n_pairs} × num_layers")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\nLoading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Process items, write parquet incrementally
    # ------------------------------------------------------------------
    writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA)

    batch_rows: Dict[str, list] = {f.name: [] for f in PARQUET_SCHEMA}
    n_written = 0

    def flush_batch():
        nonlocal n_written
        if not batch_rows["item_id"]:
            return
        table = pa.table(batch_rows, schema=PARQUET_SCHEMA)
        writer.write_table(table)
        n_written += len(batch_rows["item_id"])
        for v in batch_rows.values():
            v.clear()

    for item_idx, item_id in enumerate(tqdm(all_item_ids, desc="Items")):
        item_conds = by_id[item_id]

        # Compute hidden states for every available condition of this item
        vecs: Dict[Tuple[str, str], Optional[torch.Tensor]] = {}
        for cond, rec in item_conds.items():
            ids = rec.get("input_ids")
            if not ids:
                vecs[cond] = None
                continue
            vecs[cond] = hidden_at_eq(model, tokenizer, ids)

        # All pairwise cosine similarities
        for i, j in condition_pairs:
            cond1 = conditions[i]
            cond2 = conditions[j]
            rec1 = item_conds.get(cond1)
            rec2 = item_conds.get(cond2)
            if rec1 is None or rec2 is None:
                continue
            v1 = vecs.get(cond1)
            v2 = vecs.get(cond2)
            if v1 is None or v2 is None:
                continue

            lang1, prompt1 = cond1
            lang2, prompt2 = cond2
            behavior1  = classify_answer(rec1)
            behavior2  = classify_answer(rec2)
            extractor1 = rec1.get("extractor", "") or ""
            extractor2 = rec2.get("extractor", "") or ""

            # cos_sim per layer: (num_layers,)
            cos_sim = F.cosine_similarity(v1, v2, dim=-1, eps=1e-6).clamp(-1.0, 1.0)
            num_layers = cos_sim.shape[0]

            batch_rows["item_id"].extend(   [item_id]    * num_layers)
            batch_rows["lang1"].extend(     [lang1]      * num_layers)
            batch_rows["lang2"].extend(     [lang2]      * num_layers)
            batch_rows["prompt1"].extend(   [prompt1]    * num_layers)
            batch_rows["prompt2"].extend(   [prompt2]    * num_layers)
            batch_rows["layer"].extend(     list(range(num_layers)))
            batch_rows["cos_sim"].extend(   cos_sim.tolist())
            batch_rows["behavior1"].extend( [behavior1]  * num_layers)
            batch_rows["behavior2"].extend( [behavior2]  * num_layers)
            batch_rows["extractor1"].extend([extractor1] * num_layers)
            batch_rows["extractor2"].extend([extractor2] * num_layers)

        # Flush every N items
        if (item_idx + 1) % args.flush_every == 0:
            flush_batch()
            print(f"  {item_idx + 1}/{n_items} items, {n_written} rows written")

    flush_batch()
    writer.close()
    print(f"\nDone. Total rows: {n_written}  ->  {output_path}")


if __name__ == "__main__":
    main()
