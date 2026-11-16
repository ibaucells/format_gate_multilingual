#!/usr/bin/env python3
"""
Build a per-item table of first-generated-token information for a given model
from a results directory (default: results/raw/base_and_instruct).

For every JSONL file found under:
    <results_dir>/<model_folder>/<lang>/<prompt>.jsonl

…each record produces exactly one output row. The columns are:
    item_id, lang, prompt, type,
    answer_correct (1/0), answer_incorrect (1/0), no_answer (1/0),
    extraction_rule, first_output_id

Category definitions (exactly one flag is 1 per row):
    answer_correct   – contains_gold_label is True
    no_answer        – extracted_answer is [] (and not correct)
    answer_incorrect – extracted_answer is non-empty but not the gold label
                       (contains_gold_label is False or "yes_but_spurious")

Usage:
    python first_token_table.py \\
        --model salamandra-7b \\
        --results-dir results/raw/base_and_instruct \\
        --output first_token_table_salamandra-7b.csv

Example:
    python first_token_table.py --model Meta-Llama-3.2-3B --results-dir results/raw/base_and_instruct --output first_token_analysis/first_token_table_llama3b.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_model_dir(results_dir: Path, model_name: str) -> Path:
    candidates = [d for d in results_dir.iterdir()
                  if d.is_dir() and model_name in d.name]
    if not candidates:
        raise FileNotFoundError(
            f"No folder matching '{model_name}' under {results_dir}\n"
            f"Available folders: {[d.name for d in results_dir.iterdir() if d.is_dir()]}"
        )
    if len(candidates) > 1:
        print(
            f"WARNING: Multiple folders match '{model_name}': "
            f"{[d.name for d in candidates]}. Using first: {candidates[0].name}",
            file=sys.stderr,
        )
    return candidates[0]


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _escape_token(s: str) -> str:
    """Make whitespace characters visible for display in a CSV cell."""
    return (s
            .replace("\r\n", "\\r\\n")
            .replace("\n",   "\\n")
            .replace("\r",   "\\r")
            .replace("\t",   "\\t")
            .replace(" ",    "<sp>"))


def classify(rec: dict) -> tuple[int, int, int]:
    """Return (answer_correct, answer_incorrect, no_answer) flags (exactly one is 1)."""
    gold = rec.get("contains_gold_label")
    extracted = rec.get("extracted_answer", [])

    if gold is True:
        return (1, 0, 0)
    if not extracted:          # empty list — nothing was extracted
        return (0, 0, 1)
    return (0, 1, 0)           # extracted something but wrong / spurious


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a per-item table of first-generated-token info."
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name substring to match the folder inside results-dir "
             "(e.g. 'salamandra-7b', 'Meta-Llama-3.2-3B').",
    )
    parser.add_argument(
        "--results-dir",
        default="results/raw/base_and_instruct",
        help="Root results directory (default: results/raw/base_and_instruct).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for the output CSV (default: first_token_table_<model>.csv).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    model_dir = find_model_dir(results_dir, args.model)
    output_path = Path(args.output) if args.output else Path(
        f"first_token_table_{args.model}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the model path from the first record in the first JSONL found,
    # then load the tokenizer once for the whole run.
    model_path = None
    for lang_dir in sorted(model_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        for jsonl_path in sorted(lang_dir.glob("*.jsonl")):
            for rec in iter_jsonl(jsonl_path):
                model_path = rec.get("model")
                break
            if model_path:
                break
        if model_path:
            break

    if not model_path:
        print("ERROR: could not find 'model' field in any record", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer from: {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
    print("Tokenizer loaded.\n", file=sys.stderr)

    fieldnames = [
        "item_id", "lang", "prompt", "type",
        "answer_correct", "answer_incorrect", "no_answer",
        "extraction_rule", "first_output_id", "first_token_decoded",
    ]

    total_records = 0
    total_skipped = 0

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for lang_dir in sorted(model_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name

            for jsonl_path in sorted(lang_dir.glob("*.jsonl")):
                prompt = jsonl_path.stem
                n_file = 0

                for rec in iter_jsonl(jsonl_path):
                    out_ids = rec.get("output_ids") or []
                    if not out_ids:
                        total_skipped += 1
                        continue

                    correct, incorrect, no_ans = classify(rec)
                    first_id = out_ids[0]
                    decoded = _escape_token(
                        tokenizer.decode([first_id], skip_special_tokens=False)
                    )
                    writer.writerow({
                        "item_id":             rec.get("id", ""),
                        "lang":                lang,
                        "prompt":              prompt,
                        "type":                rec.get("type", "unknown"),
                        "answer_correct":      correct,
                        "answer_incorrect":    incorrect,
                        "no_answer":           no_ans,
                        "extraction_rule":     rec.get("extractor") or "none",
                        "first_output_id":     first_id,
                        "first_token_decoded": decoded,
                    })
                    total_records += 1
                    n_file += 1

                print(
                    f"  {lang}/{prompt}.jsonl — {n_file} records",
                    file=sys.stderr,
                )

    print(
        f"\nTotal records written: {total_records}  "
        f"(skipped – no output_ids: {total_skipped})",
        file=sys.stderr,
    )
    print(f"Output written to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
