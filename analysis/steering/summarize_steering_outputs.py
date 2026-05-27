"""
summarize_steering_outputs.py

Reads JSONL files from the steering_outputs/ folder and, for each file,
produces a CSV with counts of ANSWER-CORRECT, ANSWER-INCORRECT, NO-ANSWER,
and UNKNOWN broken down by layer (and coeff).

Steering records have fields: key, prompt, gen, extracted, extractor,
gold, correct, layer, coeff.

The answer-type logic mirrors classify_answer() in summarize_results.py,
adapted for the steering record schema:
  ANSWER-CORRECT   correct == 1
  ANSWER-INCORRECT correct == 0  AND  extracted is non-empty
  NO-ANSWER        extracted is empty AND extractor is a named no-answer extractor
  UNKNOWN          extracted is empty AND extractor is "" or "unknown"

Filename columns (parsed from the JSONL filename):
  positive   e.g. "ca+answer+ANSWER-C"  (language+prompt+answer_type of pos set)
  negative   e.g. "es+answer+NO-ANSWER"
  eval       e.g. "ar+answer+NO-ANSWER" or "same-as-neg"
  side       "negative" or "positive"

Usage
-----
  python summarize_steering_outputs.py
  python summarize_steering_outputs.py --steering-dir steering_outputs --out-dir steering_summaries
"""

import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd

from format_gate_multilingual.analysis.behavioral.summarize_results import (
    NO_ANSWER_EXTRACTORS,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Matches known feature keys followed by their value (value may contain hyphens).
_FEATURE_RE = re.compile(
    r"(answer_type|language|prompt)"
    r"-([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*?)"
    r"(?=_(?:answer_type|language|prompt)-|$)"
)


def _parse_features(segment: str) -> dict:
    """Parse 'answer_type-NO-ANSWER_language-es_prompt-answer' into a dict."""
    return {m.group(1): m.group(2) for m in _FEATURE_RE.finditer(segment)}


def _feature_label(features: dict, common: dict) -> str:
    """Build 'language+prompt+answer_type' label, falling back to common for missing keys."""
    lang = features.get("language") or common.get("language", "")
    prompt = features.get("prompt") or common.get("prompt", "")
    answer_type = features.get("answer_type") or common.get("answer_type", "")
    return "+".join(p for p in [lang, prompt, answer_type] if p)


def parse_filename(fname: str) -> dict:
    """
    Parse a steering output filename into component labels.

    Expected format:
      {model}__common__{cf}__pos__{pf}__neg__{nf}__eval__{ef}__side__{side}.jsonl

    Returns dict with keys: positive, negative, eval, side.
    """
    stem = fname.removesuffix(".jsonl")
    segments = stem.split("__")

    # Walk segments; each marker word is followed by its value segment.
    sections: dict[str, str] = {}
    markers = {"common", "pos", "neg", "eval", "side"}
    i = 0
    while i < len(segments):
        if segments[i] in markers and i + 1 < len(segments):
            sections[segments[i]] = segments[i + 1]
            i += 2
        else:
            i += 1

    common = _parse_features(sections.get("common", ""))
    pos_feat = _parse_features(sections.get("pos", ""))
    neg_feat = _parse_features(sections.get("neg", ""))
    eval_str = sections.get("eval", "")

    eval_label = (
        eval_str if eval_str == "same-as-neg"
        else _feature_label(_parse_features(eval_str), common)
    )

    return {
        "positive": _feature_label(pos_feat, common),
        "negative": _feature_label(neg_feat, common),
        "eval": eval_label,
        "side": sections.get("side", ""),
    }


# ---------------------------------------------------------------------------
# Answer classification
# ---------------------------------------------------------------------------

def classify_steering_record(record: dict) -> str:
    """Classify a steering output record into one of four answer types."""
    if record.get("correct") == 1:
        return "ANSWER-CORRECT"
    if record.get("extracted"):
        return "ANSWER-INCORRECT"
    extractor = record.get("extractor", "")
    if extractor in NO_ANSWER_EXTRACTORS:
        return "NO-ANSWER"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Per-file summarisation
# ---------------------------------------------------------------------------

def summarize_file(path: str) -> pd.DataFrame:
    """
    Read a steering output JSONL file and return a DataFrame with columns:
      layer, coeff, answer_correct, answer_incorrect, no_answer, unknown, total
    one row per (layer, coeff) combination.
    """
    counts: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"ANSWER-CORRECT": 0, "ANSWER-INCORRECT": 0, "NO-ANSWER": 0, "UNKNOWN": 0}
    )

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            answer_type = classify_steering_record(record)
            key = (record.get("layer"), record.get("coeff"))
            counts[key][answer_type] += 1

    rows = []
    for (layer, coeff), c in sorted(
        counts.items(), key=lambda x: (x[0][0] is None, x[0][0], x[0][1])
    ):
        total = sum(c.values())
        rows.append({
            "layer": layer,
            "coeff": coeff,
            "answer_correct": c["ANSWER-CORRECT"],
            "answer_incorrect": c["ANSWER-INCORRECT"],
            "no_answer": c["NO-ANSWER"],
            "unknown": c["UNKNOWN"],
            "total": total,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Summarize steering outputs by layer into CSV.")
    ap.add_argument(
        "--steering-dir",
        default=os.path.join(os.path.dirname(__file__), "steering_outputs"),
        help="Folder containing steering output JSONL files.",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "steering_summaries"),
        help="Output directory for per-file CSV summaries.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jsonl_files = sorted(
        f for f in os.listdir(args.steering_dir) if f.endswith(".jsonl")
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {args.steering_dir}")

    all_rows = []

    for fname in jsonl_files:
        path = os.path.join(args.steering_dir, fname)
        meta = parse_filename(fname)

        df = summarize_file(path)
        df.insert(0, "source_file", fname)
        df.insert(1, "positive", meta["positive"])
        df.insert(2, "negative", meta["negative"])
        df.insert(3, "eval", meta["eval"])
        df.insert(4, "side", meta["side"])

        out_path = os.path.join(args.out_dir, fname.replace(".jsonl", "_summary.csv"))
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = os.path.join(args.out_dir, "all_steering_summaries.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Combined CSV saved: {combined_path}")


if __name__ == "__main__":
    main()
