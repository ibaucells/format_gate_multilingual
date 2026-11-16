"""
summarize_steering_outputs.py

Reads JSONL files from the steering_outputs/ folder and, for each file,
produces a CSV with counts of ANSWER-CORRECT, ANSWER-INCORRECT, NO-ANSWER,
and UNKNOWN broken down by layer (and coeff).

Steering records have fields: key, prompt_text, model_output_raw,
extracted_answer, extractor, answer, correct, layer, coeff.

The answer-type logic mirrors classify_answer() in summarize_results.py,
adapted for the steering record schema:
  ANSWER-CORRECT   correct == 1
  ANSWER-INCORRECT correct == 0  AND  extracted_answer is non-empty
  NO-ANSWER        extracted_answer is empty AND extractor is a named no-answer extractor
  UNKNOWN          extracted_answer is empty AND extractor is "" or "unknown"

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

# Matches known feature keys followed by their value (value may contain hyphens
# or underscores, e.g. prompt values like "long_numeric").
# "few-shot" is a bare flag (to_dict() stores it as {"few-shot": "true"}), so it
# always shows up as literal "few-shot-true" in the filename.
_FEATURE_RE = re.compile(
    r"(answer_type|language|prompt|few-shot)"
    r"-([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*?)"
    r"(?=_(?:answer_type|language|prompt|few-shot)-|$)"
)


def _parse_features(segment: str) -> dict:
    """Parse 'answer_type-NO-ANSWER_language-es_prompt-answer' into a dict."""
    return {m.group(1): m.group(2) for m in _FEATURE_RE.finditer(segment)}


def _feature_label(features: dict, common: dict) -> str:
    """Build 'language+prompt+answer_type[+few-shot]' label, falling back to common for missing keys."""
    lang = features.get("language") or common.get("language", "")
    prompt = features.get("prompt") or common.get("prompt", "")
    answer_type = features.get("answer_type") or common.get("answer_type", "")
    parts = [p for p in [lang, prompt, answer_type] if p]
    if _is_few_shot(features, common):
        parts.append("few-shot")
    return "+".join(parts)


def _is_few_shot(features: dict, common: dict) -> bool:
    return features.get("few-shot") == "true" or common.get("few-shot") == "true"


# Compact "(language,prompt)" labels, e.g. for the by_layer pivot sheet.
_PROMPT_ABBREV = {
    "answer": "ans",
    "compute": "comp",
    "determine": "det",
    "solve": "solve",
    "long_numeric": "long_num",
}


def _abbrev_prompt(prompt: str) -> str:
    return _PROMPT_ABBREV.get(prompt, prompt)


def _is_instruct_model(model_name: str) -> bool:
    name = model_name.lower()
    return "instruct" in name or name.endswith("-it")


def _compact_label(features: dict, common: dict, model_name: str = None) -> str:
    """Build '(lang,prompt)', or '(fs,it,lang,prompt)' when few-shot/instruct apply."""
    lang = features.get("language") or common.get("language", "")
    prompt = features.get("prompt") or common.get("prompt", "")
    tags = []
    if _is_few_shot(features, common):
        tags.append("fs")
    if model_name and _is_instruct_model(model_name):
        tags.append("it")
    prefix = ",".join(tags) + "," if tags else ""
    return f"({prefix}{lang},{_abbrev_prompt(prompt)})"


def parse_filename(fname: str) -> dict:
    """
    Parse a steering output filename into component labels.

    Expected format:
      {model}__common__{cf}__pos__{pf}__neg__{nf}__eval__{ef}__side__{side}.jsonl

    Returns dict with keys: model, positive, negative, eval, side,
    positive_few_shot, negative_few_shot, steering_vec, eval_compact.
    """
    stem = fname.removesuffix(".jsonl")
    segments = stem.split("__")

    model_str = segments[0] if segments else ""
    model_names = model_str.split("+")
    model_pos_name = model_names[0] if len(model_names) >= 1 else None
    model_neg_name = model_names[1] if len(model_names) == 2 else None

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

    # eval_compact never carries fs/it tags: it describes the test condition
    # (language+prompt), not which model/shot-count produced the vector.
    eval_compact = (
        _compact_label(neg_feat, common)
        if eval_str == "same-as-neg"
        else _compact_label(_parse_features(eval_str), common)
    )

    return {
        "model": model_str,
        "positive": _feature_label(pos_feat, common),
        "negative": _feature_label(neg_feat, common),
        "eval": eval_label,
        "side": sections.get("side", ""),
        "positive_few_shot": _is_few_shot(pos_feat, common),
        "negative_few_shot": _is_few_shot(neg_feat, common),
        "steering_vec": f"{_compact_label(pos_feat, common, model_pos_name)}-"
                         f"{_compact_label(neg_feat, common, model_neg_name)}",
        "eval_compact": eval_compact,
    }


# ---------------------------------------------------------------------------
# Answer classification
# ---------------------------------------------------------------------------

def classify_steering_record(record: dict) -> str:
    """Classify a steering output record into one of four answer types."""
    if record.get("correct") == 1:
        return "ANSWER-CORRECT"
    if record.get("extracted_answer"):
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

    # Recursive: works both when all .jsonl files sit directly in --steering-dir
    # and when each job has its own subfolder (as run_steering_vectors.sh does).
    jsonl_paths = sorted(
        os.path.join(root, f)
        for root, _dirs, files in os.walk(args.steering_dir)
        for f in files
        if f.endswith(".jsonl")
    )
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files found under {args.steering_dir}")

    all_rows = []

    for path in jsonl_paths:
        fname = os.path.basename(path)
        # Relative path (not bare filename) so two different job subfolders that
        # happen to produce the identical config filename don't collide.
        rel_path = os.path.relpath(path, args.steering_dir)
        meta = parse_filename(fname)

        df = summarize_file(path)
        df.insert(0, "source_file", rel_path)
        df.insert(1, "model", meta["model"])
        df.insert(2, "steering_vec", meta["steering_vec"])
        df.insert(3, "positive", meta["positive"])
        df.insert(4, "positive_few_shot", meta["positive_few_shot"])
        df.insert(5, "negative", meta["negative"])
        df.insert(6, "negative_few_shot", meta["negative_few_shot"])
        df.insert(7, "eval", meta["eval"])
        df.insert(8, "eval_compact", meta["eval_compact"])
        df.insert(9, "side", meta["side"])

        out_name = rel_path.replace(os.sep, "__").replace(".jsonl", "_summary.csv")
        out_path = os.path.join(args.out_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = os.path.join(args.out_dir, "all_steering_summaries.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Combined CSV saved: {combined_path}")

    write_excel_summary(combined, args.out_dir)


def build_by_layer_pivot(combined: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model, steering_vec, eval, side, coeff), one column per layer
    (L2, L6, ...) holding accuracy. Baseline rows (layer is null, coeff=0) have no
    layer to pivot on and are excluded here — they're still fully visible in the
    detailed sheet/CSV.

    `side` must be part of the index: two jobs can share the same steering_vec,
    eval_compact, and coeff while differing only in eval_side (e.g. a "core"
    negative-side job and a "sanity" positive-side job built from the same
    positive/negative features) — eval_compact is derived from negative_features
    alone, so it doesn't disambiguate them. Without `side` in the index, such rows
    collide in the pivot and aggfunc="first" silently drops one job's data.
    """
    df = combined.copy()
    df["accuracy"] = df["answer_correct"] / df["total"]
    df = df[df["layer"].notna()]

    pivot = df.pivot_table(
        index=["model", "steering_vec", "eval_compact", "side", "coeff"],
        columns="layer",
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(columns={"eval_compact": "eval"})

    layer_cols = [c for c in pivot.columns if c not in ("model", "steering_vec", "eval", "side", "coeff")]
    layer_cols_sorted = sorted(layer_cols, key=float)
    rename_map = {c: f"L{int(float(c))}" for c in layer_cols_sorted}
    pivot = pivot.rename(columns=rename_map)

    return pivot[["model", "steering_vec", "eval", "side", "coeff"] + list(rename_map.values())]


def write_excel_summary(combined: pd.DataFrame, out_dir: str) -> None:
    """Write all_steering_summaries.xlsx with a 'detailed' sheet (same as the combined
    CSV) and a 'by_layer' sheet (accuracy per layer, pivoted). Requires openpyxl;
    skipped with a warning (not a hard failure) if it isn't installed."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[Excel] WARNING: openpyxl not installed (pip install openpyxl) — "
              "skipping all_steering_summaries.xlsx, CSVs are still written.")
        return

    by_layer = build_by_layer_pivot(combined)
    xlsx_path = os.path.join(out_dir, "all_steering_summaries.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="detailed", index=False)
        by_layer.to_excel(writer, sheet_name="by_layer", index=False)
    print(f"Excel summary saved: {xlsx_path}")


if __name__ == "__main__":
    main()
