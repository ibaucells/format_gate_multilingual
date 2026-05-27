"""
summarize_results.py

Reads a results folder (structure: base_dir/<model_dir>/<language>/<prompt>.jsonl)
and generates up to 5 CSV files:

  1. accuracy_mean_stdev.csv
       Model × language pivot table.
       Cell value = "mean ± stdev" of per-prompt accuracy.
       Last column is GLOBAL (aggregated over all languages).

  2. accuracy_by_type.csv
       Flat table: model, language, prompt, type, n, accuracy.

  3. accuracy_by_prompt.csv
       Flat table: model, language, prompt, n, accuracy (collapsed across problem types).

  4. answer_types.csv
       Flat table: model, language, prompt,
       n, answer_right, answer_wrong, no_answer, unknown,
       answer_right_pct, answer_wrong_pct, no_answer_pct, unknown_pct.

  4b. answer_types_by_type.csv
       Same as answer_types.csv but also broken down by problem type.
       Adds columns: type, accuracy.

  5. per_language_consistency.csv  (skipped with --skip-consistency)
       Per (model, language): cross-lingual consistency (XC) and prompt
       consistency (PC), used as input for plot_delta_heatmap() and
       summarize_delta_from_csv() in helper_graphics.py.

Answer-type definitions
-----------------------
  ANSWER-RIGHT   contains_gold_label is True
  ANSWER-WRONG   contains_gold_label is False  AND  extracted_answer is non-empty
  NO-ANSWER      extracted_answer is empty      AND  extractor is a named no-answer extractor
  UNKNOWN        extracted_answer is empty      AND  extractor == "" or "unknown" (final fallback)

Usage
-----
python summarize_results.py --results-dir results/raw/fewshot \\
                               --out-dir summarized_results_fs --skip-consistency

python summarize_results.py --results-dir results/raw/base_and_instruct \\
                               --out-dir summarized_results --skip-consistency

"""

import argparse
import json
import os
from typing import Iterator, Tuple, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Consistency helpers (inlined from inspect_csv.py to avoid dependencies)
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _jaccard_pair(a, b):
    a = a.astype(bool).to_numpy()
    b = b.astype(bool).to_numpy()
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else np.nan


def _mean_pairwise_jaccard(wide):
    cols = list(wide.columns)
    vals = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            vals.append(_jaccard_pair(wide[cols[i]], wide[cols[j]]))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan


def _fisher_mean_corr(r_vals):
    r = np.asarray(r_vals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    r = np.clip(r, -1 + _EPS, 1 - _EPS)
    return float(np.tanh(np.arctanh(r).mean()))


def _fisher_agg(df_in, group_col, corr_col, out_col):
    s = df_in[np.isfinite(df_in[corr_col])].copy()
    if s.empty:
        return pd.DataFrame(columns=[group_col, out_col])
    return (
        s.groupby(group_col, sort=False)[corr_col]
        .apply(lambda col: _fisher_mean_corr(col.to_numpy(float)))
        .reset_index(name=out_col)
    )


def _prompt_consistency_summary(df):
    """Per (model, language): Fisher-mean prompt-prompt correlation + mean accuracy."""
    rows = []
    for (model, language), g in df.groupby(["model", "language"], sort=False):
        wide = g.pivot_table(index="id", columns="prompt", values="score", aggfunc="mean")
        n_items = int(wide.shape[0])
        n_prompts = int(wide.shape[1])
        mean_acc = float(wide.mean().mean()) if n_items and n_prompts else np.nan
        degenerate = int(n_prompts >= 2 and wide.nunique(dropna=True).max() <= 1)
        if n_prompts < 2:
            avg_corr = np.nan
        elif degenerate:
            avg_corr = 1.0
        else:
            corr = wide.corr(method="pearson")
            r = corr.to_numpy()[np.triu_indices_from(corr, k=1)]
            avg_corr = _fisher_mean_corr(r)
        jaccard_index = _mean_pairwise_jaccard(wide)
        rows.append({
            "model": model, "language": language,
            "n_items": n_items, "n_prompts": n_prompts,
            "mean_acc": mean_acc, "degenerate": degenerate,
            "prompt_consistency_fisher": avg_corr,
            "jaccard_index": jaccard_index,
        })
    return pd.DataFrame(rows)


def _per_lang_consistency(df, min_periods=4):
    """Per (model, language): Fisher-mean cross-lingual correlation + mean accuracy."""
    rows = []
    for (model, prompt), g in df.groupby(["model", "prompt"], sort=False):
        wide = g.pivot_table(index="id", columns="language", values="score", aggfunc="mean")
        if wide.shape[1] < 2:
            continue
        lang_mean = wide.mean(axis=0)
        corr = wide.corr(method="pearson", min_periods=min_periods)
        corr = corr.clip(-1 + _EPS, 1 - _EPS)
        z = np.arctanh(corr.to_numpy())
        np.fill_diagonal(z, np.nan)
        avg_r = np.tanh(np.nanmean(z, axis=1))
        for lang, val in zip(corr.index, avg_r):
            rows.append({
                "model": model, "language": lang, "prompt": prompt,
                "mean_acc_lang": float(lang_mean.loc[lang]),
                "avg_corr_to_others_fisher": float(val) if np.isfinite(val) else np.nan,
            })

    df_lang_prompt = pd.DataFrame(rows)
    if df_lang_prompt.empty:
        return df_lang_prompt

    fisher_summary = _fisher_agg(
        df_lang_prompt,
        group_col=["model", "language"],
        corr_col="avg_corr_to_others_fisher",
        out_col="avg_corr_to_others_fisher_model_lang",
    )
    mean_acc_summary = (
        df_lang_prompt.groupby(["model", "language"], sort=False)
        .agg(mean_acc_lang=("mean_acc_lang", "mean"))
        .reset_index()
    )
    return mean_acc_summary.merge(fisher_summary, on=["model", "language"], how="left")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NO_ANSWER_EXTRACTORS = frozenset({
    "empty",
    "no_digit_gate",
    "unfinished_operation_repetition",
    "prompt_repetition_drift",
    "unresolved_operation_loop",
    "other_repetition_drift",
})

ANSWER_TYPES = ["ANSWER-RIGHT", "ANSWER-WRONG", "NO-ANSWER", "UNKNOWN"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def iter_records(base_results_dir: str) -> Iterator[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Yield (model, language, prompt, record) for every record in the results folder.

    Folder layout: base_results_dir / <model_dir> / <language> / <prompt>.jsonl
    Model name is derived as model_dir.split("__")[-1].
    """
    for model_entry in sorted(os.scandir(base_results_dir), key=lambda e: e.name):
        if not model_entry.is_dir():
            continue
        model = model_entry.name.split("__")[-1]
        for lang_entry in sorted(os.scandir(model_entry.path), key=lambda e: e.name):
            if not lang_entry.is_dir():
                continue
            language = lang_entry.name
            for file_entry in sorted(os.scandir(lang_entry.path), key=lambda e: e.name):
                if not file_entry.name.endswith(".jsonl"):
                    continue
                prompt = os.path.splitext(file_entry.name)[0]
                with open(file_entry.path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield model, language, prompt, json.loads(line)


# ---------------------------------------------------------------------------
# Answer-type classification
# ---------------------------------------------------------------------------

def classify_answer(record: Dict[str, Any]) -> str:
    """Return one of: ANSWER-RIGHT, ANSWER-WRONG, NO-ANSWER, UNKNOWN."""
    if record.get("contains_gold_label") is True:
        return "ANSWER-RIGHT"
    if record.get("extracted_answer"):      # non-empty list → has answer but wrong
        return "ANSWER-WRONG"
    extractor = record.get("extractor")
    if extractor == "unknown" or extractor == "unfinished_think":
        return "UNKNOWN"
    # check if extractor is a known no-answer extractor (e.g. "empty", "no_digit_gate", etc.)
    if extractor in NO_ANSWER_EXTRACTORS:
        return "NO-ANSWER"
    print(f"Warning: record with unexpected extractor '{extractor}'.")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def build_flat_records(base_results_dir: str):
    """
    Return a list of dicts, one per (model, language, prompt, problem_id).
    Keys: model, language, prompt, id, problem_type, answer_type, correct (0/1).
    """
    rows = []
    for model, language, prompt, rec in iter_records(base_results_dir):
        answer_type = classify_answer(rec)
        rows.append({
            "model": model,
            "language": language,
            "prompt": prompt,
            "id": rec.get("id", ""),
            "type": rec.get("type", ""),
            "answer_type": answer_type,
            "correct": 1 if answer_type == "ANSWER-RIGHT" else 0,
        })
    return rows


# ---------------------------------------------------------------------------
# Table 1: mean ± stdev accuracy  (model × language pivot)
# ---------------------------------------------------------------------------

def build_accuracy_mean_stdev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must have columns: model, language, prompt, correct.

    Steps:
      1. Accuracy per (model, language, prompt).
      2. Mean ± stdev across prompts → one cell per (model, language).
      3. Add GLOBAL column (mean ± stdev across all language-prompt combos).
      4. Pivot to model × language.
    """
    # Step 1: per (model, language, prompt) accuracy
    per_prompt = (
        df.groupby(["model", "language", "prompt"], sort=False)["correct"]
        .mean()
        .reset_index(name="accuracy")
    )

    # Step 2: mean ± stdev across prompts
    per_lang = (
        per_prompt.groupby(["model", "language"], sort=False)["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_lang["std"] = per_lang["std"].fillna(0.0)
    per_lang["value"] = per_lang.apply(
        lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1
    )

    # Step 3: GLOBAL per model
    global_stats = (
        per_prompt.groupby("model", sort=False)["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    global_stats["std"] = global_stats["std"].fillna(0.0)
    global_stats["language"] = "GLOBAL"
    global_stats["value"] = global_stats.apply(
        lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1
    )

    combined = pd.concat(
        [per_lang[["model", "language", "value"]],
         global_stats[["model", "language", "value"]]],
        ignore_index=True,
    )

    # Step 4: pivot
    table = combined.pivot(index="model", columns="language", values="value")
    cols = sorted(c for c in table.columns if c != "GLOBAL")
    if "GLOBAL" in table.columns:
        cols += ["GLOBAL"]
    return table[cols]


# ---------------------------------------------------------------------------
# Table 2: accuracy per (model, language, prompt, type)
# ---------------------------------------------------------------------------

def build_accuracy_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Flat table: model, language, prompt, type, n, accuracy."""
    agg = (
        df.groupby(["model", "language", "prompt", "type"], sort=False)["correct"]
        .agg(n="count", accuracy="mean")
        .reset_index()
    )
    return agg.sort_values(["model", "language", "prompt", "type"])


# ---------------------------------------------------------------------------
# Table 2b: accuracy per (model, language, prompt) collapsed across types
# ---------------------------------------------------------------------------

def build_accuracy_by_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """Flat table: model, language, prompt, n, accuracy (collapsed across problem types)."""
    agg = (
        df.groupby(["model", "language", "prompt"], sort=False)["correct"]
        .agg(n="count", accuracy="mean")
        .reset_index()
    )
    return agg.sort_values(["model", "language", "prompt"])


# ---------------------------------------------------------------------------
# Table 3: answer-type counts per (model, language, prompt)
# ---------------------------------------------------------------------------

def build_answer_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flat table with counts and percentages of each answer type
    per (model, language, prompt).
    """
    counts = (
        df.groupby(["model", "language", "prompt", "answer_type"], sort=False)
        .size()
        .reset_index(name="count")
    )
    # Pivot so each answer type becomes a column
    wide = counts.pivot_table(
        index=["model", "language", "prompt"],
        columns="answer_type",
        values="count",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    # Ensure all 4 columns exist even if some are absent in data
    for col in ANSWER_TYPES:
        if col not in wide.columns:
            wide[col] = 0

    # Rename to snake_case
    rename = {
        "ANSWER-RIGHT": "answer_right",
        "ANSWER-WRONG": "answer_wrong",
        "NO-ANSWER":    "no_answer",
        "UNKNOWN":      "unknown",
    }
    wide = wide.rename(columns=rename)

    wide["n"] = wide[["answer_right", "answer_wrong", "no_answer", "unknown"]].sum(axis=1)

    for col, pct in [("answer_right", "answer_right_pct"),
                     ("answer_wrong", "answer_wrong_pct"),
                     ("no_answer",    "no_answer_pct"),
                     ("unknown",      "unknown_pct")]:
        wide[pct] = (wide[col] / wide["n"]).round(4)

    col_order = [
        "model", "language", "prompt", "n",
        "answer_right", "answer_wrong", "no_answer", "unknown",
        "answer_right_pct", "answer_wrong_pct", "no_answer_pct", "unknown_pct",
    ]
    return wide[col_order].sort_values(["model", "language", "prompt"])


# ---------------------------------------------------------------------------
# Table 3b: answer-type counts per (model, language, prompt, type)
# ---------------------------------------------------------------------------

def build_answer_types_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flat table with accuracy, counts and percentages of each answer type
    per (model, language, prompt, type).  Mirrors build_answer_types() but
    adds the 'type' dimension and an accuracy column.
    """
    counts = (
        df.groupby(["model", "language", "prompt", "type", "answer_type"], sort=False)
        .size()
        .reset_index(name="count")
    )
    wide = counts.pivot_table(
        index=["model", "language", "prompt", "type"],
        columns="answer_type",
        values="count",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    for col in ANSWER_TYPES:
        if col not in wide.columns:
            wide[col] = 0

    rename = {
        "ANSWER-RIGHT": "answer_right",
        "ANSWER-WRONG": "answer_wrong",
        "NO-ANSWER":    "no_answer",
        "UNKNOWN":      "unknown",
    }
    wide = wide.rename(columns=rename)

    wide["n"] = wide[["answer_right", "answer_wrong", "no_answer", "unknown"]].sum(axis=1)
    wide["accuracy"] = (wide["answer_right"] / wide["n"]).round(4)

    for col, pct in [("answer_right", "answer_right_pct"),
                     ("answer_wrong", "answer_wrong_pct"),
                     ("no_answer",    "no_answer_pct"),
                     ("unknown",      "unknown_pct")]:
        wide[pct] = (wide[col] / wide["n"]).round(4)

    col_order = [
        "model", "language", "prompt", "type", "n", "accuracy",
        "answer_right", "answer_wrong", "no_answer", "unknown",
        "answer_right_pct", "answer_wrong_pct", "no_answer_pct", "unknown_pct",
    ]
    return wide[col_order].sort_values(["model", "language", "prompt", "type"])


# ---------------------------------------------------------------------------
# Table 4: per-language consistency  (XC and PC, input for delta heatmap)
# ---------------------------------------------------------------------------

def build_consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per (model, language) cross-lingual consistency (XC) and prompt
    consistency (PC).

    Output columns match what plot_delta_heatmap() and summarize_delta_from_csv()
    in helper_graphics.py expect:
      model, language, mean_acc_lang,
      avg_corr_to_others_fisher_model_lang,
      prompt_consistency_fisher_overall
    """
    scores = df[["model", "language", "prompt", "id", "correct"]].rename(
        columns={"correct": "score"}
    )
    scores["score"] = scores["score"].astype("float32")
    for col in ["model", "language", "prompt"]:
        scores[col] = scores[col].astype("category")

    per_lang = _per_lang_consistency(scores)
    prompt_summary = _prompt_consistency_summary(scores)

    result = per_lang.merge(
        prompt_summary[["model", "language", "prompt_consistency_fisher"]].rename(
            columns={"prompt_consistency_fisher": "prompt_consistency_fisher_overall"}
        ),
        on=["model", "language"],
        how="left",
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--out-dir", required=True, help="Path to output directory")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip consistency computation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load and preprocess data
    flat_records = build_flat_records(args.results_dir)
    df = pd.DataFrame(flat_records)

    # --- Table 1 ---
    print("Building accuracy_mean_stdev.csv ...")
    t1 = build_accuracy_mean_stdev(df)
    out1 = os.path.join(args.out_dir, "accuracy_mean_stdev.csv")
    t1.to_csv(out1)
    print(f"  Saved: {out1}")

    # --- Table 2 ---
    print("Building accuracy_by_type.csv ...")
    t2 = build_accuracy_by_type(df)
    out2 = os.path.join(args.out_dir, "accuracy_by_type.csv")
    t2.to_csv(out2, index=False)
    print(f"  Saved: {out2}")

    # --- Table 2b ---
    print("Building accuracy_by_prompt.csv ...")
    t2b = build_accuracy_by_prompt(df)
    out2b = os.path.join(args.out_dir, "accuracy_by_prompt.csv")
    t2b.to_csv(out2b, index=False)
    print(f"  Saved: {out2b}")

    # --- Table 3 ---
    print("Building answer_types.csv ...")
    t3 = build_answer_types(df)
    out3 = os.path.join(args.out_dir, "answer_types.csv")
    t3.to_csv(out3, index=False)
    print(f"  Saved: {out3}")

    # --- Table 3b ---
    print("Building answer_types_by_type.csv ...")
    t3b = build_answer_types_by_type(df)
    out3b = os.path.join(args.out_dir, "answer_types_by_type.csv")
    t3b.to_csv(out3b, index=False)
    print(f"  Saved: {out3b}")

    # --- Table 4 ---
    if not args.skip_consistency:
        print("Building per_language_consistency.csv ...")
        t4 = build_consistency_report(df)
        out4 = os.path.join(args.out_dir, "per_language_consistency.csv")
        t4.to_csv(out4, index=False)
        print(f"  Saved: {out4}")


if __name__ == "__main__":
    main()
