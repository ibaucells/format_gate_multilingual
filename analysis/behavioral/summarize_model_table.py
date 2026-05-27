"""
summarize_model_table.py

Reads summarized_results/accuracy_by_prompt.csv and summarized_results/answer_types.csv
and prints a table with one row per model:

  model | mean acc | stdev-lang | stdev-prompt | empty acc | fake acc (stdev) | NO-ANSWER rate

Definitions
-----------
  mean acc      : mean accuracy across all real (lang, prompt) pairs
  stdev-lang    : stdev of per-language means (each lang mean = avg over prompts),
                  computed over real languages only
  stdev-prompt  : stdev of per-prompt means (each prompt mean = avg over real langs)
  empty acc     : mean accuracy for language="empty" across all prompts
  fake acc      : mean ± stdev of accuracy for language="fake" across prompts
  NO-ANSWER rate: fraction of responses classified as NO-ANSWER (all langs, all prompts)

"Real" languages: all languages except "empty" and "fake".

Usage
-----
  python summarize_model_table.py --summarized-dir /path/to/summarized_results --out table.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

CONTROL_LANGS = {"empty", "fake"}


def build_table(acc_df: pd.DataFrame, ans_df: pd.DataFrame) -> pd.DataFrame:
    models = acc_df["model"].unique()
    rows = []

    for model in sorted(models):
        m_acc = acc_df[acc_df["model"] == model]
        m_ans = ans_df[ans_df["model"] == model]

        real = m_acc[~m_acc["language"].isin(CONTROL_LANGS)]

        # ── mean acc (weighted by n) ──────────────────────────────────────────
        total_n    = real["n"].sum()
        mean_acc   = (real["accuracy"] * real["n"]).sum() / total_n if total_n > 0 else np.nan

        # ── stdev-lang: stdev of per-language weighted means ──────────────────
        lang_means = (
            real.groupby("language")
            .apply(lambda g: (g["accuracy"] * g["n"]).sum() / g["n"].sum())
        )
        stdev_lang = float(lang_means.std(ddof=1)) if len(lang_means) > 1 else np.nan

        # ── stdev-prompt: stdev of per-prompt weighted means ──────────────────
        prompt_means = (
            real.groupby("prompt")
            .apply(lambda g: (g["accuracy"] * g["n"]).sum() / g["n"].sum())
        )
        stdev_prompt = float(prompt_means.std(ddof=1)) if len(prompt_means) > 1 else np.nan

        # ── empty acc ─────────────────────────────────────────────────────────
        empty_rows = m_acc[m_acc["language"] == "empty"]
        if not empty_rows.empty:
            empty_acc = (empty_rows["accuracy"] * empty_rows["n"]).sum() / empty_rows["n"].sum()
        else:
            empty_acc = np.nan

        # ── fake acc (mean ± stdev across prompts) ────────────────────────────
        fake_rows = m_acc[m_acc["language"] == "fake"]
        if not fake_rows.empty:
            fake_mean  = (fake_rows["accuracy"] * fake_rows["n"]).sum() / fake_rows["n"].sum()
            fake_stdev = float(fake_rows["accuracy"].std(ddof=1)) if len(fake_rows) > 1 else 0.0
        else:
            fake_mean  = np.nan
            fake_stdev = np.nan

        # ── NO-ANSWER rate ────────────────────────────────────────────────────
        if not m_ans.empty:
            na_count = int(m_ans["no_answer"].sum())
            na_rate  = na_count / m_ans["n"].sum()
        else:
            na_count = np.nan
            na_rate  = np.nan

        # ── top-3 / bottom-3 setups (real langs only) ────────────────────────
        def _fmt_setups(subset):
            return "; ".join(
                f"({r['language']}, {r['prompt']}, {r['accuracy']:.3f})"
                for _, r in subset.iterrows()
            )

        sorted_real = m_acc.sort_values("accuracy", ascending=False)
        top3    = _fmt_setups(sorted_real.head(3))
        bottom3 = _fmt_setups(sorted_real.tail(3).sort_values("accuracy"))

        rows.append({
            "model":           model,
            "mean_acc":        mean_acc,
            "stdev_lang":      stdev_lang,
            "stdev_prompt":    stdev_prompt,
            "empty_acc":       empty_acc,
            "fake_acc_mean":   fake_mean,
            "fake_acc_stdev":  fake_stdev,
            "no_answer_count": na_count,
            "no_answer_rate":  na_rate,
            "top3_setups":     top3,
            "bottom3_setups":  bottom3,
        })

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display["mean_acc"]       = display["mean_acc"].map("{:.3f}".format)
    display["stdev_lang"]     = display["stdev_lang"].map("{:.3f}".format)
    display["stdev_prompt"]   = display["stdev_prompt"].map("{:.3f}".format)
    display["empty_acc"]      = display["empty_acc"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    display["fake_acc"]       = display.apply(
        lambda r: f"{r['fake_acc_mean']:.3f} ± {r['fake_acc_stdev']:.3f}"
                  if pd.notna(r["fake_acc_mean"]) else "—",
        axis=1,
    )
    display["no_answer_count"] = display["no_answer_count"].map(
        lambda x: str(int(x)) if pd.notna(x) else "—"
    )
    display["no_answer_rate"] = display["no_answer_rate"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    out = display[["model", "mean_acc", "stdev_lang", "stdev_prompt",
                   "empty_acc", "fake_acc", "no_answer_count", "no_answer_rate",
                   "top3_setups", "bottom3_setups"]]
    out.columns = ["model", "mean acc", "stdev-lang", "stdev-prompt",
                   "empty acc", "fake acc (stdev)", "NO-ANSWER count", "NO-ANSWER rate",
                   "top-3 setups (lang,prompt,acc)", "bottom-3 setups (lang,prompt,acc)"]
    return out.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to save the table as CSV.")
    ap.add_argument(
        "--summarized-dir",
        type=Path,
        required=True,
        help="Path to directory containing accuracy_by_prompt.csv and answer_types.csv",
    )
    args = ap.parse_args()

    acc_csv = args.summarized_dir / "accuracy_by_prompt.csv"
    ans_csv = args.summarized_dir / "answer_types.csv"

    for p in (acc_csv, ans_csv):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    acc_df = pd.read_csv(acc_csv)
    ans_df = pd.read_csv(ans_csv)

    table = build_table(acc_df, ans_df)
    print(format_table(table))

    # ── NO-ANSWER breakdown by (lang, prompt) per model ───────────────────────
    print("\n\n=== NO-ANSWER breakdown by (language, prompt) ===")
    for model in sorted(ans_df["model"].unique()):
        m_ans = ans_df[ans_df["model"] == model]
        breakdown = (
            m_ans[m_ans["no_answer"] > 0][["language", "prompt", "no_answer", "n"]]
            .copy()
            .sort_values("no_answer", ascending=False)
        )
        if breakdown.empty:
            print(f"\n{model}: no no-answers")
            continue
        breakdown["rate"] = (breakdown["no_answer"] / breakdown["n"]).map("{:.3f}".format)
        breakdown["no_answer"] = breakdown["no_answer"].astype(int)
        breakdown["n"] = breakdown["n"].astype(int)
        print(f"\n{model}:")
        print(breakdown.to_string(index=False))

    if args.out:
        # Save raw numeric version
        table.to_csv(args.out, index=False, float_format="%.4f")
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
