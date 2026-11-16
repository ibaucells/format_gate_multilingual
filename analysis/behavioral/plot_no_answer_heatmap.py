#!/usr/bin/env python3
"""
plot_no_answer_heatmap.py

Reads summarized_results/answer_types_by_type.csv and plots a heatmap of
NO-ANSWER rate with language on the Y-axis and problem type on the X-axis.

One or more models can be given: each model becomes a side-by-side panel in
a single figure, sharing the same colour scale so cells are directly comparable.
Language rows are ordered by descending overall NO-ANSWER rate (aggregated
across all specified models so the order is shared across panels).

Usage
-----
# Single model, all prompts aggregated:
python plot_no_answer_heatmap.py \
    --csv summarized_results/answer_types_by_type.csv \
    --model Qwen3-4B-Base \
    --all-prompts \
    --output plots/

# Three models side by side:
python plot_no_answer_heatmap.py \
    --csv summarized_results/answer_types_by_type.csv \
    --model Qwen3-0.6B-Base Qwen3-4B-Base salamandra-7b \
    --all-prompts \
    --output plots/

# Single prompt instead of aggregating:
python plot_no_answer_heatmap.py \
    --csv summarized_results/answer_types_by_type.csv \
    --model Qwen3-4B-Base \
    --prompt answer \
    --output plots/

# Multiple models, single prompt:
python plot_no_answer_heatmap.py \
    --csv summarized_results/answer_types_by_type.csv \
    --model gemma-3-4b-pt Meta-Llama-3.2-3B Qwen3-1.7B-Base \
    --all-prompts \
    --output plots/

# X=prompt, Y=language (aggregate over all problem types):
python plot_no_answer_heatmap.py \
    --csv summarized_results/answer_types_by_type.csv \
    --model Qwen3-4B-Base \
    --by-prompt \
    --output plots/

python plot_no_answer_heatmap.py \
    --csv summarized_results/answer_types_by_type.csv \
    --model gemma-3-4b-pt Meta-Llama-3.2-3B Qwen3-1.7B-Base \
    --by-prompt \
    --output plots/

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TYPE_ORDER = ["sum", "sub", "mul", "div", "mix", "eq"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_model_data(df: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = df[df["model"] == model]
    if subset.empty:
        available = sorted(df["model"].unique())
        raise ValueError(
            f"Model '{model}' not found in CSV.\n"
            f"Available models: {available}"
        )
    return subset


def aggregate_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Sum raw counts over all prompts, then recompute rates."""
    agg = (
        df.groupby(["model", "language", "type"], sort=False)[
            ["n", "answer_right", "answer_wrong", "no_answer", "unknown"]
        ]
        .sum()
        .reset_index()
    )
    total = agg["n"]
    agg["answer_right_pct"] = (agg["answer_right"] / total).round(4)
    agg["answer_wrong_pct"] = (agg["answer_wrong"] / total).round(4)
    agg["no_answer_pct"]    = (agg["no_answer"]    / total).round(4)
    agg["unknown_pct"]      = (agg["unknown"]       / total).round(4)
    agg["accuracy"]         = agg["answer_right_pct"]
    return agg


def aggregate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Sum raw counts over all problem types, then recompute rates (keeps prompts distinct)."""
    agg = (
        df.groupby(["model", "language", "prompt"], sort=False)[
            ["n", "answer_right", "answer_wrong", "no_answer", "unknown"]
        ]
        .sum()
        .reset_index()
    )
    total = agg["n"]
    agg["answer_right_pct"] = (agg["answer_right"] / total).round(4)
    agg["answer_wrong_pct"] = (agg["answer_wrong"] / total).round(4)
    agg["no_answer_pct"]    = (agg["no_answer"]    / total).round(4)
    agg["unknown_pct"]      = (agg["unknown"]       / total).round(4)
    agg["accuracy"]         = agg["answer_right_pct"]
    return agg


def filter_prompt(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    subset = df[df["prompt"] == prompt]
    if subset.empty:
        available = sorted(df["prompt"].unique())
        raise ValueError(
            f"Prompt '{prompt}' not found.\n"
            f"Available prompts: {available}"
        )
    return subset


def lang_order_by_no_answer(model_data: dict) -> list:
    """Languages sorted by descending mean NO-ANSWER rate, averaged equally across models."""
    per_model_rates = []
    for df in model_data.values():
        rates = (
            df.groupby("language")[["no_answer", "n"]]
            .sum()
            .assign(rate=lambda d: d["no_answer"] / d["n"])["rate"]
        )
        per_model_rates.append(rates)
    # Unweighted mean across models so each model contributes equally
    mean_rates = pd.concat(per_model_rates, axis=1).mean(axis=1)
    return list(mean_rates.sort_values(ascending=False).index)


def build_matrix(df: pd.DataFrame, value_col: str,
                 lang_order: list, col_order_hint: list,
                 col_dim: str = "type") -> pd.DataFrame:
    """Pivot to (language × col_dim) matrix — language on rows (Y), col_dim on columns (X)."""
    pivot = df.pivot_table(
        index="language",
        columns=col_dim,
        values=value_col,
        aggfunc="mean",
    )
    pivot.columns.name = None
    pivot.index.name = None
    # Preferred order first, remaining appended sorted
    col_order = [t for t in col_order_hint if t in pivot.columns] + \
                [t for t in sorted(pivot.columns) if t not in col_order_hint]
    # Languages: shared order derived from all-models data
    row_order = [l for l in lang_order if l in pivot.index]
    return pivot.reindex(index=row_order, columns=col_order)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_annot_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """All cells annotated as plain integers (e.g. 45 for 45%), no % sign."""
    return matrix.map(lambda v: f"{v * 100:.0f}" if pd.notna(v) else "")


def draw_panel(matrix: pd.DataFrame, ax: plt.Axes, title: str,
               cmap: str, show_ylabel: bool, xlabel: str = "Problem type"):
    """
    Draw one heatmap panel.  Colours are normalised to [0, 1] relative to this
    panel's own maximum so the full colour range is used per model.
    Annotations show the original percentage values as plain integers.
    """
    panel_max = float(matrix.max().max())
    norm_matrix = matrix / panel_max if panel_max > 0 else matrix
    annot = build_annot_matrix(matrix)
    sns.heatmap(
        norm_matrix,
        ax=ax,
        annot=annot,
        fmt="",
        annot_kws={"size": 7},
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar=False,
        yticklabels=show_ylabel,
    )
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Language" if show_ylabel else "", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)


def make_figure(matrices: list, titles: list, out_path: Path, cmap: str,
                xlabel: str = "Problem type"):
    """Create a 1×N figure with one panel per model and a single shared colorbar."""
    n = len(matrices)
    n_rows, n_cols = matrices[0].shape
    panel_w = max(2.5, n_cols * 0.75)
    panel_h = max(3.0, n_rows * 0.55)
    label_w = 1.2   # space for Y-axis labels on the first panel
    fig, axes = plt.subplots(
        1, n,
        figsize=(panel_w * n + label_w, panel_h),
    )
    if n == 1:
        axes = [axes]

    for i, (mat, title, ax) in enumerate(zip(matrices, titles, axes)):
        draw_panel(mat, ax, title, cmap=cmap, show_ylabel=(i == 0), xlabel=xlabel)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Heatmap of NO-ANSWER rate: language (Y) × problem type (X)."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to answer_types_by_type.csv")
    parser.add_argument("--model", required=True, nargs="+",
                        help="One or more model names (each becomes a panel)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-prompts", action="store_true",
                       help="Aggregate counts across all prompts before plotting (X=type)")
    group.add_argument("--prompt",
                       help="Name of a single prompt to plot (e.g. 'answer') (X=type)")
    group.add_argument("--by-prompt", action="store_true",
                       help="Aggregate counts across all problem types; plot X=prompt, Y=language")
    parser.add_argument("--output", default=".",
                        help="Output directory for the PDF (default: current dir)")
    parser.add_argument("--also-accuracy", action="store_true",
                        help="Also save an accuracy (ANSWER-RIGHT rate) heatmap")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_csv(csv_path)

    if args.all_prompts:
        prompt_label = "all-prompts"
    elif args.by_prompt:
        prompt_label = "by-prompt"
    else:
        prompt_label = args.prompt

    # --- Prepare per-model data frames ---
    model_data = {}
    for model in args.model:
        df_model = get_model_data(all_df, model)
        if args.all_prompts:
            df_model = aggregate_prompts(df_model)
        elif args.by_prompt:
            df_model = aggregate_types(df_model)
        else:
            df_model = filter_prompt(df_model, args.prompt)
        model_data[model] = df_model

    # Language order: unweighted mean NO-ANSWER rate per language across models
    langs_sorted = lang_order_by_no_answer(model_data)

    safe_models = "_vs_".join(m.replace("/", "_") for m in args.model)

    # --- NO-ANSWER rate figure ---
    if args.by_prompt:
        na_matrices = [
            build_matrix(model_data[m], "no_answer_pct", langs_sorted,
                         col_order_hint=[], col_dim="prompt")
            for m in args.model
        ]
    else:
        na_matrices = [
            build_matrix(model_data[m], "no_answer_pct", langs_sorted,
                         col_order_hint=TYPE_ORDER, col_dim="type")
            for m in args.model
        ]
    na_vmaxes = [max(float(mat.max().max()), 0.01) for mat in na_matrices]
    na_titles  = [
        f"{m}  (max: {vmax:.0%})" for m, vmax in zip(args.model, na_vmaxes)
    ]
    x_label = "Prompt" if args.by_prompt else "Problem type"
    make_figure(
        na_matrices,
        titles=na_titles,
        out_path=out_dir / f"{safe_models}_{prompt_label}_no_answer_rate.pdf",
        cmap="OrRd",
        xlabel=x_label,
    )

    # --- optional accuracy figure ---
    if args.also_accuracy:
        if args.by_prompt:
            acc_matrices = [
                build_matrix(model_data[m], "accuracy", langs_sorted,
                             col_order_hint=[], col_dim="prompt")
                for m in args.model
            ]
        else:
            acc_matrices = [
                build_matrix(model_data[m], "accuracy", langs_sorted,
                             col_order_hint=TYPE_ORDER, col_dim="type")
                for m in args.model
            ]
        acc_vmaxes = [max(float(mat.max().max()), 0.01) for mat in acc_matrices]
        acc_titles  = [
            f"{m}  (max: {vmax:.0%})" for m, vmax in zip(args.model, acc_vmaxes)
        ]
        make_figure(
            acc_matrices,
            titles=acc_titles,
            out_path=out_dir / f"{safe_models}_{prompt_label}_accuracy.pdf",
            cmap="YlGn",
            xlabel=x_label,
        )

    print("Done.")


if __name__ == "__main__":
    main()
