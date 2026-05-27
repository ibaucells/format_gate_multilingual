#!/usr/bin/env python3
"""
Read the cosine-similarity parquet produced by compute_cosine_similarities.py
and write aggregated CSVs to an output directory.

Aggregations produced
---------------------
1. by_condition_pair.csv
      (lang1, lang2, prompt1, prompt2, layer)  → avg over items

2. by_behavior_pair.csv
      (behavior1, behavior2, layer)             → avg over items × condition pairs

3. by_lang_pair.csv
      (lang1, lang2, layer)                     → avg over items × prompts

4. by_prompt_pair.csv
      (prompt1, prompt2, layer)                 → avg over items × langs

5. by_lang_behavior.csv
      (lang1, lang2, behavior1, behavior2, layer)

6. by_prompt_behavior.csv
      (prompt1, prompt2, behavior1, behavior2, layer)

7. by_sameflag_behavior.csv
      (same_lang, same_prompt, behavior1, behavior2, layer)

Each table has columns: cos_sim_mean, cos_sim_std, n_pairs

Usage
-----
  python aggregate_cosine_similarities.py \\
    --input  cosine_sims/Qwen3-0.6B-Base.parquet \\
    --output cosine_sims_agg/Qwen3-0.6B-Base/ \\
    --curves
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def agg_groupby(df: pd.DataFrame, groupby_cols: list) -> pd.DataFrame:
    g = (
        df.groupby(groupby_cols, observed=True)["cos_sim"]
        .agg(cos_sim_mean="mean", cos_sim_std="std", n_pairs="count")
        .reset_index()
    )
    g["cos_sim_std"] = g["cos_sim_std"].fillna(0.0)
    return g


def save(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"  wrote {len(df):>10,} rows  ->  {path}")


def plot_condition_heatmaps(
    df_raw: pd.DataFrame,
    output_dir: Path,
    layers: list = None,
    step: int = 5,
    start: int = 1,
    vmin: float = None,
    vmax: float = None,
    title_prefix: str = "",
):
    """
    For each selected layer, build a (lang_prompt) × (lang_prompt) heatmap of
    mean cosine similarity and save it as a PNG.

    Parameters
    ----------
    df_raw      : raw parquet dataframe (item_id, lang1, lang2, prompt1, prompt2,
                  layer, cos_sim, behavior1, behavior2)
    output_dir  : directory where PNGs are written
    layers      : explicit list of layer indices to plot; if None, uses
                  range(start, max_layer+1, step)
    step        : spacing when auto-generating layer list (default 5)
    start       : first layer when auto-generating (default 1)
    vmin/vmax   : colour-scale limits (shared across all layer plots)
    title_prefix: prepended to each figure title (e.g. model name)
    """
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Build condition labels: "lang__prompt"
    df_raw = df_raw.copy()
    df_raw["cond1"] = df_raw["lang1"].astype(str) + "__" + df_raw["prompt1"].astype(str)
    df_raw["cond2"] = df_raw["lang2"].astype(str) + "__" + df_raw["prompt2"].astype(str)

    # Sorted list of all conditions
    all_conds = sorted(set(df_raw["cond1"].unique()) | set(df_raw["cond2"].unique()))
    n = len(all_conds)
    idx = {c: i for i, c in enumerate(all_conds)}

    # Select layers
    max_layer = int(df_raw["layer"].max())
    if layers is None:
        layers = list(range(start, max_layer + 1, step))

    # Pre-aggregate: mean cos_sim per (cond1, cond2, layer)
    agg = (
        df_raw.groupby(["cond1", "cond2", "layer"], observed=True)["cos_sim"]
        .mean()
        .reset_index()
    )

    # Shared colour scale across all selected layers
    if vmin is None:
        vmin = float(agg["cos_sim"].min())
    if vmax is None:
        vmax = float(agg["cos_sim"].max())

    for layer in layers:
        layer_df = agg[agg["layer"] == layer]
        if layer_df.empty:
            continue

        # Fill symmetric matrix
        mat = np.full((n, n), np.nan)
        for _, row in layer_df.iterrows():
            i, j = idx.get(row["cond1"]), idx.get(row["cond2"])
            if i is not None and j is not None:
                mat[i, j] = row["cos_sim"]
                mat[j, i] = row["cos_sim"]   # mirror
        np.fill_diagonal(mat, 1.0)

        fig, ax = plt.subplots(figsize=(max(10, n // 3), max(8, n // 3)))
        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=all_conds,
            yticklabels=all_conds,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            square=True,
            linewidths=0,
        )
        title = f"{title_prefix}  layer {layer}".strip()
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=6, rotation=90)
        ax.tick_params(axis="y", labelsize=6, rotation=0)
        plt.tight_layout()

        out_png = heatmap_dir / f"layer_{layer:03d}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  heatmap layer {layer:>3d}  ->  {out_png}")


def plot_lang_heatmaps(
    csv_condition_pair: Path,
    output_dir: Path,
    prompt: str = "answer",
    layers: list = None,
    step: int = 5,
    start: int = 1,
    vmin: float = None,
    vmax: float = None,
    title_prefix: str = "",
):
    """
    Build a lang × lang heatmap of mean cosine similarity for a fixed prompt,
    for selected layers. Reads from by_condition_pair.csv.

    Parameters
    ----------
    csv_condition_pair : path to by_condition_pair.csv
    output_dir         : directory where PNGs are written
    prompt             : fix both prompt1 and prompt2 to this value (e.g. "answer")
    layers             : explicit layer list; if None uses range(start, max+1, step)
    step / start       : auto layer selection params
    vmin / vmax        : shared colour-scale limits (auto if None)
    title_prefix       : prepended to figure title (e.g. model name)
    """
    heatmap_dir = output_dir / f"heatmaps_lang_{prompt}"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_condition_pair)

    # Filter to the chosen prompt, but "empty" language always uses "answer"
    mask = (
        ((df["lang1"] != "empty") & (df["prompt1"] == prompt) |
         (df["lang1"] == "empty") & (df["prompt1"] == "answer")) &
        ((df["lang2"] != "empty") & (df["prompt2"] == prompt) |
         (df["lang2"] == "empty") & (df["prompt2"] == "answer"))
    )
    df = df[mask].copy()
    if df.empty:
        raise ValueError(f"No rows found for prompt='{prompt}' in {csv_condition_pair}")

    all_langs = sorted(set(df["lang1"].unique()) | set(df["lang2"].unique()))
    n = len(all_langs)
    idx = {l: i for i, l in enumerate(all_langs)}

    max_layer = int(df["layer"].max())
    if layers is None:
        layers = list(range(start, max_layer + 1, step))

    if vmin is None:
        vmin = float(df["cos_sim_mean"].min())
    if vmax is None:
        vmax = float(df["cos_sim_mean"].max())

    for layer in layers:
        layer_df = df[df["layer"] == layer]
        if layer_df.empty:
            continue

        mat = np.full((n, n), np.nan)
        for _, row in layer_df.iterrows():
            i, j = idx.get(row["lang1"]), idx.get(row["lang2"])
            if i is not None and j is not None:
                mat[i, j] = row["cos_sim_mean"]
                mat[j, i] = row["cos_sim_mean"]
        np.fill_diagonal(mat, 1.0)

        fig, ax = plt.subplots(figsize=(n * 0.6 + 2, n * 0.6 + 1.5))
        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=all_langs,
            yticklabels=all_langs,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            square=True,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7},
            linewidths=0.3,
        )
        title = f"{title_prefix}  prompt={prompt}  layer {layer}".strip()
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=9, rotation=45)
        ax.tick_params(axis="y", labelsize=9, rotation=0)
        plt.tight_layout()

        out_png = heatmap_dir / f"layer_{layer:03d}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  lang heatmap (prompt={prompt}) layer {layer:>3d}  ->  {out_png}")


def _bootstrap_ci(item_layer_mat: np.ndarray, n_bootstrap: int = 1000, seed: int = 42):
    """
    Bootstrap 95% CI over items.
    The bootstrap CI measures uncertainty in the mean, not the spread of individual observations. 
    It resamples with replacement from the existing items to create "new" datasets, computes the mean for each, and uses the distribution of these means to estimate the confidence interval.

    item_layer_mat : (n_items, n_layers) — per-item mean cos_sim per layer
    Returns (ci_low, ci_high) each shape (n_layers,)
    """
    rng = np.random.default_rng(seed)
    n_items = item_layer_mat.shape[0]
    idx = rng.integers(0, n_items, size=(n_bootstrap, n_items))
    boot_means = np.nanmean(item_layer_mat[idx], axis=1)   # (n_bootstrap, n_layers)
    ci_low  = np.nanpercentile(boot_means, 2.5,  axis=0)
    ci_high = np.nanpercentile(boot_means, 97.5, axis=0)
    return ci_low, ci_high


def plot_lang_prompt_curves(
    df_raw: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
    n_bootstrap: int = 1000,
):
    """
    3-curve plot with 95% bootstrapped CIs (resampling at item level):
      - Within-language, different-prompt  (lang1==lang2, prompt1!=prompt2)
      - Different-language, same-prompt    (lang1!=lang2, prompt1==prompt2)
      - Different-language, different-prompt (lang1!=lang2, prompt1!=prompt2)
    Saves as curves_lang_prompt.png.
    """
    # Filtering: Removes rows where lang1 or lang2 is "empty" or "fake".
    _exclude = {"empty", "fake"}
    df = df_raw[
        ~df_raw["lang1"].astype(str).isin(_exclude) &
        ~df_raw["lang2"].astype(str).isin(_exclude)
    ].copy()
    df["same_lang"]   = (df["lang1"].astype(str) == df["lang2"].astype(str))
    df["same_prompt"] = (df["prompt1"].astype(str) == df["prompt2"].astype(str))

    # Grouping: Creates boolean masks for each of the 3 conditions using same_lang and same_prompt flags.
    groups = {
        "Within-lang, diff-prompt":    df["same_lang"]  & ~df["same_prompt"],
        "Diff-lang, same-prompt":      ~df["same_lang"] &  df["same_prompt"],
        "Diff-lang, diff-prompt":      ~df["same_lang"] & ~df["same_prompt"],
    }

    colors = ["#e15759", "#4e79a7", "#76b7b2"]

    # Create figure/axis before plotting so `ax` is available inside loop
    fig, ax = plt.subplots(figsize=(16, 8))

    all_means = []
    all_ci_low = []
    all_ci_high = []

    # Per-item matrix: For each condition, groups by (item_id, layer) and averages cos_sim, producing a matrix of shape (n_items, n_layers). The last layer is dropped.
    for (label, mask), color in zip(groups.items(), colors):
        sub = df[mask]

        # Per-item per-layer mean → (n_items, n_layers) matrix
        item_layer = (
            sub.groupby(["item_id", "layer"], observed=True)["cos_sim"]
            .mean()
            .unstack("layer")       # rows=items, cols=layers
            .sort_index(axis=1)     # ensure layers are sorted
        )

        # Mean + CI: Computes the column-wise mean across items, and calls _bootstrap_ci() to get the confidence bands.
        mat      = item_layer.values[:, :-1].astype(float)  # drop last layer
        layers   = item_layer.columns.tolist()[:-1]
        mean_vec = np.nanmean(mat, axis=0)

        ci_low, ci_high = _bootstrap_ci(mat, n_bootstrap=n_bootstrap)

        # Plot: Draws the mean line and shades the CI band with fill_between.
        ax.plot(layers, mean_vec, label=label, color=color, linewidth=2)
        ax.fill_between(layers, ci_low, ci_high, color=color, alpha=0.4)

        all_means.append(mean_vec)
        all_ci_low.append(ci_low)
        all_ci_high.append(ci_high)

        mid = len(layers) // 2
        ci_width = ci_high[mid] - ci_low[mid]
        print(f"  {label}: {len(item_layer)} items, layers {layers[0]}–{layers[-1]}, "
              f"CI width (mid layer): {ci_width:.6f}")

    # Zoom y-axis to actual data range + 20% padding around spread
    all_low  = np.nanmin([v for v in all_ci_low])
    all_high = np.nanmax([v for v in all_ci_high])
    pad = (all_high - all_low) * 0.2
    ax.set_ylim(all_low - pad, all_high + pad)

    # Remove title and increase font sizes for readability
    ax.set_xlabel("Layer", fontsize=24)
    ax.set_ylabel("Mean cosine similarity", fontsize=24)
    # No title (user requested it removed)
    # ax.set_title(f"{title_prefix}  cosine similarity by condition type".strip())
    ax.legend(fontsize=18, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.tight_layout()

    # Include the model name (title_prefix) in the filename when provided
    if title_prefix:
        safe_prefix = str(title_prefix).replace("/", "_").replace(" ", "_")
        out_png = output_dir / f"curves_lang_prompt_{safe_prefix}.png"
    else:
        out_png = output_dir / "curves_lang_prompt.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  curves plot  ->  {out_png}")


def plot_behavior_pair_curves(
    df_raw: pd.DataFrame,
    output_dir: Path,
    beh_set_a: set,
    beh_set_b: set,
    out_filename: str,
    ylabel: str,
    plot_title: str,
    title_prefix: str = "",
    n_bootstrap: int = 1000,
    seed: int = 42,
):
    """
    For each language (excluding 'empty' and 'fake'), for each item:
      - Collect all same-language pairs where one side has a behavior in beh_set_a
        and the other in beh_set_b.
      - Sample one such pair per item, balancing which prompt appears on the
        beh_set_a side across items (round-robin by least-used prompt).
    Plots cosine similarity vs layer, one curve per language.
    n_items used is shown in each legend entry.
    """
    EXCLUDE_LANGS = {"empty", "fake"}
    rng = np.random.default_rng(seed)

    df = df_raw.copy()
    df["lang1_s"]   = df["lang1"].astype(str)
    df["lang2_s"]   = df["lang2"].astype(str)
    df["beh1_s"]    = df["behavior1"].astype(str)
    df["beh2_s"]    = df["behavior2"].astype(str)
    df["prompt1_s"] = df["prompt1"].astype(str)
    df["prompt2_s"] = df["prompt2"].astype(str)

    # Same-language pairs only (full df, needed for baselines)
    same_all = df[
        df["lang1_s"] == df["lang2_s"]
    ].copy()
    same_all = same_all[~same_all["lang1_s"].isin(EXCLUDE_LANGS)]

    # Keep only rows where one side is in beh_set_a and the other in beh_set_b
    b1_a = same_all["beh1_s"].isin(beh_set_a)
    b2_a = same_all["beh2_s"].isin(beh_set_a)
    b1_b = same_all["beh1_s"].isin(beh_set_b)
    b2_b = same_all["beh2_s"].isin(beh_set_b)
    same = same_all[(b1_a & b2_b) | (b1_b & b2_a)].copy()

    if same.empty:
        print(f"  No {plot_title} same-language pairs found.")
        return

    # Normalise: prompt_a = prompt on the beh_set_a side
    is_b1_a = same["beh1_s"].isin(beh_set_a)
    same["prompt_a"] = np.where(is_b1_a, same["prompt1_s"], same["prompt2_s"])
    same["prompt_b"] = np.where(is_b1_a, same["prompt2_s"], same["prompt1_s"])

    langs = sorted(l for l in same["lang1_s"].unique() if l not in EXCLUDE_LANGS)

    fig,      ax      = plt.subplots(figsize=(12, 6))
    fig_d,    ax_d    = plt.subplots(figsize=(12, 6))
    cmap = plt.colormaps["tab10"].resampled(max(len(langs), 1))
    all_ci_low, all_ci_high = [], []
    all_delta_low, all_delta_high = [], []
    last_layers = None

    for idx, lang in enumerate(langs):
        lang_df  = same[same["lang1_s"] == lang]
        all_df   = same_all[same_all["lang1_s"] == lang]

        pair_idx = lang_df[["item_id", "prompt_a", "prompt_b"]].drop_duplicates()

        # For each item, pick the pair whose prompt_a is least used so far
        prompt_counter: Counter = Counter()
        selected = []
        for item_id in sorted(pair_idx["item_id"].unique()):
            candidates = pair_idx[pair_idx["item_id"] == item_id]
            counts = candidates["prompt_a"].map(lambda p: prompt_counter[p])
            min_c  = counts.min()
            best   = candidates[counts == min_c]
            chosen = best.iloc[int(rng.integers(len(best)))]
            selected.append({
                "item_id":  item_id,
                "prompt_a": chosen["prompt_a"],
                "prompt_b": chosen["prompt_b"],
            })
            prompt_counter[chosen["prompt_a"]] += 1

        sel_df = pd.DataFrame(selected)
        prompt_b_counter = Counter(sel_df["prompt_b"])
        merged = lang_df.merge(sel_df, on=["item_id", "prompt_a", "prompt_b"])
        if merged.empty:
            continue

        item_layer = (
            merged.groupby(["item_id", "layer"], observed=True)["cos_sim"]
            .mean().unstack("layer").sort_index(axis=1)
        )
        n_items  = len(item_layer)
        mat      = item_layer.values[:, :-1].astype(float)
        layers   = item_layer.columns.tolist()[:-1]
        last_layers = layers
        mean_vec = np.nanmean(mat, axis=0)
        ci_low, ci_high = _bootstrap_ci(mat, n_bootstrap=n_bootstrap)

        # Baseline: (within_beh_set_a + within_beh_set_b) / 2 per item
        def _within_item_layer(pool, beh_set):
            sub = pool[(pool["beh1_s"].isin(beh_set)) & (pool["beh2_s"].isin(beh_set))]
            if sub.empty:
                return None
            il = (sub.groupby(["item_id", "layer"], observed=True)["cos_sim"]
                  .mean().unstack("layer").sort_index(axis=1))
            return il.iloc[:, :-1]  # drop last layer

        wa = _within_item_layer(all_df, beh_set_a)
        wb = _within_item_layer(all_df, beh_set_b)

        if wa is None or wb is None:
            d_ci_low = d_ci_high = delta = shared = None
        else:
            shared_layers = [l for l in layers
                             if l in wa.columns and l in wb.columns]
            common_items  = (item_layer.index
                             .intersection(wa.index)
                             .intersection(wb.index))
            shared        = list(shared_layers)
            mat_aligned   = item_layer.loc[common_items, shared_layers].values.astype(float)
            wa_aligned    = wa.loc[common_items, shared_layers].values.astype(float)
            wb_aligned    = wb.loc[common_items, shared_layers].values.astype(float)
            bl_aligned    = (wa_aligned + wb_aligned) / 2.0
            delta         = np.nanmean(bl_aligned, axis=0) - np.nanmean(mat_aligned, axis=0)
            delta_mat     = bl_aligned - mat_aligned
            d_ci_low, d_ci_high = _bootstrap_ci(delta_mat, n_bootstrap=n_bootstrap)

        color = cmap(idx)

        # --- main curves plot ---
        ax.plot(layers, mean_vec, label=f"{lang} (n={n_items})", color=color, linewidth=2)
        ax.fill_between(layers, ci_low, ci_high, color=color, alpha=0.3)
        min_pos = int(np.nanargmin(mean_vec))
        ax.plot(layers[min_pos], mean_vec[min_pos],
                marker="v", color=color, markersize=7, zorder=5)
        ax.annotate(f"L{layers[min_pos]}",
                    xy=(layers[min_pos], mean_vec[min_pos]),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=7, color=color)

        # --- delta plot ---
        if delta is not None:
            ax_d.plot(shared, delta, label=f"{lang} (n={n_items})", color=color, linewidth=2)
            ax_d.fill_between(shared, d_ci_low, d_ci_high, color=color, alpha=0.3)
            max_pos = int(np.nanargmax(delta))
            ax_d.plot(shared[max_pos], delta[max_pos],
                      marker="^", color=color, markersize=7, zorder=5)
            ax_d.annotate(f"L{shared[max_pos]}",
                          xy=(shared[max_pos], delta[max_pos]),
                          xytext=(4, 4), textcoords="offset points",
                          fontsize=7, color=color)
            all_delta_low.append(d_ci_low)
            all_delta_high.append(d_ci_high)
            delta_info = f"| max delta at layer {shared[max_pos]} ({delta[max_pos]:.4f})"
        else:
            delta_info = "| delta: N/A (missing within-behavior pairs)"

        all_ci_low.append(ci_low)
        all_ci_high.append(ci_high)
        print(f"  {lang}: {n_items} items | min at layer {layers[min_pos]} "
              f"({mean_vec[min_pos]:.4f}) {delta_info} | prompt_a: {dict(prompt_counter)} "
              f"| prompt_b: {dict(prompt_b_counter)}")

    # --- finalise main curves plot ---
    if all_ci_low:
        lo  = np.nanmin([v for v in all_ci_low])
        hi  = np.nanmax([v for v in all_ci_high])
        pad = (hi - lo) * 0.2
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}  {plot_title} cosine sim per language".strip())
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    if last_layers:
        ax.set_xticks(last_layers)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.3)
    plt.figure(fig.number)
    plt.tight_layout()
    out_png = output_dir / out_filename
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_filename}  ->  {out_png}")

    # --- finalise delta plot ---
    if all_delta_low:
        lo  = np.nanmin([v for v in all_delta_low])
        hi  = np.nanmax([v for v in all_delta_high])
        pad = max((hi - lo) * 0.2, 1e-4)
        ax_d.set_ylim(lo - pad, hi + pad)
    ax_d.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_d.set_xlabel("Layer")
    ax_d.set_ylabel(f"Δ cosine sim  (baseline − {plot_title})")
    ax_d.set_title(f"{title_prefix}  {plot_title} delta per language".strip())
    ax_d.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    if last_layers:
        ax_d.set_xticks(last_layers)
    ax_d.tick_params(axis="x", labelsize=7)
    ax_d.grid(True, alpha=0.3)
    plt.figure(fig_d.number)
    plt.tight_layout()
    delta_filename = "delta_" + out_filename
    out_delta_png = output_dir / delta_filename
    fig_d.savefig(out_delta_png, dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"  {delta_filename}  ->  {out_delta_png}")


def plot_answer_vs_noanswer_curves(
    df_raw: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
    n_bootstrap: int = 1000,
):
    plot_behavior_pair_curves(
        df_raw, output_dir,
        beh_set_a={"ANSWER-RIGHT", "ANSWER-WRONG"},
        beh_set_b={"NO-ANSWER"},
        out_filename="curves_answer_vs_noanswer.png",
        ylabel="Mean cosine similarity (ANSWER vs NO-ANSWER)",
        plot_title="ANSWER vs NO-ANSWER",
        title_prefix=title_prefix,
        n_bootstrap=n_bootstrap,
    )


def plot_right_vs_wrong_curves(
    df_raw: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
    n_bootstrap: int = 1000,
):
    plot_behavior_pair_curves(
        df_raw, output_dir,
        beh_set_a={"ANSWER-RIGHT"},
        beh_set_b={"ANSWER-WRONG"},
        out_filename="curves_right_vs_wrong.png",
        ylabel="Mean cosine similarity (ANSWER-RIGHT vs ANSWER-WRONG)",
        plot_title="ANSWER-RIGHT vs ANSWER-WRONG",
        title_prefix=title_prefix,
        n_bootstrap=n_bootstrap,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="Input .parquet from compute_cosine_similarities.py")
    parser.add_argument("--output", required=True,
                        help="Output directory for aggregated CSV files")
    parser.add_argument("--heatmap", action="store_true",
                        help="Also generate per-layer condition×condition heatmaps")
    parser.add_argument("--heatmap-step", type=int, default=5,
                        help="Layer step for heatmaps (default: 5)")
    parser.add_argument("--heatmap-start", type=int, default=1,
                        help="First layer for heatmaps (default: 1)")
    parser.add_argument("--curves", action="store_true",
                        help="Generate 3-curve plot: within-lang/diff-prompt, "
                             "diff-lang/same-prompt, diff-lang/diff-prompt vs layer")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for CI (default: 1000)")
    parser.add_argument("--lang-heatmap", metavar="PROMPT",
                        help="Generate lang×lang heatmap for a fixed prompt "
                             "(e.g. --lang-heatmap answer). Uses by_condition_pair.csv.")
    parser.add_argument("--skip-last-layer", action="store_true",
                        help="Exclude the last (highest-index) layer from all outputs")
    parser.add_argument("--answer-vs-noanswer", action="store_true",
                        help="Plot ANSWER vs NO-ANSWER cosine sim curves per language")
    parser.add_argument("--right-vs-wrong", action="store_true",
                        help="Plot ANSWER-RIGHT vs ANSWER-WRONG cosine sim curves per language")
    parser.add_argument("--extractor", default=None,
                        help="Comma-separated extractor values to keep. Retains rows where "
                             "extractor1 OR extractor2 is in the list. "
                             "Requires parquet built with extractor columns.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path} ...")
    df = pd.read_parquet(input_path, engine="pyarrow")

    for col in ["lang1", "lang2", "prompt1", "prompt2", "behavior1", "behavior2"]:
        df[col] = df[col].astype("category")
    df["layer"]   = df["layer"].astype("int16")
    df["cos_sim"] = df["cos_sim"].astype("float32")

    print(f"Loaded {len(df):,} rows, {df['item_id'].nunique()} items, "
          f"{df['layer'].nunique()} layers")

    if args.skip_last_layer:
        last_layer = int(df["layer"].max())
        df = df[df["layer"] != last_layer]
        print(f"Skipping last layer ({last_layer}). Remaining layers: {df['layer'].nunique()}")

    if args.extractor:
        allowed = {e.strip() for e in args.extractor.split(",") if e.strip()}
        e1 = df["extractor1"].astype(str)
        e2 = df["extractor2"].astype(str)
        df = df[e1.isin(allowed) | e2.isin(allowed)]
        print(f"Extractor filter {allowed}: {len(df):,} rows remaining")

    print()

    df["same_lang"]   = (df["lang1"].astype(str) == df["lang2"].astype(str))
    df["same_prompt"] = (df["prompt1"].astype(str) == df["prompt2"].astype(str))

    # Uncomment to save aggregated info as CSV
    # save(agg_groupby(df, ["lang1", "lang2", "prompt1", "prompt2", "layer"]),
    #      output_dir / "by_condition_pair.csv")

    # save(agg_groupby(df, ["behavior1", "behavior2", "layer"]),
    #      output_dir / "by_behavior_pair.csv")

    # save(agg_groupby(df, ["lang1", "lang2", "layer"]),
    #      output_dir / "by_lang_pair.csv")

    # save(agg_groupby(df, ["prompt1", "prompt2", "layer"]),
    #      output_dir / "by_prompt_pair.csv")

    # save(agg_groupby(df, ["lang1", "lang2", "behavior1", "behavior2", "layer"]),
    #      output_dir / "by_lang_behavior.csv")

    # save(agg_groupby(df, ["prompt1", "prompt2", "behavior1", "behavior2", "layer"]),
    #      output_dir / "by_prompt_behavior.csv")

    # save(agg_groupby(df, ["same_lang", "same_prompt", "behavior1", "behavior2", "layer"]),
    #      output_dir / "by_sameflag_behavior.csv")

    if args.curves:
        print("\nGenerating curves plot ...")
        model_name = Path(args.input).stem
        plot_lang_prompt_curves(df, output_dir, title_prefix=model_name,
                                n_bootstrap=args.n_bootstrap)

    if args.answer_vs_noanswer:
        print("\nGenerating ANSWER vs NO-ANSWER curves ...")
        model_name = Path(args.input).stem
        plot_answer_vs_noanswer_curves(df, output_dir, title_prefix=model_name,
                                       n_bootstrap=args.n_bootstrap)

    if args.right_vs_wrong:
        print("\nGenerating ANSWER-RIGHT vs ANSWER-WRONG curves ...")
        model_name = Path(args.input).stem
        plot_right_vs_wrong_curves(df, output_dir, title_prefix=model_name,
                                   n_bootstrap=args.n_bootstrap)

    if args.lang_heatmap:
        print(f"\nGenerating lang×lang heatmaps (prompt={args.lang_heatmap}) ...")
        model_name = Path(args.input).stem
        plot_lang_heatmaps(
            csv_condition_pair=output_dir / "by_condition_pair.csv",
            output_dir=output_dir,
            prompt=args.lang_heatmap,
            step=args.heatmap_step,
            start=args.heatmap_start,
            title_prefix=model_name,
        )

    if args.heatmap:
        print("\nGenerating heatmaps ...")
        model_name = Path(args.input).stem
        plot_condition_heatmaps(
            df,
            output_dir,
            step=args.heatmap_step,
            start=args.heatmap_start,
            title_prefix=model_name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
