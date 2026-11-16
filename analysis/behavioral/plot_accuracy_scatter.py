"""
plot_accuracy_scatter.py

Creates a 3×1 summary plot of accuracy distributions across three model families.
Each panel shows one family with three conditions: zero-shot base, few-shot base, and zero-shot instruct.
Points are colored by language (steelblue for natural languages, special colors for 'fake' and 'empty').

Usage:
  python plot_accuracy_violins.py \\
    --zs-results summarized_results/accuracy_by_prompt.csv \\
    --fs-results summarized_results_fs/accuracy_by_prompt.csv \\
    --out-dir ./plots

Example:
python plot_accuracy_violins.py \\
  --zs-results summarized_results/accuracy_by_prompt.csv \\
  --fs-results summarized_results_fs/accuracy_by_prompt.csv \\
  --out-dir ./scatter_plots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Plot accuracy distributions (3x1 summary across model families)."
    )
    ap.add_argument(
        "--zs-results",
        required=True,
        help="Path to zero-shot accuracy_by_prompt.csv",
    )
    ap.add_argument(
        "--fs-results",
        required=True,
        help="Path to few-shot accuracy_by_prompt.csv",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for plot",
    )
    ap.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Output image format: png (default) or pdf (vector)",
    )
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    # Read data
    print("Loading zero-shot results...")
    df_zs = pd.read_csv(args.zs_results)
    print(f"  Loaded {len(df_zs)} records from {args.zs_results}")

    print("Loading few-shot results...")
    df_fs = pd.read_csv(args.fs_results)
    print(f"  Loaded {len(df_fs)} records from {args.fs_results}")

    # Define model families
    families = [
        ("gemma-3-4b-pt", "gemma-3-4b-pt", "gemma-3-4b-it", "Gemma-3 4B"),
        ("Meta-Llama-3.2-3B", "Meta-Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.2 3B"),
        ("Qwen3-1.7B-Base", "Qwen3-1.7B-Base", "Qwen3-1.7B", "Qwen3 1.7B"),
    ]

    sns.set_style("whitegrid")

    # Prepare combined data for all families
    all_family_data = []

    for zs_base, fs_base, it_model, family_name in families:
        print(f"\nProcessing family: {family_name}")

        # Get data for each condition
        zs_base_data = df_zs[df_zs["model"] == zs_base].copy()
        fs_base_data = df_fs[df_fs["model"] == fs_base].copy()
        it_data = df_zs[df_zs["model"] == it_model].copy()

        if zs_base_data.empty:
            print(f"  WARNING: No zero-shot base data for {zs_base}")
        if fs_base_data.empty:
            print(f"  WARNING: No few-shot base data for {fs_base}")
        if it_data.empty:
            print(f"  WARNING: No instructed data for {it_model}")

        if zs_base_data.empty and fs_base_data.empty and it_data.empty:
            print(f"  SKIP: No data found for {family_name}")
            continue

        # Add condition labels
        zs_base_data["condition"] = "Zero-shot Base"
        fs_base_data["condition"] = "Few-shot Base"
        it_data["condition"] = "Zero-shot Instruct"

        # Combine
        combined = pd.concat(
            [zs_base_data, fs_base_data, it_data],
            ignore_index=True,
        )

        all_family_data.append((combined, family_name))

    # Create 3x1 per-model plot (one panel per model, three conditions per panel)
    print("\nCreating 3x1 per-model plot...")

    n_fam = len(all_family_data)
    fig_pm, axes_pm = plt.subplots(1, n_fam, figsize=(6 * n_fam, 6), sharey=True)

    if n_fam == 1:
        axes_pm = [axes_pm]

    conditions = ["Zero-shot Base", "Few-shot Base", "Zero-shot Instruct"]

    for row_idx, (combined, family_name) in enumerate(all_family_data):
        ax = axes_pm[row_idx]

        _lang_items = sorted(combined["language"].unique())
        _special_colors = {"fake": "#e74c3c", "empty": "#f1c40f"}
        _lang_colors = {lang: _special_colors.get(lang, "steelblue") for lang in _lang_items}

        for cond_idx, condition in enumerate(conditions):
            data_cond = combined[combined["condition"] == condition]
            if data_cond.empty:
                continue
            for lang in _lang_items:
                subset = data_cond[data_cond["language"] == lang]
                if subset.empty:
                    continue
                x_jit = np.random.normal(cond_idx, 0.18, size=len(subset))
                ax.scatter(x_jit, subset["accuracy"],
                           marker="o",
                           color=_lang_colors[lang],
                           s=55,
                           alpha=0.65,
                           edgecolors="black",
                           linewidth=0.3,
                           label=lang if cond_idx == 0 else "_nolegend_")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy" if row_idx == 0 else "", fontsize=17)
        ax.set_xlabel("")
        ax.set_title(family_name, fontweight="bold", fontsize=17)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Zero-shot", "Few-shot", "Instruct"], fontsize=17)
        ax.tick_params(axis="y", labelsize=17)

    # Legend in the last panel
    legend_handles = []
    legend_labels = []

    legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor="steelblue", markersize=10,
                                     markeredgecolor="black", markeredgewidth=0.3))
    legend_labels.append("Natural language")

    label_map = {"fake": ("Fake", "#e74c3c"), "empty": ("Empty", "#f1c40f")}
    for key, (display_label, color) in label_map.items():
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor=color, markersize=10,
                                         markeredgecolor="black", markeredgewidth=0.3))
        legend_labels.append(display_label)

    axes_pm[-1].legend(legend_handles, legend_labels,
                       loc="lower right",
                       fontsize=17,
                       frameon=True,
                       framealpha=0.8)

    plt.tight_layout()
    out_path_pm = f"{args.out_dir}/accuracy_scatter_by_family.{args.format}"
    plt.savefig(out_path_pm, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path_pm}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
