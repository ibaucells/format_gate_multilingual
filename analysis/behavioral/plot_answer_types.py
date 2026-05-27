"""
Figure 2: Answer type breakdown
Grouped stacked bar chart: 3 model families × 3 conditions
    (zero-shot PT, few-shot PT, IT)
Each bar aggregates all (lang, prompt) data points for that model.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ANSWER_TYPES_CSV = "answer_types.csv"
OUTPUT_PNG_NAME = "plot_answer_types_new.png"

# ── Model families ─────────────────────────────────────────────────────────────
# Each entry: (display_name, zs_pt_model, fs_pt_model, it_model)
# zs_pt_model / it_model come from ZS_CSV; fs_pt_model comes from FS_CSV.
MODEL_FAMILIES = [
    ("Gemma-3-4B",   "gemma-3-4b-pt",      "gemma-3-4b-pt",      "gemma-3-4b-it"),
    ("Llama-3.2-3B", "Meta-Llama-3.2-3B",  "Meta-Llama-3.2-3B",  "Llama-3.2-3B-Instruct"),
    ("Qwen3-1.7B",   "Qwen3-1.7B-Base",    "Qwen3-1.7B-Base",    "Qwen3-1.7B"),
]

CONDITIONS = ["Zero-shot", "Few-shot", "Instruct"]

# ── Color palette (colorblind-friendly) ────────────────────────────────────────
COLORS = {
    "answer_right": "#2E7D32",
    "answer_wrong": "#C62828",
    "no_answer":    "#1565C0",
    "unknown":      "#757575",
}
LABELS = {
    "answer_right": "ANSWER-RIGHT",
    "answer_wrong": "ANSWER-WRONG",
    "no_answer":    "NO-ANSWER",
    "unknown":      "UNKNOWN",
}
STACK_ORDER = ["answer_right", "answer_wrong", "no_answer"]

COUNT_COLS = ["n", "answer_right", "answer_wrong", "no_answer", "unknown"]


def aggregate_model(df: pd.DataFrame, model_name: str) -> dict:
    """Sum raw counts for a given model across all (lang, prompt) rows."""
    rows = df[df["model"] == model_name][COUNT_COLS].sum()
    total = rows["n"] - rows["unknown"]
    result = {"n": total}
    for col in ["answer_right", "answer_wrong", "no_answer"]:
        result[f"{col}_pct"] = rows[col] / total if total > 0 else 0.0
    return result


def build_bar_data(zs_df: pd.DataFrame, fs_df: pd.DataFrame) -> list[list[dict]]:
    """
    Returns a list (one per model family) of lists (one per condition)
    of dicts with pct values and n.
    """
    data = []
    for _, zs_pt, fs_pt, it in MODEL_FAMILIES:
        family_bars = [
            aggregate_model(zs_df, zs_pt),
            aggregate_model(fs_df, fs_pt),
            aggregate_model(zs_df, it),
        ]
        data.append(family_bars)
    return data


def plot(bar_data: list[list[dict]], output_path: Path) -> None:
    n_families = len(MODEL_FAMILIES)
    n_conds = len(CONDITIONS)
    bar_width = 0.22
    group_gap = 0.9        # center-to-center distance between family groups
    cond_colors_edge = ["#888888", "#444444", "#000000"]  # subtle edge tints per cond

    fig, ax = plt.subplots(figsize=(10, 6.2))

    # x positions: group centers spaced by group_gap
    group_centers = np.arange(n_families) * group_gap
    # offsets within a group so the 3 bars are centred around group_center
    offsets = (np.arange(n_conds) - (n_conds - 1) / 2) * (bar_width + 0.02)

    all_bars: dict[str, list] = {k: [] for k in STACK_ORDER}  # for legend

    for fi, (family_bars, center) in enumerate(zip(bar_data, group_centers)):
        for ci, (bar_dict, offset) in enumerate(zip(family_bars, offsets)):
            x = center + offset
            bottom = 0.0
            for key in STACK_ORDER:
                v = bar_dict[f"{key}_pct"]
                rect = ax.bar(
                    x, v, bar_width,
                    bottom=bottom,
                    color=COLORS[key],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                if v >= 0.05:
                    ax.text(
                        x, bottom + v / 2,
                        f"{v:.0%}",
                        ha="center", va="center",
                        fontsize=15, color="white", fontweight="bold",
                    )
                if fi == 0 and ci == 0:
                    all_bars[key].append(rect)
                bottom += v

            # condition label below each bar
            ax.text(
                x, -0.02,
                CONDITIONS[ci],
                ha="right", va="top",
                fontsize=16, color="#333333",
                rotation=40,
                transform=ax.get_xaxis_transform(),
            )


    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f[0] for f in MODEL_FAMILIES], fontsize=21, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=80)

    ax.set_ylabel("Proportion of responses", fontsize=20)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="y", labelsize=18)
    last_bar_edge = group_centers[-1] + offsets[-1] + bar_width / 2
    ax.set_xlim(group_centers[0] - group_gap * 0.55, last_bar_edge + 0.05)

    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in STACK_ORDER]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=3,
        frameon=False,
        fontsize=16,
        borderaxespad=0,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot answer-type breakdown from summarized results directories."
    )
    parser.add_argument(
        "--zs-summarized-dir",
        type=Path,
        required=True,
        help="Directory containing zero-shot summarized results, including answer_types.csv",
    )
    parser.add_argument(
        "--fs-summarized-dir",
        type=Path,
        required=True,
        help="Directory containing few-shot summarized results, including answer_types.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to <zs-summarized-dir>/plot_answer_types_new.png",
    )
    args = parser.parse_args()

    zs_csv = args.zs_summarized_dir / ANSWER_TYPES_CSV
    fs_csv = args.fs_summarized_dir / ANSWER_TYPES_CSV
    output_png = args.output or (args.zs_summarized_dir / OUTPUT_PNG_NAME)

    for path in (zs_csv, fs_csv):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

    zs_df = pd.read_csv(zs_csv)
    fs_df = pd.read_csv(fs_csv)

    bar_data = build_bar_data(zs_df, fs_df)

    # Print a quick summary
    for (name, *_), family_bars in zip(MODEL_FAMILIES, bar_data):
        print(f"\n{name}")
        for cond, bd in zip(CONDITIONS, family_bars):
            pcts = "  ".join(f"{k}={bd[f'{k}_pct']:.1%}" for k in STACK_ORDER)
            print(f"  {cond:14s}  n={int(bd['n']):,}  {pcts}")

    plot(bar_data, output_png)


if __name__ == "__main__":
    main()
