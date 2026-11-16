"""
Figure 2: Answer type breakdown
Grouped stacked bar chart: N model families × 3 conditions
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
OUTPUT_NAME = "plot_answer_types"

# ── Model families ─────────────────────────────────────────────────────────────
# Each entry: (display_name, zs_pt_model, fs_pt_model, it_model)
# zs_pt_model / it_model come from the zero-shot CSV; fs_pt_model from the few-shot CSV.
MODEL_FAMILIES = [
    ("Gemma-3-4B",    "gemma-3-4b-pt",       "gemma-3-4b-pt",       "gemma-3-4b-it"),
    ("Gemma-3-12B",   "gemma-3-12b-pt",      "gemma-3-12b-pt",      "gemma-3-12b-it"),
    ("Llama-3.2-1B",  "Llama-3.2-1B",        "Llama-3.2-1B",        "Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B",  "Meta-Llama-3.2-3B",   "Meta-Llama-3.2-3B",   "Llama-3.2-3B-Instruct"),
    ("Llama-3.1-8B",  "Meta-Llama-3.1-8B",   "Meta-Llama-3.1-8B",   "Meta-Llama-3.1-8B-Instruct"),
    ("Qwen3-0.6B",    "Qwen3-0.6B-Base",     "Qwen3-0.6B-Base",     "Qwen3-0.6B"),
    ("Qwen3-1.7B",    "Qwen3-1.7B-Base",     "Qwen3-1.7B-Base",     "Qwen3-1.7B"),
    ("Qwen3-4B",      "Qwen3-4B-Base",       "Qwen3-4B-Base",       "Qwen3-4B"),
    ("Qwen3-8B",      "Qwen3-8B-Base",       "Qwen3-8B-Base",       "Qwen3-8B"),
    ("Qwen3-14B",     "Qwen3-14B-Base",      "Qwen3-14B-Base",      "Qwen3-14B"),
    ("EuroLLM-1.7B",  "EuroLLM-1.7B",        "EuroLLM-1.7B",        "EuroLLM-1.7B-Instruct"),
    ("EuroLLM-9B",    "EuroLLM-9B",          "EuroLLM-9B",          "EuroLLM-9B-Instruct"),
    ("Teuken-7B",     "Teuken-7B-base-v0.6", "Teuken-7B-base-v0.6", "Teuken-7B-instruct-v0.6"),
    ("Salamandra-2B", "salamandra-2b",       "salamandra-2b",       "salamandra-2b-instruct"),
    ("Salamandra-7B", "salamandra-7b",       "salamandra-7b",       "salamandra-7b-instruct"),
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
    "answer_right": "Correct",
    "answer_wrong": "Wrong",
    "no_answer":    "No-answer",
    "unknown":      "Unknown",
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


def small_caps_text(ax, x, y, text, fontsize=12, small_scale=0.75, **kwargs):
    """
    Render `text` in a fake small-caps style: first letter of each word
    at normal size, remaining letters uppercase at a smaller size.
    Uses only matplotlib (no LaTeX/extra packages).

    Honors a `transform` kwarg (e.g. ax.transAxes) for positioning outside
    data coordinates; defaults to ax.transData like ax.text() does.
    Returns (final_x, texts): the final x position (same coordinate space
    as `x`, so callers can chain multiple labels left-to-right) and the
    list of Text artists created, so callers can reposition them later.
    """
    transform = kwargs.get("transform", ax.transData)
    small_fontsize = fontsize * small_scale
    ax.figure.canvas.draw()  # ensure a renderer exists before measuring text
    renderer = ax.figure.canvas.get_renderer()

    cur_x = x
    texts = []
    for word in text.split(" "):
        if not word:
            continue
        first, rest = word[0].upper(), word[1:].upper()

        # Draw first letter at full size
        t1 = ax.text(cur_x, y, first, fontsize=fontsize, **kwargs)
        texts.append(t1)
        ax.figure.canvas.draw()  # force layout so we can measure the bbox
        bbox1 = t1.get_window_extent(renderer=renderer)
        bbox1_data = bbox1.transformed(transform.inverted())
        cur_x = bbox1_data.x1

        # Draw the rest of the word at smaller size, right after
        if rest:
            t2 = ax.text(cur_x, y, rest, fontsize=small_fontsize, **kwargs)
            texts.append(t2)
            ax.figure.canvas.draw()
            bbox2 = t2.get_window_extent(renderer=renderer)
            bbox2_data = bbox2.transformed(transform.inverted())
            cur_x = bbox2_data.x1

        # Small space between words
        cur_x += (bbox1_data.x1 - bbox1_data.x0) * 0.3

    return cur_x, texts


def draw_small_caps_legend(ax, keys, y=1.08, fontsize=26,
                           swatch_w=0.032, swatch_h=0.032, item_gap=0.04) -> None:
    """Hand-drawn legend: a color swatch followed by a small-caps label,
    for each key in `keys`, laid out left-to-right and centered horizontally
    (axes-fraction coords) above the plot."""
    transform = ax.transAxes
    patches, texts = [], []
    cur_x = 0.0
    for key in keys:
        patch = mpatches.Rectangle(
            (cur_x, y - swatch_h / 2), swatch_w, swatch_h,
            transform=transform, color=COLORS[key], clip_on=False,
        )
        ax.add_patch(patch)
        patches.append(patch)
        cur_x += swatch_w + 0.008
        cur_x, item_texts = small_caps_text(
            ax, cur_x, y, LABELS[key],
            fontsize=fontsize, transform=transform,
            ha="left", va="center", color="#222222",
        )
        texts.extend(item_texts)
        cur_x += item_gap

    total_width = cur_x - item_gap  # drop the trailing gap after the last item
    shift = (1.0 - total_width) / 2.0
    for patch in patches:
        patch.set_x(patch.get_x() + shift)
    for txt in texts:
        tx, ty = txt.get_position()
        txt.set_position((tx + shift, ty))


def plot(bar_data: list[list[dict]], output_path: Path) -> None:
    n_families = len(MODEL_FAMILIES)
    n_conds = len(CONDITIONS)
    bar_width = 0.22 if n_families <= 3 else 0.18
    group_gap = 0.9        # center-to-center distance between family groups
    xtick_fontsize = 21 if n_families <= 3 else 13
    cond_label_fontsize = 16 if n_families <= 3 else 10
    bar_label_fontsize = 15 if n_families <= 3 else 9
    cond_colors_edge = ["#888888", "#444444", "#000000"]  # subtle edge tints per cond

    fig, ax = plt.subplots(figsize=(max(10, n_families * 1.7), 6.2))

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
                        fontsize=bar_label_fontsize, color="white", fontweight="bold",
                    )
                if fi == 0 and ci == 0:
                    all_bars[key].append(rect)
                bottom += v

            # condition label below each bar
            ax.text(
                x, -0.02,
                CONDITIONS[ci],
                ha="right", va="top",
                fontsize=cond_label_fontsize, color="#333333",
                rotation=40,
                transform=ax.get_xaxis_transform(),
            )


    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f[0] for f in MODEL_FAMILIES], fontsize=xtick_fontsize, fontweight="bold")
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
    draw_small_caps_legend(ax, STACK_ORDER)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Figure 2: answer-type breakdown from summarized results directories."
    )
    parser.add_argument(
        "--zs-summarized-dir",
        type=Path,
        required=True,
        help="Directory with zero-shot summarized results (PT + IT), including answer_types.csv",
    )
    parser.add_argument(
        "--fs-summarized-dir",
        type=Path,
        required=True,
        help="Directory with few-shot summarized results (PT only), including answer_types.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path without extension; --format sets it. "
             "Defaults to <zs-summarized-dir>/plot_answer_types",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Output image format: png (default) or pdf (vector)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Restrict to these model family display names "
             "(e.g. --models Qwen3-4B Llama-3.2-3B Gemma-3-4B). Default: all families.",
    )
    args = parser.parse_args()

    global MODEL_FAMILIES
    if args.models:
        name_set = set(args.models)
        available = [f[0] for f in MODEL_FAMILIES]
        missing = name_set - set(available)
        if missing:
            raise ValueError(f"Unknown model(s): {sorted(missing)}. Available: {available}")
        MODEL_FAMILIES = [f for f in MODEL_FAMILIES if f[0] in name_set]

    zs_csv = args.zs_summarized_dir / ANSWER_TYPES_CSV
    fs_csv = args.fs_summarized_dir / ANSWER_TYPES_CSV
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

    output_base = args.output or (args.zs_summarized_dir / OUTPUT_NAME)
    output_path = output_base.with_suffix(f".{args.format}")
    plot(bar_data, output_path)


if __name__ == "__main__":
    main()
