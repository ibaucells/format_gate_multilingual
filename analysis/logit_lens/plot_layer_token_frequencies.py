"""
plot_layer_token_frequencies.py

Reads a pivot CSV produced by layer_token_frequencies.py --pivot and draws
an area plot of the top-N token predictions across layers.

  X axis : layer index
  Y axis : % of dataset items whose top-1 prediction at that layer was <token>

The default is a stacked area chart so you can see both individual token
trajectories and their combined coverage.  Pass --no-stack for overlapping
semi-transparent areas instead.

Usage:
  python plot_layer_token_frequencies.py pivot.csv
  python plot_layer_token_frequencies.py pivot.csv --top 10
  python plot_layer_token_frequencies.py pivot.csv --no-stack
  python plot_layer_token_frequencies.py pivot.csv --out figure.png
  python plot_layer_token_frequencies.py logit_lens/to_analyse/pivot_gemma.csv --top 5 --title "Gemma — all"

Real examples:
python plot_layer_token_frequencies.py logit_lens/to_analyse/pivot_gemma.csv --top 5 --title "Gemma — all" --out logit_lens/to_analyse/area_chart_gemma.png

# Two pivots from a folder (side-by-side subplots)
python plot_layer_token_frequencies.py \
  --split-folder logit_lens/to_analyse/split_by_categories/gemma \
  --out logit_lens/to_analyse/area_chart_gemma_by_answer_type.png \
  --min-pct 30
  
# With options
python plot_layer_token_frequencies.py --split-folder path/to/folder/ --top 10 --out figure.png --title "Gemma — split"

# Single pivot (unchanged behaviour)
python plot_layer_token_frequencies.py pivot.csv --top 5 --out figure.png

python plot_layer_token_frequencies.py \
  --split-folder logit_lens/to_analyse/split_by_categories/gemma \
  --split-out logit_lens/to_analyse/gemma_answer.png \
             logit_lens/to_analyse/gemma_no_answer.png \
             logit_lens/to_analyse/gemma_legend.png \
  --min-pct 20

python plot_layer_token_frequencies.py \
  --split-folder logit_lens/to_analyse/split_by_categories/llama \
  --split-out logit_lens/to_analyse/llama_answer.png \
             logit_lens/to_analyse/llama_no_answer.png \
             logit_lens/to_analyse/llama_legend.png \
  --min-pct 20

python plot_layer_token_frequencies.py \
  --split-folder logit_lens/to_analyse/split_by_categories/qwen \
  --split-out logit_lens/to_analyse/qwen_answer.png \
             logit_lens/to_analyse/qwen_no_answer.png \
             logit_lens/to_analyse/qwen_legend.png \
  --min-pct 20
"""

import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print(
        "matplotlib is required.  Install it with:\n"
        "  pip install matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print(
        "pandas is required.  Install it with:\n"
        "  pip install pandas",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

import ast as _ast


def _normalize_token(s):
    """
    Normalize a token string for matching.

    The pivot CSV may store tokens either as raw strings (old behaviour) or as
    their Python repr (e.g. "' '" or "'?\\n'", new behaviour).  Users on the
    command line may also pass either form.  This function collapses both to
    the raw string so comparisons always work.
    """
    try:
        result = _ast.literal_eval(s)
        if isinstance(result, str):
            return result
    except (ValueError, SyntaxError):
        pass
    return s


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pivot(path: Path, top_n: int, rank_by: str = "total", key_tokens=None,
               min_pct: float = None):
    """
    Load the pivot CSV and return:
      layers       : sorted list of layer indices (int)
      df_pct       : DataFrame  [token x layer]  pct values for selected tokens + OTHER row
      dominant_pct : dict  layer -> float  (pct of the most frequent token at each layer)

    key_tokens : explicit list of token strings (as they appear in the CSV) to show.
                 If given, rank_by is ignored.  Everything else is collapsed into OTHER.

    rank_by : "total"     — top-N tokens by sum across all layers (default)
              "last"      — top-N tokens by count at the last layer only
              "per-layer" — union of: top-N at the last layer
                                    + dominant token at every other layer
              int str     — top-N tokens by count at that specific layer index
    """
    df = pd.read_csv(path)

    # Validate expected columns
    if "token" not in df.columns or "total_count" not in df.columns:
        print(
            f"Error: expected 'token' and 'total_count' columns in {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract mean top-1 prob overlay row before any token processing
    meta = df[df["token"] == "__mean_top1_prob__"]
    df = df[df["token"] != "__mean_top1_prob__"].reset_index(drop=True)

    # Discover layer indices from column names like L0_pct, L12_pct …
    pct_cols = [c for c in df.columns if c.startswith("L") and c.endswith("_pct")]
    if not pct_cols:
        print("Error: no L{i}_pct columns found in the CSV.", file=sys.stderr)
        sys.exit(1)

    layers = sorted(int(c[1:c.index("_")]) for c in pct_cols)

    # Per-layer dominant % — max pct across ALL tokens at each layer (before filtering)
    dominant_pct = {
        layer: df[f"L{layer}_pct"].max()
        for layer in layers
    }

    # ---- Select the key token rows ----------------------------------------
    if key_tokens is not None:
        # Match user-supplied tokens against the CSV using normalised comparison,
        # so both repr form ("' '") and raw form (" ") work regardless of how
        # the CSV was generated.
        tok_map = {_normalize_token(t): t for t in df["token"].values}
        ordered_tokens = []
        missing = []
        for kt in key_tokens:
            norm = _normalize_token(kt)
            if norm in tok_map:
                ordered_tokens.append(tok_map[norm])
            else:
                missing.append(kt)
        if missing:
            print(
                f"Warning: these key tokens were not found in the CSV and will be skipped: "
                f"{missing}",
                file=sys.stderr,
            )
        if not ordered_tokens:
            print("Error: none of the specified key tokens were found in the CSV.", file=sys.stderr)
            print("Run with --list-tokens to see all available token strings.", file=sys.stderr)
            sys.exit(1)
        df_top = df[df["token"].isin(ordered_tokens)].copy()
        df_top = df_top.set_index("token").loc[ordered_tokens].reset_index()
        print(f"Key tokens (matched to CSV): {ordered_tokens}")

    elif min_pct is not None:
        pct_cols_all = [f"L{l}_pct" for l in layers]
        df["_peak_pct"] = df[pct_cols_all].max(axis=1)
        # Sort by layer 0 coverage first (for clearer stacking order), then by peak
        df["_l0_pct"] = df[f"L{layers[0]}_pct"] if layers else 0
        df_top = (df[df["_peak_pct"] >= min_pct]
                  .sort_values(["_l0_pct", "_peak_pct"], ascending=[False, False])
                  .drop(columns=["_peak_pct", "_l0_pct"])
                  .reset_index(drop=True)
                  .copy())
        ordered_tokens = df_top["token"].tolist()
        if not ordered_tokens:
            print(f"Warning: no tokens reach {min_pct}% at any layer.", file=sys.stderr)
        else:
            print(f"Tokens with peak pct >= {min_pct}%: {ordered_tokens}")

    elif rank_by == "per-layer":
        last_col = f"L{layers[-1]}_count"
        last_layer_tokens = set(df.nlargest(top_n, last_col)["token"].tolist())
        per_layer_tokens = set()
        for layer in layers:
            count_col = f"L{layer}_count"
            best_tok = df.loc[df[count_col].idxmax(), "token"]
            per_layer_tokens.add(best_tok)
        selected_tokens = last_layer_tokens | per_layer_tokens

        def sort_key(tok):
            row = df[df["token"] == tok]
            last_cnt = int(row[last_col].values[0]) if len(row) else 0
            return (0 if tok in last_layer_tokens else 1, -last_cnt)

        ordered_tokens = sorted(selected_tokens, key=sort_key)
        df_top = df[df["token"].isin(selected_tokens)].copy()
        df_top = df_top.set_index("token").loc[ordered_tokens].reset_index()
        print(
            f"Token selection: top-{top_n} at last layer "
            f"+ dominant token per layer  →  {len(ordered_tokens)} unique token(s)"
        )
    else:
        if rank_by == "total":
            rank_col = "total_count"
        elif rank_by == "last":
            rank_col = f"L{layers[-1]}_count"
        else:
            try:
                layer_idx = int(rank_by)
            except ValueError:
                print(
                    f"Error: --rank-by must be 'total', 'last', 'per-layer', or a layer index. "
                    f"Got: {rank_by!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            rank_col = f"L{layer_idx}_count"
            if rank_col not in df.columns:
                print(f"Error: column {rank_col!r} not found in CSV.", file=sys.stderr)
                sys.exit(1)

        print(f"Ranking tokens by: {rank_col}")
        df_top = df.nlargest(top_n, rank_col).copy()

    # ---- Build [token x layer] pct matrix for key tokens ------------------
    pct_matrix = {}
    for layer in layers:
        col = f"L{layer}_pct"
        pct_matrix[layer] = df_top[col].values

    df_pct = pd.DataFrame(pct_matrix, index=df_top["token"].values)

    if not meta.empty:
        overlay_pct = {layer: float(meta[f"L{layer}_pct"].values[0]) for layer in layers}
        overlay_label = "Mean top-1 prob (×100)"
    else:
        overlay_pct = dominant_pct
        overlay_label = "% matching dominant token"

    return layers, df_pct, overlay_pct, overlay_label


# ---------------------------------------------------------------------------
# Joint token selection (for split mode)
# ---------------------------------------------------------------------------

def _joint_top_tokens_by_pct(path_a: Path, path_n: Path, top_last: int,
                              min_pct: float = 20.0):
    """
    Select tokens jointly from both pivot CSVs:

      1. Top *top_last* tokens at the last layer, ranked by max pct across
         both pivots at that layer.
      2. Any additional token that reaches >= *min_pct* % at any layer in
         either pivot (excluding tokens already selected in step 1).

    Returns last-layer tokens first, then threshold tokens ordered by peak pct.
    """
    def _read(path):
        df = pd.read_csv(path)
        pct_cols = [c for c in df.columns if c.startswith("L") and c.endswith("_pct")]
        layers = sorted(int(c[1:c.index("_")]) for c in pct_cols)
        return df, layers

    df_a, layers_a = _read(path_a)
    df_n, layers_n = _read(path_n)

    all_tokens = set(df_a["token"].tolist()) | set(df_n["token"].tolist())
    all_tokens.discard("__mean_top1_prob__")
    last_layer = max(set(layers_a) | set(layers_n))

    def _pct_at(tok, df, layer):
        row = df[df["token"] == tok]
        if row.empty:
            return 0.0
        return float(row[f"L{layer}_pct"].values[0])

    def _peak_pct(tok, df, layers):
        if not layers:
            return 0.0
        return max(_pct_at(tok, df, l) for l in layers)

    # Step 1: top-last at the last layer (jointly); skipped when top_last == 0
    if top_last > 0:
        last_tokens = sorted(
            all_tokens,
            key=lambda t: max(_pct_at(t, df_a, last_layer) if last_layer in layers_a else 0.0,
                              _pct_at(t, df_n, last_layer) if last_layer in layers_n else 0.0),
            reverse=True,
        )[:top_last]
    else:
        last_tokens = []
    last_set = set(last_tokens)

    # Step 2: any token hitting >= min_pct at any layer in either pivot
    threshold_tokens = sorted(
        (t for t in all_tokens
         if t not in last_set
         and max(_peak_pct(t, df_a, layers_a), _peak_pct(t, df_n, layers_n)) >= min_pct),
        key=lambda t: max(_peak_pct(t, df_a, layers_a), _peak_pct(t, df_n, layers_n)),
        reverse=True,
    )

    return last_tokens + threshold_tokens




# Colour palette — 20 standard, muted colours for academic figures
_PALETTE = [
    "#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5",
    "#70AD47", "#C5504A", "#9E7BB5", "#7F7F7F", "#BDB76B",
    "#2E75B6", "#D97706", "#7C7C7C", "#997300", "#5B7C99",
    "#548235", "#92534B", "#6D5C8C", "#666666", "#8B7355"
]


def _build_colour_map(tokens):
    """
    Assign a stable color to each token string.
    OTHER always gets gray; named tokens get palette colors in order.
    Returns dict {token_str: hex_color}.
    """
    palette_colours = [c for c in _PALETTE if c != "#8C8C8C"]
    colour_map = {}
    palette_idx = 0
    for tok in tokens:
        if str(tok) == "OTHER":
            colour_map[str(tok)] = "#BBBBBB"
        else:
            colour_map[str(tok)] = palette_colours[palette_idx % len(palette_colours)]
            palette_idx += 1
    return colour_map


def _display_label(tok):
    if str(tok) == "OTHER":
        return "OTHER"
    raw = _normalize_token(str(tok))
    inner = repr(raw)[1:-1]
    return f'"{inner}"'


def plot_area(layers, df_pct, dominant_pct, stacked: bool, title: str,
              out_path=None, ax=None, show_legend: bool = True,
              colour_map=None, overlay_label: str = "% matching dominant token"):
    """
    Draw the area chart.

    When *ax* is None a new figure is created, saved/shown, and the function
    returns None.  When *ax* is supplied the chart is drawn onto it and the
    caller is responsible for saving/showing the parent figure.
    Set *show_legend* to False to suppress the per-axis legend (e.g. when the
    caller will add a single shared figure-level legend instead).
    colour_map : optional dict {token_str -> hex_color} for consistent colors
                 across multiple subplots.  If None, colors are assigned locally.
    """
    import numpy as np

    n_tokens = len(df_pct)
    if colour_map is None:
        colour_map = _build_colour_map(df_pct.index)
    colours = [colour_map.get(str(tok), "#BBBBBB") for tok in df_pct.index]
    x = np.array(layers)

    token_labels = [_display_label(t) for t in df_pct.index]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))

    if stacked:
        ys = [df_pct.iloc[i].values for i in range(n_tokens)]
        ax.stackplot(x, *ys, labels=token_labels, colors=colours, alpha=0.85)
        ax.set_ylabel("Cumulative % of items (stacked)", fontsize=18)
    else:
        for i in range(n_tokens):
            y = df_pct.iloc[i].values
            ax.fill_between(x, y, alpha=0.35, color=colours[i])
            ax.plot(x, y, color=colours[i], linewidth=1.8, label=token_labels[i])
        ax.set_ylabel("% of items with this top-1 prediction", fontsize=18)

    dom_y = np.array([dominant_pct[layer] for layer in layers])
    ax.plot(
        x, dom_y,
        color="black", linewidth=1.5, linestyle="--",
        label=overlay_label,
        zorder=5,
    )

    ax.margins(x=0)
    ax.set_xlabel("Layer", fontsize=18)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=20))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.tick_params(axis="both", labelsize=18)

    if show_legend:
        ax.legend(
            title="Top-1 token",
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            framealpha=0.9,
            fontsize=18,
            title_fontsize=18,
        )

    if standalone:
        plt.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {out_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_split_pivots(folder: Path):
    """
    Return (answer_path, no_answer_path) from a folder that contains pivot CSVs
    ending with _ANSWER.csv and _NO-ANSWER.csv respectively.
    """
    answer_files = sorted(folder.glob("*_ANSWER.csv"))
    no_answer_files = sorted(folder.glob("*_NO-ANSWER.csv"))

    def _pick(files, label):
        if not files:
            print(f"Error: no {label} pivot CSV found in {folder}.", file=sys.stderr)
            sys.exit(1)
        if len(files) > 1:
            print(
                f"Warning: multiple {label} pivot CSVs found; using {files[0].name}",
                file=sys.stderr,
            )
        return files[0]

    return _pick(answer_files, "_ANSWER"), _pick(no_answer_files, "_NO-ANSWER")


def plot_split(answer_path: Path, no_answer_path: Path, top_n: int, rank_by: str,
               key_tokens, stacked: bool, title_base: str, out_path,
               min_pct: float = None, split_out_paths=None):
    """Load both pivot CSVs and draw them side by side with a single shared legend."""
    # When using --min-pct, allow each plot to show its own complete token set
    # (all tokens reaching the threshold) for accurate 100% stacking.
    # For other modes, derive a joint set so tokens important in one category
    # still appear in both panels for consistency.
    if key_tokens is None:
        if min_pct is not None:
            # Don't constrain by a joint token set; let each plot show all tokens >= min_pct
            key_tokens = None
        else:
            key_tokens = _joint_top_tokens_by_pct(answer_path, no_answer_path,
                                                   top_last=top_n)

    layers_a, df_a, dom_a, lbl_a = load_pivot(answer_path, top_n, rank_by=rank_by, key_tokens=key_tokens, min_pct=min_pct)
    layers_n, df_n, dom_n, lbl_n = load_pivot(no_answer_path, top_n, rank_by=rank_by, key_tokens=key_tokens, min_pct=min_pct)

    def _total_cases(pivot_path):
        df = pd.read_csv(pivot_path)
        df = df[df["token"] != "__mean_top1_prob__"]
        count_cols = [c for c in df.columns
                      if c.startswith("L") and c.endswith("_count") and c != "total_count"]
        if not count_cols:
            return 0
        last_layer = max(int(c[1:c.index("_")]) for c in count_cols)
        return int(df[f"L{last_layer}_count"].sum())

    n_answer = _total_cases(answer_path)
    n_no_answer = _total_cases(no_answer_path)
    print(f"ANSWER cases:    {n_answer}")
    print(f"NO-ANSWER cases: {n_no_answer}")
    print(f"Total:           {n_answer + n_no_answer}")

    # Build a single color map from the joint token set (order already fixed above).
    all_tokens = list(dict.fromkeys(
        [str(t) for t in df_a.index] + [str(t) for t in df_n.index]
    ))
    shared_colour_map = _build_colour_map(all_tokens)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))

    plot_area(layers_a, df_a, dom_a, stacked=stacked,
              title=f"{title_base} — ANSWER", ax=ax1, show_legend=False,
              colour_map=shared_colour_map, overlay_label=lbl_a)
    plot_area(layers_n, df_n, dom_n, stacked=stacked,
              title=f"{title_base} — NO-ANSWER", ax=ax2, show_legend=False,
              colour_map=shared_colour_map, overlay_label=lbl_n)

    # Build legend from the shared color map (one entry per token, in order)
    import matplotlib.patches as mpatches
    handles = []
    for tok, color in shared_colour_map.items():
        handles.append(mpatches.Patch(color=color, label=tok))
    # Add the dashed overlay line entry
    import matplotlib.lines as mlines
    handles.append(mlines.Line2D([], [], color="black", linewidth=1.5,
                                 linestyle="--", label=lbl_a))

    fig.legend(
        handles=handles,
        title="Top-1 token",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0,
        framealpha=0.9,
    )

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {out_path}")
    else:
        plt.show()

    plt.close(fig)

    if split_out_paths is not None:
        answer_out, no_answer_out = split_out_paths[:2]
        legend_out = split_out_paths[2] if len(split_out_paths) > 2 else None

        fig_a, ax_a = plt.subplots(figsize=(12, 6))
        plot_area(layers_a, df_a, dom_a, stacked=stacked,
                  title="", ax=ax_a, show_legend=False,
                  colour_map=shared_colour_map, overlay_label=lbl_a)
        fig_a.tight_layout()
        fig_a.savefig(answer_out, dpi=150, bbox_inches="tight")
        plt.close(fig_a)
        print(f"ANSWER figure saved to {answer_out}")

        fig_n, ax_n = plt.subplots(figsize=(12, 6))
        plot_area(layers_n, df_n, dom_n, stacked=stacked,
                  title="", ax=ax_n, show_legend=False,
                  colour_map=shared_colour_map, overlay_label=lbl_n)
        fig_n.tight_layout()
        fig_n.savefig(no_answer_out, dpi=150, bbox_inches="tight")
        plt.close(fig_n)
        print(f"NO-ANSWER figure saved to {no_answer_out}")

        if legend_out is not None:
            import matplotlib.patches as mpatches
            import matplotlib.lines as mlines
            handles = [
                mpatches.Patch(color=color, label=_display_label(tok))
                for tok, color in shared_colour_map.items()
            ]
            handles.append(mlines.Line2D([], [], color="black", linewidth=1.5,
                                         linestyle="--", label=lbl_a))
            fig_leg = plt.figure()
            ax_leg = fig_leg.add_subplot(111)
            ax_leg.axis("off")
            ax_leg.legend(handles=handles, title="Top-1 token", loc="center",
                          framealpha=0.9, fontsize=18, title_fontsize=18)
            fig_leg.savefig(legend_out, dpi=150, bbox_inches="tight")
            plt.close(fig_leg)
            print(f"Legend figure saved to {legend_out}")


def main():
    ap = argparse.ArgumentParser(
        description="Area plot of top-N token predictions per layer from a pivot CSV."
    )
    ap.add_argument(
        "pivot_csv", nargs="?", default=None,
        help="Pivot CSV produced by layer_token_frequencies.py --pivot. "
             "Omit when using --split-folder."
    )
    ap.add_argument(
        "--split-folder", dest="split_folder", default=None, metavar="FOLDER",
        help=(
            "Folder containing two pivot CSVs ending with _ANSWER.csv and "
            "_NO-ANSWER.csv (produced by layer_token_frequencies.py "
            "--split-answer-category --pivot). Draws them side by side."
        )
    )
    ap.add_argument(
        "--top", type=int, default=5, metavar="N",
        help="Number of top tokens to show (default: 5)."
    )
    ap.add_argument(
        "--no-stack", dest="stacked", action="store_false",
        help="Draw overlapping semi-transparent areas instead of a stacked chart."
    )
    ap.add_argument(
        "--title", default=None,
        help="Plot title (default: derived from the CSV filename)."
    )
    ap.add_argument(
        "--out", default=None, metavar="FILE",
        help="Save figure to this file (PNG/PDF/SVG). If omitted, opens interactive window."
    )
    ap.add_argument(
        "--rank-by", dest="rank_by", default="total", metavar="LAYER",
        help=(
            "How to select tokens to plot: "
            "'total' (default) = top-N by sum across all layers; "
            "'last' = top-N by last layer count; "
            "'per-layer' = top-N at last layer + dominant token at every layer; "
            "or an integer layer index, e.g. '26'. Ignored when --key-tokens is given."
        )
    )
    ap.add_argument(
        "--key-tokens", dest="key_tokens", nargs="+", default=None, metavar="TOKEN",
        help=(
            "Explicit list of tokens to plot (as they appear in the 'token' column of the CSV, "
            "i.e. in repr form, e.g. \"' '\" \"' -'\" \"'?\\\\n'\"). "
            "Everything else is collapsed into an OTHER (gray) category. "
            "Use --list-tokens to see all available token strings."
        )
    )
    ap.add_argument(
        "--min-pct", dest="min_pct", type=float, default=None, metavar="PCT",
        help=(
            "Only include tokens that are the top-1 prediction for at least PCT%% of "
            "examples at some layer (e.g. --min-pct 5). "
            "Overrides --top and --rank-by. Ignored when --key-tokens is given."
        ),
    )
    ap.add_argument(
        "--list-tokens", dest="list_tokens", action="store_true",
        help="Print all token strings in the CSV (sorted by total_count desc) and exit."
    )
    ap.add_argument(
        "--split-out", dest="split_out", nargs="+", default=None,
        metavar="FILE",
        help=(
            "Save the two panels as separate image files (same width, no legend): "
            "ANSWER_FILE NO_ANSWER_FILE [LEGEND_FILE]. "
            "If a third path is given, the legend is saved there as a standalone image. "
            "Only valid with --split-folder."
        ),
    )
    ap.set_defaults(stacked=True)
    args = ap.parse_args()

    if args.split_folder:
        folder = Path(args.split_folder)
        if not folder.is_dir():
            print(f"Error: {folder} is not a directory.", file=sys.stderr)
            sys.exit(1)
        answer_path, no_answer_path = _find_split_pivots(folder)
        title_base = args.title or f"Top-{args.top} token predictions per layer — {folder.name}"
        out_path = Path(args.out) if args.out else None
        split_out_paths = [Path(p) for p in args.split_out] if args.split_out else None
        plot_split(
            answer_path, no_answer_path,
            top_n=args.top, rank_by=args.rank_by, key_tokens=args.key_tokens,
            stacked=args.stacked, title_base=title_base, out_path=out_path,
            min_pct=args.min_pct, split_out_paths=split_out_paths,
        )
        return

    if not args.pivot_csv:
        ap.error("pivot_csv is required unless --split-folder is given.")

    pivot_path = Path(args.pivot_csv)
    if not pivot_path.exists():
        print(f"Error: {pivot_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.list_tokens:
        import pandas as _pd
        df = _pd.read_csv(pivot_path)
        print(f"\nTokens in {pivot_path.name} (sorted by total_count):\n")
        for tok in df.sort_values("total_count", ascending=False)["token"]:
            print(f"  {tok}")
        sys.exit(0)

    layers, df_pct, dominant_pct, overlay_label = load_pivot(
        pivot_path, args.top, rank_by=args.rank_by, key_tokens=args.key_tokens,
        min_pct=args.min_pct,
    )

    title = args.title or f"Top-{args.top} token predictions per layer — {pivot_path.stem}"
    out_path = Path(args.out) if args.out else None

    plot_area(layers, df_pct, dominant_pct, stacked=args.stacked, title=title, out_path=out_path,
              overlay_label=overlay_label)


if __name__ == "__main__":
    main()
