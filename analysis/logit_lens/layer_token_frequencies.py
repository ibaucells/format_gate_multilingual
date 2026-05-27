"""
layer_token_frequencies.py

Reads all logit-lens CSV files from a folder and, for every layer, counts
how often each token appeared as the top-1 prediction across all dataset
items (all files combined).

Output per layer:
  - Absolute count and percentage for every distinct top-1 token.

Usage:
  python layer_token_frequencies.py logit_lens/to_analyse/gemma/all
  python layer_token_frequencies.py logit_lens/to_analyse/gemma/all --top 20
  python layer_token_frequencies.py logit_lens/to_analyse/gemma/all --type addition
  python layer_token_frequencies.py logit_lens/to_analyse/gemma/all --layers 0,5,10,last
  python layer_token_frequencies.py logit_lens/to_analyse/gemma/all --csv out.csv
  python layer_token_frequencies.py logit_lens/to_analyse/gemma/all --pivot pivot.csv

# Two separate pivots (pivot_ANSWER.csv and pivot_NO-ANSWER.csv)
python layer_token_frequencies.py logit_lens/to_analyse/gemma/all \
    --pivot logit_lens/to_analyse/split_by_categories/gemma/pivot.csv --split-answer-category

# Two CSVs + two pivots
python layer_token_frequencies.py logit_lens/to_analyse/gemma/all \
    --csv out.csv --pivot logit_lens/to_analyse/split_by_categories/gemma/pivot.csv --split-answer-category

# Plot: per-layer % of examples with each last-layer top-1 token, split by ANSWER / NO-ANSWER
python layer_token_frequencies.py logit_lens/to_analyse/gemma/all --plot out.png
python layer_token_frequencies.py logit_lens/to_analyse/gemma/all --plot out.png --plot-top 5

"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Helpers (shared with analyze_logit_lens.py)
# ---------------------------------------------------------------------------

def unsanitize(value: str) -> str:
    """Reverse the sanitize_field encoding from logit_lens.py."""
    return (value
            .replace("<comma>", ",")
            .replace("<sp>", " ")
            .replace("\\r\\n", "\r\n")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t"))


def discover_layers(fieldnames: List[str]) -> List[int]:
    """Return sorted list of layer indices found as L{i}_tok1 columns."""
    indices = []
    for f in fieldnames:
        if f.startswith("L") and f.endswith("_tok1"):
            try:
                indices.append(int(f[1:f.index("_")]))
            except ValueError:
                pass
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Answer-category classification
# ---------------------------------------------------------------------------

NO_ANSWER_EXTRACTORS: frozenset = frozenset({
    "empty",
    "no_digit_gate",
    "unfinished_operation_repetition",
    "prompt_repetition_drift",
    "unresolved_operation_loop",
    "other_repetition_drift",
})


def _answer_category(row: dict) -> str:
    """Return 'NO-ANSWER' or 'ANSWER' for a logit-lens CSV row."""
    is_correct = row.get("is_correct", "1")
    extraction_rule = row.get("extraction_rule", "")
    if str(is_correct).strip() in ("0", "False", "false", "") and extraction_rule in NO_ANSWER_EXTRACTORS:
        return "NO-ANSWER"
    return "ANSWER"


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def collect_data(folder: Path,
                 problem_type_filter: Optional[str],
                 answer_category_filter: Optional[str] = None,
                 ) -> Tuple[List[int], Dict[int, Dict[str, int]], int, Dict[int, float]]:
    """
    Walk all CSV files in *folder* and accumulate per-layer token counts.

    Parameters
    ----------
    answer_category_filter : "ANSWER", "NO-ANSWER", or None (no filter).
        Items are classified via _answer_category(); only matching rows are counted.

    Returns
    -------
    layer_order : list[int]
        Sorted list of all layer indices found across all files.
    counts : dict[layer_index -> dict[token -> count]]
        Absolute frequencies.
    total_rows : int
        Number of rows processed (after optional filters).
    mean_probs : dict[layer_index -> float]
        Mean top-1 softmax probability (0–1) per layer across all rows.
    """
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {folder}:")
    for p in csv_files:
        print(f"  {p.name}")

    # layer index -> {token -> count}
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prob_sums: Dict[int, float] = defaultdict(float)
    prob_counts: Dict[int, int] = defaultdict(int)
    all_layer_indices: Set[int] = set()
    total_rows = 0

    for path in csv_files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            layers = discover_layers(fieldnames)
            all_layer_indices.update(layers)

            for row in reader:
                if problem_type_filter:
                    if row.get("problem_type", "") != problem_type_filter:
                        continue
                if answer_category_filter:
                    if _answer_category(row) != answer_category_filter:
                        continue

                total_rows += 1
                for layer in layers:
                    tok_raw = row.get(f"L{layer}_tok1", "")
                    if tok_raw == "":
                        continue
                    tok = unsanitize(tok_raw)
                    counts[layer][tok] += 1
                    prob_raw = row.get(f"L{layer}_prob1", "")
                    if prob_raw:
                        try:
                            prob_sums[layer] += float(prob_raw)
                            prob_counts[layer] += 1
                        except ValueError:
                            pass

    mean_probs = {
        l: prob_sums[l] / prob_counts[l] if prob_counts[l] else 0.0
        for l in all_layer_indices
    }
    return sorted(all_layer_indices), counts, total_rows, mean_probs


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def parse_layer_selection(spec: str, all_layers: list[int]) -> list[int]:
    """Parse a comma-separated layer spec like '0,5,10,last' into indices."""
    result = []
    for part in spec.split(","):
        part = part.strip()
        if part.lower() == "last":
            result.append(all_layers[-1])
        else:
            idx = int(part)
            if idx not in all_layers:
                print(f"Warning: layer {idx} not found in data, skipping.", file=sys.stderr)
            else:
                result.append(idx)
    # Preserve order, deduplicate
    seen = set()
    deduped = []
    for v in result:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def print_layer_frequencies(layer_order: List[int],
                             counts: Dict[int, Dict[str, int]],
                             total_rows: int,
                             top_n: Optional[int]):
    """Print a human-readable frequency table for each layer."""
    for layer in layer_order:
        layer_counts = counts[layer]
        layer_total = sum(layer_counts.values())
        if layer_total == 0:
            continue

        sorted_tokens = sorted(layer_counts.items(), key=lambda x: -x[1])
        if top_n is not None:
            sorted_tokens = sorted_tokens[:top_n]

        print(f"\n{'─' * 65}")
        print(f"  Layer {layer:>3}   (rows with a prediction: {layer_total} / {total_rows})")
        print(f"{'─' * 65}")
        print(f"  {'Token':<30}  {'Count':>8}  {'% of layer':>12}  {'% of total':>12}")
        print(f"  {'─' * 28}  {'─' * 8}  {'─' * 12}  {'─' * 12}")

        for tok, cnt in sorted_tokens:
            pct_layer = 100.0 * cnt / layer_total
            pct_total = 100.0 * cnt / total_rows
            print(f"  {repr(tok):<30}  {cnt:>8}  {pct_layer:>11.2f}%  {pct_total:>11.2f}%")

        if top_n and len(counts[layer]) > top_n:
            others = len(counts[layer]) - top_n
            print(f"  ... ({others} more token type(s) not shown)")


def write_csv(out_path: Path,
              layer_order: List[int],
              counts: Dict[int, Dict[str, int]],
              total_rows: int):
    """Write per-layer token frequencies to a CSV file."""
    rows = []
    for layer in layer_order:
        layer_counts = counts[layer]
        layer_total = sum(layer_counts.values())
        if layer_total == 0:
            continue
        for tok, cnt in sorted(layer_counts.items(), key=lambda x: -x[1]):
            rows.append({
                "layer": layer,
                "token": repr(tok),
                "count": cnt,
                "pct_of_layer": round(100.0 * cnt / layer_total, 4),
                "pct_of_total": round(100.0 * cnt / total_rows, 4),
                "layer_total": layer_total,
                "grand_total": total_rows,
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "token", "count",
                        "pct_of_layer", "pct_of_total",
                        "layer_total", "grand_total"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to {out_path}  ({len(rows)} rows)")


def write_pivot_csv(out_path: Path,
                    layer_order: List[int],
                    counts: Dict[int, Dict[str, int]],
                    total_rows: int,
                    mean_probs: Optional[Dict[int, float]] = None):
    """
    Write a wide-format pivot CSV:
      rows    = distinct tokens (sorted by total count across all layers, desc)
      columns = token | total_count | L0_count | L0_pct | L1_count | L1_pct | ...

    *_pct columns are percentages of that layer's total predictions.
    """
    # Collect every token that appears in any layer
    all_tokens: Dict[str, int] = defaultdict(int)
    for layer_counts in counts.values():
        for tok, cnt in layer_counts.items():
            all_tokens[tok] += cnt

    tokens_sorted = sorted(all_tokens.keys(), key=lambda t: -all_tokens[t])

    # Pre-compute layer totals
    layer_totals = {layer: sum(counts[layer].values()) for layer in layer_order}

    # Build column names
    fieldnames = ["token", "total_count"]
    for layer in layer_order:
        fieldnames.append(f"L{layer}_count")
        fieldnames.append(f"L{layer}_pct")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tok in tokens_sorted:
            row: Dict[str, object] = {
                "token": repr(tok),
                "total_count": all_tokens[tok],
            }
            for layer in layer_order:
                cnt = counts[layer].get(tok, 0)
                lt = layer_totals[layer]
                row[f"L{layer}_count"] = cnt
                row[f"L{layer}_pct"] = round(100.0 * cnt / lt, 4) if lt else 0.0
            writer.writerow(row)

        if mean_probs is not None:
            meta_row: Dict[str, object] = {
                "token": "__mean_top1_prob__",
                "total_count": 0,
            }
            for layer in layer_order:
                meta_row[f"L{layer}_count"] = 0
                meta_row[f"L{layer}_pct"] = round(mean_probs.get(layer, 0.0) * 100, 4)
            writer.writerow(meta_row)

    print(f"\nPivot CSV written to {out_path}  "
          f"({len(tokens_sorted)} token(s) x {len(layer_order)} layer(s))")


# ---------------------------------------------------------------------------
# Plot: per-layer top-1 % by last-layer token, split by answer category
# ---------------------------------------------------------------------------

def collect_data_for_plot(
    folder: Path,
    problem_type_filter: Optional[str],
) -> Tuple[List[int], Dict[str, int], Dict[str, Dict[int, Dict[str, int]]]]:
    """
    Single-pass collection for the progression plot.

    Returns
    -------
    layer_order : list[int]
    cat_totals  : dict[category -> total row count]
    cat_counts  : dict[category -> dict[layer -> dict[token -> count]]]
    """
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}", file=sys.stderr)
        sys.exit(1)

    cat_totals: Dict[str, int] = defaultdict(int)
    # category -> layer -> token -> count
    cat_counts: Dict[str, Dict[int, Dict[str, int]]] = {
        "ANSWER": defaultdict(lambda: defaultdict(int)),
        "NO-ANSWER": defaultdict(lambda: defaultdict(int)),
    }
    all_layer_indices: Set[int] = set()

    for path in csv_files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            layers = discover_layers(fieldnames)
            all_layer_indices.update(layers)

            for row in reader:
                if problem_type_filter and row.get("problem_type", "") != problem_type_filter:
                    continue
                cat = _answer_category(row)
                cat_totals[cat] += 1
                for layer in layers:
                    tok_raw = row.get(f"L{layer}_tok1", "")
                    if tok_raw:
                        cat_counts[cat][layer][unsanitize(tok_raw)] += 1

    return sorted(all_layer_indices), dict(cat_totals), cat_counts


def plot_top1_progression(
    layer_order: List[int],
    cat_totals: Dict[str, int],
    cat_counts: Dict[str, Dict[int, Dict[str, int]]],
    out_path: Path,
    top_n: Optional[int],
):
    """
    Two-panel figure (ANSWER | NO-ANSWER).
    X-axis  : layer index
    Y-axis  : % of examples in that panel where token T is the top-1 prediction
    Lines   : one per distinct token that is top-1 at the last layer
    top_n   : if set, only the top_n most-frequent last-layer tokens are plotted
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        print("matplotlib is required for --plot. Install it with: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    last_layer = layer_order[-1]
    categories = ("ANSWER", "NO-ANSWER")

    # Build the global set of qualifying tokens:
    # keep any token whose last-layer share is >= 5% in at least one panel,
    # then apply the optional top_n cap (ranked by max share across panels).
    MIN_PCT = 5.0
    candidate_scores: Dict[str, float] = {}
    for cat in categories:
        total = cat_totals.get(cat, 0)
        if total == 0:
            continue
        for tok, cnt in cat_counts[cat].get(last_layer, {}).items():
            pct = 100.0 * cnt / total
            if pct >= MIN_PCT:
                candidate_scores[tok] = max(candidate_scores.get(tok, 0.0), pct)

    tokens_global = sorted(candidate_scores, key=lambda t: -candidate_scores[t])
    if top_n is not None:
        tokens_global = tokens_global[:top_n]

    # Assign one color per token, shared across both panels.
    palette = cm.tab10(np.linspace(0, 1, max(len(tokens_global), 1)))
    token_color = {tok: palette[i] for i, tok in enumerate(tokens_global)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, cat in zip(axes, categories):
        total = cat_totals.get(cat, 0)
        if total == 0:
            ax.set_title(f"{cat}  (no data)")
            ax.set_xlabel("Layer")
            ax.set_ylabel("% of examples")
            continue

        for tok in tokens_global:
            y_values = [
                100.0 * cat_counts[cat].get(layer, {}).get(tok, 0) / total
                for layer in layer_order
            ]
            ax.plot(layer_order, y_values, label=repr(tok),
                    color=token_color[tok], linewidth=1.6)

        ax.set_title(f"{cat}  (n={total})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("% of examples")
        ax.legend(fontsize=12, loc="upper left", framealpha=0.7)
        ax.set_xlim(layer_order[0], layer_order[-1] + 0.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Per-layer top-1 token frequency analysis across logit-lens CSVs."
    )
    ap.add_argument(
        "folder",
        help="Folder containing the logit-lens CSV files to analyse."
    )
    ap.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Show only the top N tokens per layer (default: all)."
    )
    ap.add_argument(
        "--type", dest="problem_type", default=None,
        help="Filter to rows whose 'problem_type' column matches this value."
    )
    ap.add_argument(
        "--layers", default=None,
        help="Comma-separated list of layer indices to display, e.g. '0,5,10,last'. "
             "Default: all layers."
    )
    ap.add_argument(
        "--csv", dest="csv_out", default=None, metavar="FILE",
        help="Also write long-format results to this CSV file."
    )
    ap.add_argument(
        "--pivot", dest="pivot_out", default=None, metavar="FILE",
        help="Write a wide-format pivot CSV: rows = tokens, columns = layers "
             "(L{i}_count and L{i}_pct per layer)."
    )
    ap.add_argument(
        "--split-answer-category", action="store_true",
        help="Produce separate outputs for ANSWER and NO-ANSWER item categories. "
             "Items are NO-ANSWER when is_correct==0 and extraction_rule is a "
             "known no-answer extractor; everything else is ANSWER. "
             "When --csv or --pivot are given, files are suffixed _ANSWER / _NO-ANSWER "
             "before the extension."
    )
    ap.add_argument(
        "--plot", dest="plot_out", default=None, metavar="FILE",
        help="Save a 2-panel figure (ANSWER | NO-ANSWER) showing, for each layer, "
             "the %% of examples where each last-layer top-1 token is the prediction. "
             "Supports any matplotlib-supported format (.png, .pdf, .svg, …)."
    )
    ap.add_argument(
        "--plot-top", type=int, default=None, metavar="N",
        help="Limit the plot to the top N last-layer tokens per panel (default: all)."
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.", file=sys.stderr)
        sys.exit(1)

    def _suffixed_path(base: Optional[str], suffix: str) -> Optional[Path]:
        """Insert *suffix* before the file extension of *base*, e.g. out.csv → out_ANSWER.csv."""
        if base is None:
            return None
        p = Path(base)
        return p.with_stem(p.stem + suffix)

    def _run(category_filter: Optional[str] = None):
        label = f" [{category_filter}]" if category_filter else ""
        layer_order, counts, total_rows, mean_probs = collect_data(
            folder, args.problem_type, answer_category_filter=category_filter
        )

        if total_rows == 0:
            print(f"No rows matched the given filters{label}.", file=sys.stderr)
            return

        print(f"\nTotal rows processed{label}: {total_rows}")
        if args.problem_type:
            print(f"Filtered to problem_type: {args.problem_type!r}")
        if category_filter:
            print(f"Filtered to answer category: {category_filter!r}")

        display_layers = layer_order
        if args.layers:
            display_layers = parse_layer_selection(args.layers, layer_order)

        display_counts = {k: counts[k] for k in display_layers}
        print_layer_frequencies(display_layers, display_counts, total_rows, args.top)

        suffix = f"_{category_filter}" if category_filter else ""

        if args.csv_out:
            csv_path = _suffixed_path(args.csv_out, suffix) if suffix else Path(args.csv_out)
            write_csv(csv_path, layer_order, counts, total_rows)

        if args.pivot_out:
            pivot_path = _suffixed_path(args.pivot_out, suffix) if suffix else Path(args.pivot_out)
            write_pivot_csv(pivot_path, layer_order, counts, total_rows, mean_probs=mean_probs)

    if args.split_answer_category:
        for category in ("ANSWER", "NO-ANSWER"):
            _run(category_filter=category)
    else:
        _run()

    if args.plot_out:
        layer_order, cat_totals, cat_counts = collect_data_for_plot(
            folder, args.problem_type
        )
        plot_top1_progression(
            layer_order, cat_totals, cat_counts,
            Path(args.plot_out), args.plot_top,
        )


if __name__ == "__main__":
    main()
