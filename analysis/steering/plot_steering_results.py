#!/usr/bin/env python3
"""
Plot steering accuracy by layer.

Curves:
  positive accuracy   red   solid  (starts at layer 0 = baseline)
  negative accuracy   blue  solid  (starts at layer 0 = baseline)
  extra (optional)    blue  dashed (starts at layer 0 = baseline)

python plot_steering_results.py neg.csv pos.csv -o out.png -t "Title"
python plot_steering_results.py neg.csv pos.csv --extra extra.csv -o out.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys


def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["accuracy"] = df["answer_correct"] / df["total"]
    return df


def get_baseline_accuracy(df: pd.DataFrame):
    rows = df[df["coeff"] == 0]
    if rows.empty:
        return None
    return rows["accuracy"].mean()


def get_layer_series(df: pd.DataFrame, coeff_filter=None) -> pd.DataFrame:
    sub = df[df["coeff"] != 0]
    if coeff_filter is not None:
        sub = sub[sub["coeff"] == coeff_filter]
    return (
        sub.groupby("layer")
        .agg(accuracy=("accuracy", "mean"))
        .reset_index()
        .sort_values("layer")
    )


def prepend_baseline(series: pd.DataFrame, baseline_acc) -> pd.DataFrame:
    if baseline_acc is None:
        return series
    baseline_row = pd.DataFrame({"layer": [0], "accuracy": [baseline_acc]})
    return pd.concat([baseline_row, series], ignore_index=True)


def pct_fmt(x, _):
    return f"{x:.0%}"


def main():
    parser = argparse.ArgumentParser(description="Plot steering accuracy by layer.")
    parser.add_argument("negative_file", help="Full path to negative steering CSV")
    parser.add_argument("positive_file", help="Full path to positive steering CSV")
    parser.add_argument("-o", "--output", default="steering_plot.png")
    parser.add_argument("-t", "--title", default="Steering Results by Layer")
    parser.add_argument(
        "--coeff",
        type=float,
        default=None,
        help="Filter to a specific non-zero coefficient (default: average all)",
    )
    parser.add_argument(
        "--extra",
        default=None,
        help="Optional third CSV file plotted as a blue dashed line",
    )
    parser.add_argument(
        "--extra-label",
        default=None,
        dest="extra_label",
        help="Legend label for the extra line (default: derived from baseline accuracy)",
    )
    parser.add_argument(
        "--save_legend_apart",
        action="store_true",
        help="Save the legend as a separate file instead of embedding it in the plot",
    )
    args = parser.parse_args()

    plt.rcParams.update({
        'font.size': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.labelsize': 20,
    })

    pos_df = compute_rates(pd.read_csv(args.positive_file))
    neg_df = compute_rates(pd.read_csv(args.negative_file))

    pos_base = get_baseline_accuracy(pos_df)
    neg_base = get_baseline_accuracy(neg_df)

    pos_layers = prepend_baseline(get_layer_series(pos_df, args.coeff), pos_base)
    neg_layers = prepend_baseline(get_layer_series(neg_df, args.coeff), neg_base)

    extra_layers = None
    extra_base = None
    if args.extra:
        extra_df = compute_rates(pd.read_csv(args.extra))
        extra_base = get_baseline_accuracy(extra_df)
        extra_layers = prepend_baseline(get_layer_series(extra_df, args.coeff), extra_base)

    if pos_layers.empty and neg_layers.empty:
        sys.exit("No data rows found in either file.")

    RED = "#d62728"
    BLUE = "#1f77b4"

    fig, ax = plt.subplots(figsize=(13, 6))

    if not pos_layers.empty:
        ax.plot(pos_layers["layer"], pos_layers["accuracy"],
                color=RED, linestyle="-", linewidth=2, marker="s", markersize=4,
                label="NO-ANS\u2192ANS (= position)")

    if not neg_layers.empty:
        ax.plot(neg_layers["layer"], neg_layers["accuracy"],
                color=BLUE, linestyle="-", linewidth=2, marker="s", markersize=4,
                label="ANS\u2192NO-ANS (= position)")

    if extra_layers is not None and not extra_layers.empty:
        if args.extra_label:
            extra_label = args.extra_label
        else:
            extra_label = "ANS\u2192NO-ANS (all positions)"
        ax.plot(extra_layers["layer"], extra_layers["accuracy"],
                color=RED, linestyle="--", linewidth=2, marker="o", markersize=4,
                label=extra_label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.04, 1.02)

    extra_set = set(extra_layers["layer"].tolist()) if extra_layers is not None and not extra_layers.empty else set()
    data_layers = sorted(
        set(pos_layers["layer"].tolist()) | set(neg_layers["layer"].tolist()) | extra_set
    )
    all_layers = list(range(int(data_layers[0]), int(data_layers[-1]) + 1))
    ax.set_xlim(all_layers[0] - 0.5, all_layers[-1] + 0.5)
    ax.xaxis.set_major_locator(mticker.FixedLocator(all_layers))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(l) for l in all_layers]))
    ax.tick_params(axis="x", rotation=45)
    if not args.save_legend_apart:
        ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")

    if args.save_legend_apart:
        handles, labels = ax.get_legend_handles_labels()
        legend_fig = plt.figure()
        leg = legend_fig.legend(handles=handles, labels=labels, loc="center", fontsize=14)
        stem, ext = args.output.rsplit(".", 1) if "." in args.output else (args.output, "png")
        legend_path = f"{stem}_legend.{ext}"
        legend_fig.savefig(legend_path, dpi=150, bbox_inches=leg.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
        print(f"Saved legend → {legend_path}")
        plt.close(legend_fig)

    plt.show()


if __name__ == "__main__":
    main()
