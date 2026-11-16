#!/usr/bin/env python3
"""
Standalone legend for steering plots, with colors grouped by direction
instead of by "= position" vs "all positions":

  NO-ANS→ANS (= position)      blue  solid  square marker
  ANS→NO-ANS (= position)      red   solid  square marker
  ANS→NO-ANS (all positions)   red   dashed circle marker

Same visual style as plot_steering_results.py's --save_legend_apart output
(framed box, single row), but built independently so it doesn't change the
colors of the actual plots produced by that script.

Usage:
  python make_steering_legend_recolored.py -o legend.pdf
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RED = "#d62728"
BLUE = "#1f77b4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="steering_legend_recolored.pdf")
    args = parser.parse_args()

    handles = [
        mlines.Line2D([], [], color=BLUE, linestyle="-", linewidth=2,
                      marker="s", markersize=4, label="NO-ANS→ANS (= position)"),
        mlines.Line2D([], [], color=RED, linestyle="-", linewidth=2,
                      marker="s", markersize=4, label="ANS→NO-ANS (= position)"),
        mlines.Line2D([], [], color=RED, linestyle="--", linewidth=2,
                      marker="o", markersize=4, label="ANS→NO-ANS (all positions)"),
    ]

    fig = plt.figure()
    leg = fig.legend(handles=handles, loc="center", fontsize=14, ncol=len(handles))
    fig.savefig(
        args.output, dpi=150,
        bbox_inches=leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
    )
    print(f"Saved → {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
