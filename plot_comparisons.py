#!/usr/bin/env python3

"""
plot_comparisons.py:
    compare forecasts using score differentials and give confidence sequences.

Usage:
    python plot_comparisons.py -d forecasts/default.csv -p k29_poly3 -q laplace -o plots/default
    python plot_comparisons.py -d forecasts/random.csv -p k29_poly3 -q laplace -o plots/random
"""

import argparse

import comparecast as cc


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="compare sequential probabilistic forecasts"
                    " using confidence sequences and e-processes"
    )
    parser.add_argument("-d", "--data", "--data-name", required=True,
                        help="forecast data given as a csv file")
    parser.add_argument("-p", "--name-p", type=str, required=True,
                        help="column name of the first forecaster")
    parser.add_argument("-q", "--name-q", type=str, required=True,
                        help="column name of the second forecaster")
    parser.add_argument("-s", "--scoring-rule", default="brier",
                        help="name of the scoring rule to be used (default: brier)")
    parser.add_argument("-l", "--lag", type=int, default=1,
                        help="forecast lag. Currently requires compute_cs is False if lag > 1. (default: 1)")
    parser.add_argument("-o", "--plots-dir", default="./plots",
                        help="output directory for plots (default: ./plots)")
    parser.add_argument("-a", "--alpha", type=float, default=0.05,
                        help="significance level for confidence sequences (default: 0.05)")
    parser.add_argument("-b", "--boundary-type", type=str, default="mixture",
                        help="type of uniform boundary used for CS (default: mixture)")
    parser.add_argument("--v-opt", type=float, default=10,
                        help="value of intrinsic time where the boundary is optimized. (Default: 10)")
    parser.add_argument("--baselines", nargs="+", default=[],
                        help="list of baselines to compare (options: h, ci, acs)")
    parser.add_argument("--n-repeats", type=int, default=10000,
                        help="number of repeated trials for miscoverage rate calculation (default: 10000)")
    parser.add_argument("--plot-diagnostics", action="store_true",
                        help="if true, also plot a diagnostic plot (default: False)")
    parser.add_argument("--diagnostics-fn", type=str, default="miscoverage",
                        help="which diagnostics function to use (default: miscoverage)")
    parser.add_argument("--diagnostics-baselines", nargs="+", default=["ci"],
                        help="which baseline CS/CIs to include for diagnostics (default: ci)")
    parser.add_argument("--use-logx", action="store_true",
                        help="if true, use the logarithmic scale on the x-axis")
    parser.add_argument("--linewidth", type=float, default=2,
                        help="line width. (default: 2)")
    parser.add_argument("--ylim-scale", type=float, default=0.6,
                        help="scale of the y-axis limit for the CS plot. (default: 0.6)")
    parser.add_argument("--no-title", action="store_true",
                        help="do not add a title to the plot. (default: False)")

    args = parser.parse_args()
    cc.plot_comparison(**vars(args))


if __name__ == "__main__":
    main()

