#!/usr/bin/env python3

"""
plot_comparisons.py:
    compare forecasts using score differentials and give confidence sequences.

Usage:
    python plot_comparisons.py -d forecasts/default.csv -p k29_poly3 -q laplace -o plots/default --plot-width
    python plot_comparisons.py -d forecasts/random.csv -p k29_poly3 -q laplace -o plots/random --plot-width
    python plot_comparisons.py -d forecasts/default.csv -p always_0 -q always_1 -o plots/default --plot-width
"""

import argparse

import comparecast as cc


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="compare probability forecasts using "
                    "time-uniform empirical Bernstein confidence sequences"
    )
    parser.add_argument("-d", "--data", "--data-name", required=True,
                        help="forecast data given as a csv file")
    parser.add_argument("-p", "--name-p", type=str, required=True,
                        help="column name of the first set of forecasts")
    parser.add_argument("-q", "--name-q", type=str, required=True,
                        help="column name of the second set of forecasts")
    parser.add_argument("-s", "--scoring-rule", default="brier",
                        help="scoring rule (default: brier)")
    parser.add_argument("-o", "--plots-dir", default="./plots",
                        help="output directory for plots (default: ./plots)")
    parser.add_argument("-a", "--alpha", type=float, default=0.05,
                        help="significance level for confidence sequences "
                             "(default: 0.05)")
    parser.add_argument("-l", "--lo", type=float, default=-1.,
                        help="minimum value of score differentials "
                             "(default: -1)")
    parser.add_argument("-u", "--hi", type=float, default=1.,
                        help="maximum value of score differentials "
                             "(default: 1)")
    parser.add_argument("-b", "--boundary-type", type=str, default="mixture",
                        help="type of uniform boundary used for CS "
                             "(default: mixture)")
    parser.add_argument("--v-opt", type=float, default=10,
                        help="value of intrinsic time where the boundary is "
                             "optimized. (Default: 10)")
    parser.add_argument("--compare-baselines", nargs="+", default=[],
                        help="list of baselines to compare "
                             "(options: h, ci, acs)")
    parser.add_argument("--plot-fder", action="store_true",
                        help="if true, also plot false decision rates "
                             "(default: True)")
    parser.add_argument("--plot-miscoverage", action="store_true",
                        help="if true, also plot cumulative miscoverage rates "
                             "(default: False)")
    parser.add_argument("--n-repeats", type=int, default=10000,
                        help="number of repeated trials for "
                             "miscoverage rate calculation (default: 10000)")
    parser.add_argument("--plot-width", action="store_true",
                        help="if true, also plot the width of "
                             "confidence sequences/intervals")
    parser.add_argument("--no-logx", dest="use_logx", action="store_false",
                        help="do not use log-scale on the time axis")

    args = parser.parse_args()
    cc.plot_comparison(**vars(args))


if __name__ == "__main__":
    main()

