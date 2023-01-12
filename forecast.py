#!/usr/bin/env python3

"""
forecast.py:
    make forecasts on synthetic or real data.

Usage:
    python forecast.py -d default -n 10000 -f all -o forecasts/default.csv -p plots/default
    python forecast.py -d forecasts/default.csv -f all -p plots/default --plot-only
    python forecast.py -d sigmoid -n 10000 -f all -o forecasts/sigmoid.csv -p plots/sigmoid
    python forecast.py -d forecasts/sigmoid.csv -f all -p plots/sigmoid --plot-only
    python forecast.py -d random -n 10000 -f all -o forecasts/random.csv -p plots/random
    python forecast.py -d forecasts/random.csv -f all -p plots/random --plot-only
"""

import argparse

import comparecast as cc


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="generate forecasts on synthetic data"
    )
    parser.add_argument("-d", "--data", required=True,
                        help="forecast data, given as either a csv filename "
                             "or the name of one of the presets")
    parser.add_argument("-n", "-t", "--n-rounds", type=int, default=0,
                        help="total number of rounds")
    parser.add_argument("-e", "--noise", type=float, default=0.1,
                        help="independent Gaussian noise for "
                             "true probabilities in synthetic data generation")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random seed for synthetic data generation")
    parser.add_argument("-f", "--forecasters", nargs="+", default=["all"],
                        help="list of forecasters to use (default: all)")
    parser.add_argument("-o", "--out-file", default=None,
                        help="csv filename to store the outputs")
    parser.add_argument("-p", "--plots-dir", default="./plots",
                        help="output directory for plots (default: ./plots)")
    parser.add_argument("--plot-only", action="store_true",
                        help="if provided, retrieves existing forecast data "
                             "and produce plots only (no forecasting)")
    parser.add_argument("--use-logx", dest="use_logx", action="store_true",
                        help="do not use log-scale on the time axis")
    args = parser.parse_args()

    data = cc.data_utils.synthetic.get_data(
        args.data, args.n_rounds, args.noise, args.seed)
    if not args.plot_only:
        data = cc.forecast(data, args.forecasters, args.out_file)
    cc.plot_forecasts(data, args.forecasters, args.plots_dir, args.use_logx)


if __name__ == "__main__":
    main()

