"""
Some plotting utilities
"""

import logging
import os.path
from typing import Iterable, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from comparecast.comparecast import compare_forecasts
from comparecast.forecasters import FORECASTERS_ALL
from comparecast.scoring import get_scoring_rule
from comparecast.confint import confint_lai
from comparecast.diagnostics import compute_true_deltas, compute_diagnostics
from comparecast.data_utils.weather import AIRPORTS, LAGS, read_precip_fcs
from comparecast.eprocess import eprocess_hz


DEFAULT_CONTEXT = "paper"
DEFAULT_STYLE = "whitegrid"
DEFAULT_PALETTE = "colorblind"
DEFAULT_FONT = "Avenir"
DEFAULT_FONTS_SCALE = 1.75


def set_theme(
        context: str = DEFAULT_CONTEXT,
        style: str = DEFAULT_STYLE,
        palette: str = DEFAULT_PALETTE,
        font: str = DEFAULT_FONT,
        font_scale: float = DEFAULT_FONTS_SCALE,
        **kwargs
):
    """Set default plotting style.

    Alias for `seaborn.set_theme()` with different defaults.
    """
    return sns.set_theme(context, style, palette, font, font_scale, **kwargs)


def get_color_by_index(index: int, palette: str = DEFAULT_PALETTE):
    """Get color by integer indices."""
    colors = sns.color_palette(palette)
    return colors[index % len(colors)]


def get_colors(palette: str = DEFAULT_PALETTE):
    """Get colors defined by a seaborn palette."""
    return sns.color_palette(palette)


def plot_forecasts(
        data: pd.DataFrame,
        forecasters: Iterable[str],
        plots_dir: str = "./plots",
        use_logx: bool = True,
        figsize: Tuple[int, int] = (12, 5),
        linewidth: float = 3,
        legend_out: bool = True,
        savefig_ext: str = "pdf",
        savefig_dpi: float = 300,
        **theme_kwargs
):
    """Plot forecasts along with the dataset.

    Also plots ``data`` or ``true_probs``,
    if those columns are available in ``data``.

    Args:
        data: A ``pd.DataFrame`` containing a set of forecasts as its columns.
            Must contain columns ``time`` and each element of ``forecasters``.
        forecasters: list of forecasters (columns of ``data``) to be plotted.
        plots_dir: directory to save the plots.
        use_logx: whether to use the log-scale on the time (x) axis.
        figsize: output figure size.
        linewidth: line width.
        legend_out: whether to draw the legend outside the plot.
        legend_loc: location of the legend. Only applies when ``legend_out == False``.

    Returns: None
        Saves a plot to ``{plots_dir}/forecasters.pdf``.
    """
    if "all" in forecasters:
        forecasters = [f for f in FORECASTERS_ALL if f in data.columns and f != "random"]
    for name in forecasters:
        assert name in data.columns, (
            f"invalid forecaster name {name}. "
            f"available: {data.columns.to_list()}"
        )

    set_theme(**theme_kwargs)
    colors = get_colors()
    normal_colors = colors[:3] + colors[4:7] + colors[8:]  # remove red, gray
    gray = colors[7]
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    plt.figure(figsize=figsize, facecolor="white")
    if "true_probs" in data.columns:
        plt.scatter(data.time, data.true_probs, marker=".", alpha=0.3,
                    color=gray, label=r"reality ($r_t$)")
    elif "y" in data.columns:
        plt.scatter(data.time, data.y, marker=".", alpha=0.3,
                    color=gray, label=r"data ($y_t$)")
    for i, name in enumerate(forecasters):
        plt.plot(data.time, data[name], linewidth=linewidth, alpha=0.8,
                 color=normal_colors[i % len(normal_colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 label=name)
    plt.title("Forecasters", fontsize="large")
    plt.xlabel("Time")
    plt.ylabel("Probability Forecast")
    plt.ylim(-0.05, 1.05)
    if use_logx:
        plt.xscale("log")
        plt.xlim(10, len(data))
    if legend_out:
        plt.legend(loc="lower right", bbox_to_anchor=(1.32, 0), frameon=False)
    else:
        plt.legend(loc="best")
    plt.tight_layout()
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(
            os.path.join(plots_dir, f"forecasters.{savefig_ext}"),
            dpi=savefig_dpi,
            transparent=True,
        )


LABELS = {
    "y": r"$\hat\Delta_t$",
    "true": r"$\Delta_t$",
    "cs": "EB CS",  # default
    "h": "Hoeffding CS",
    "acs": "Asymptotic CS",
    "ci": "Fixed-Time CI",
    "dm": "DM Test",
    "gw": "GW Test",
}
COLORS = {
    "y": get_color_by_index(7),      # gray
    "true": "darkred",               # darkred
    "cs": get_color_by_index(0),     # blue
    "h": get_color_by_index(9),      # skyblue
    "ci": get_color_by_index(1),     # orange
    "acs": get_color_by_index(2),    # green
    "dm": get_color_by_index(2),     # green
    "gw": get_color_by_index(3),     # darkorange
}
LINESTYLES = {
    "y": "solid",
    "true": "solid",
    "cs": "solid",
    "h": "dashdot",
    "acs": "dashed",
    "ci": "dotted",
    # not to be used alongside acs/h
    "dm": "dashed",
    "gw": "dashdot",
}


def plot_comparison(
        data: Union[str, pd.DataFrame],
        name_p: str,
        name_q: str,
        scoring_rule: str = "brier",
        lag: int = 1,
        aligned_outcomes: bool = True,
        plots_dir: str = "./plots",
        alpha: float = 0.05,
        boundary_type: str = "mixture",
        v_opt: float = 10,
        baselines: tuple = (),
        plot_e: bool = True,
        plot_width: bool = True,
        plot_diagnostics: bool = False,
        diagnostics_fn: str = "miscoverage",
        diagnostics_baselines: tuple = ("ci", ),
        n_repeats: int = 1000,
        use_logx: bool = False,
        linewidth: int = 2,
        xlim: tuple = None,
        ylim_scale: float = 0.6,
        no_title: bool = False,
        savefig_ext: str = "pdf",
        savefig_dpi: float = 300,
        **theme_kwargs
) -> Tuple[pd.DataFrame, plt.Axes]:
    """Compare two sequential forecasters by plotting
     confidence sequences and e-processes for their average forecast score differentials.

    Produces up to four plots:
        1. Plot of confidence sequences/intervals on average score differentials
        2. (Optional) Widths of confidence sequences/intervals
        3. (Optional) Plot of e-processes for the nulls that "p is no better/worse than q on average"
        4. (Optional) Diagnostics plot (umulative miscoverage rates or false decision rate

    Args:
        data: pandas dataframe or path to a saved csv containing
            forecasts and data as columns
            (e.g., output of :py:func:`comparecast.forecasters.forecast`.)
        name_p: column name of the first forecaster
        name_q: column name of the second forecaster
        scoring_rule: name of the scoring rule to be used (default: brier)
        lag: forecast lag. Currently requires compute_cs is False if lag > 1. (default: 1)
        aligned_outcomes: whether the outcomes are aligned with the forecasts, if lag > 1.
            (default: True)
        plots_dir: directory to store all plots (default: ./plots)
        alpha: significance level for confidence sequences (default: 0.05)
        boundary_type: type of uniform boundary used for CS (default: mixture)
        v_opt: value of intrinsic time when the boundary is optimized.
            Default is 10; set to ``None`` in *post-hoc* analyses (only)
            to optimize the boundary at the last time step.
        baselines: compare with other baselines provided, including
            Hoeffding-style CS (``h``), asymptotic CS (``acs``), and
            fixed-time CI (``ci``). (default: ``()``)
        plot_e: if True, also plot e-processes (default: True)
        n_repeats: number of repeated trials for miscoverage rate calculation
            (default: 10000)
        plot_width: if true, also plot the width of CS/CI
        plot_diagnostics: if true, also plot a diagnostics plot of the CS/CIs
        diagnostics_fn: which diagnostics function to use (default: miscoverage)
        diagnostics_baselines: which baseline CS/CIs to include for diagnostics
            (default: ``("ci", )``)
        use_logx: if true, use the logarithmic scale on the x-axis.
        linewidth: line width. (default: 2)
        xlim: x-axis (time) limits as a tuple. (default: None)
        ylim_scale: scale of the y-axis limit for the CS plot. (default: 0.6)
        no_title: do not add a title to the plot. (default: False)

    Returns:
        A tuple of two items.
            1. A ``pd.DataFrame`` containing the score differentials (deltas)
               and the confidence sequences (cs)
            2. A ``matplotlib.pyplot.Axes`` object
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)

    T = len(data)
    ps, qs, ys, times = [
        data[name].values for name in [name_p, name_q, "y", "time"]
    ]

    # CS/CI & e-process calculations
    results = compare_forecasts(
        data, name_p, name_q,
        scoring_rule=scoring_rule,
        lag=lag, aligned_outcomes=aligned_outcomes,
        alpha=alpha, boundary_type=boundary_type, v_opt=v_opt,
        compute_e=plot_e)
    suffixes = {"cs": ""}
    if "h" in baselines:
        results_h = compare_forecasts(
            data, name_p, name_q,
            scoring_rule=scoring_rule,
            lag=lag, aligned_outcomes=aligned_outcomes,
            alpha=alpha, use_hoeffding=True,
            boundary_type=boundary_type, v_opt=v_opt,
            compute_e=False)
        suffixes["h"] = "_h"
        results = results.merge(results_h, on=["time"], suffixes=(None, suffixes["h"]))
    if "acs" in baselines:
        results_acs = compare_forecasts(
            data, name_p, name_q,
            scoring_rule=scoring_rule,
            lag=lag, aligned_outcomes=aligned_outcomes,
            alpha=alpha, use_asymptotic=True,
            compute_e=False)
        suffixes["acs"] = "_acs"
        results = results.merge(results_acs, on=["time"], suffixes=(None, suffixes["acs"]))
    if "ci" in baselines:
        lcbs_ci, ucbs_ci = confint_lai(
            ps, qs, ys, None, scoring_rule=scoring_rule, alpha=alpha)
        results_ci = pd.DataFrame({
            "time": times,
            "lcb": lcbs_ci,
            "ucb": ucbs_ci,
        })
        suffixes["ci"] = "_ci"
        results = results.merge(results_ci, on=["time"], suffixes=(None, suffixes["ci"]))

    # Plot true deltas if true_probs or true_means exist in data
    if "true_probs" in data:
        results["true_means"] = data["true_probs"].values
    elif "true_means" in data:
        results["true_means"] = data["true_means"].values
    has_true_mean = "true_means" in results
    if has_true_mean:
        results["true_deltas"] = compute_true_deltas(
            ps, qs, results["true_means"], scoring_rule)
        logging.info(f"True Delta [T=%d]: %.5f",
                     T, results.true_deltas.tail(1).item())

    # Diagnostics
    if plot_diagnostics:
        assert has_true_mean, "plotting diagnostics require having true means"
        diagnostics = compute_diagnostics(
            data, name_p, name_q,
            diagnostics_fn=diagnostics_fn,
            n_repeats=n_repeats,
            scoring_rule=scoring_rule,
            alpha=alpha,
            # boundary_type="stitching",  # to save time
            boundary_type="mixture",
            baselines=diagnostics_baselines,
        )
    else:
        diagnostics = {}

    # Set up grids for the plot: 1x1, 1x2, 1x3, or 2x2
    set_theme(**theme_kwargs)
    n_figures = 1 + plot_width + plot_e + plot_diagnostics
    fig_properties = {
        1: dict(nrows=1, ncols=1, figsize=(5, 4)),
        2: dict(nrows=1, ncols=2, figsize=(12, 5)),
        3: dict(nrows=1, ncols=3, figsize=(15, 5)),
        4: dict(nrows=2, ncols=2, figsize=(15, 10)),
    }

    fig, axes = plt.subplots(**fig_properties[n_figures])
    # constrained_layout=True, facecolor="white")

    # get axes as a list
    if n_figures == 1:
        axes = [axes]
    elif n_figures == 4:
        axes = [a for ax in axes for a in ax]

    # set x-axis properties
    if use_logx:
        xscale = "log"
        xlim = (100, T) if xlim is not None else None
    else:
        xscale = "linear"

    # compute bounds and y-axis limits based on scoring rules
    if scoring_rule == "winkler":
        q0 = min(min(qs), min(1 - qs))
        lo, hi = 1 - 2 / q0, 1
    else:
        a, b = get_scoring_rule(scoring_rule).bounds
        lo, hi = a - b, b - a
    y_radius = ylim_scale * hi if abs(lo) == hi else (ylim_scale / 4) * (hi - lo)
    ylim = (-y_radius, y_radius)

    i = 0

    # Plot 1: confidence sequences & intervals
    # axes[i].axhline(color=COLORS["y"], alpha=0.5, linewidth=linewidth)
    axes[i].axhline(y=0, color="black", alpha=0.5, linewidth=linewidth)
    for name, suffix in suffixes.items():
        axes[i].plot(times, results["ucb" + suffix],
                     alpha=0.8, color=COLORS[name],
                     linestyle=LINESTYLES[name], linewidth=linewidth,
                     label=LABELS[name])
        axes[i].plot(times, results["lcb" + suffix],
                     alpha=0.8, color=COLORS[name],
                     linestyle=LINESTYLES[name], linewidth=linewidth)
    if has_true_mean:
        axes[i].plot(times, results["true_deltas"],
                     alpha=0.8, color=COLORS["true"],
                     linestyle=LINESTYLES["true"], linewidth=linewidth + 2,  # to distinguish with cs
                     label=LABELS["true"])

    cs_str = "CS/CI" if "ci" in baselines else "CS"
    param_str = r"$\Delta_t$" if scoring_rule != "winkler" else r"$W_t$"
    axes[i].set(
        xscale=xscale,
        xlim=xlim,
        ylim=ylim,
        xlabel="Time" if n_figures <= 3 else None,
        # ylabel=cs_ci + r" for $\Delta_t$",
    )
    axes[i].set_title(f"{100 * (1 - alpha):g}% {cs_str} for {param_str}",
                      fontweight="bold", fontsize="large")
    centers = (results["ucb"].tail(1) + results["lcb"].tail(1)) / 2
    axes[i].legend(loc="upper right" if centers.mean() < 0 else "lower right",
                   fontsize="medium")
    i += 1

    # Plot 2: CS width
    if plot_width:
        for name, suffix in suffixes.items():
            axes[i].plot(times,
                         results["ucb" + suffix] - results["lcb" + suffix],
                         alpha=0.8,
                         color=COLORS[name],
                         linestyle=LINESTYLES[name],
                         linewidth=linewidth,
                         label=LABELS[name])
        axes[i].set(
            xscale=xscale,
            xlim=xlim,
            ylim=(0, (hi - lo) * 0.25),
            xlabel="Time" if n_figures <= 3 else None,
            # ylabel="Width",
        )
        axes[i].set_title("Width of CS" + ("/CI" if "ci" in baselines else ""),
                          fontweight="bold", fontsize="large")
        axes[i].legend(ncol=2 if n_figures in [2, 4] else 1,
                       fontsize="medium")
        i += 1

    # Plot 3: e-process
    # horizontal line is at 2/alpha, as each e-process is one-sided.
    if plot_e:
        axes[i].axhline(y=1, color="black", alpha=0.5, linewidth=linewidth)
        axes[i].axhline(y=2 / alpha, color=COLORS["y"], alpha=0.8,
                        linestyle="dotted", linewidth=1.5)
        axes[i].plot(times,
                     results["e_pq"],
                     color=get_color_by_index(5),  # brown
                     linestyle="dashed",
                     linewidth=linewidth,
                     label=r"$H_0: \Delta_t \leq 0, \forall t$")
        axes[i].plot(times,
                     results["e_qp"],
                     color=get_color_by_index(4),  # purple
                     linestyle="solid",
                     linewidth=linewidth,
                     label=r"$H_0: \Delta_t \geq 0, \forall t$")
        axes[i].set(
            xscale=xscale,
            xlim=xlim,
            yscale="log",
            ylim=(10**-4, 10**4),
            xlabel="Time",
            # ylabel="E-Process (log-scale)",
        )
        axes[i].set_title("E-Process (log-scale)", fontweight="bold", fontsize="large")
        axes[i].legend(loc="lower left", fontsize="medium")
        i += 1

    # Plot 4: diagnostics
    if plot_diagnostics:
        # significance level
        axes[i].axhline(y=alpha, color=COLORS["y"], alpha=0.8,
                        linestyle="dotted", linewidth=1.5)
        # method can be cs, ci, dm, gw
        for method in diagnostics:
            # can contain NaNs
            valid = np.isfinite(diagnostics[method])
            axes[i].plot(times[valid], diagnostics[method][valid],
                         linestyle=LINESTYLES[method], linewidth=linewidth,
                         color=COLORS[method], label=LABELS[method])
        # axes[i].plot(times, np.repeat(alpha, T),
        #              linestyle=LINESTYLES["y"], linewidth=linewidth,
        #              color=COLORS["y"], label="Significance Level")
        if diagnostics_fn == "miscoverage":
            rate_name = "Cumulative Miscoverage Rate"
        elif diagnostics_fn == "fder":
            rate_name = "False Decision Rate"
        elif diagnostics_fn == "cfdr":
            rate_name = "Cumulative Type I Error"
        else:
            rate_name = "Diagnostics"
        axes[i].set(
            xscale=xscale,
            xlim=xlim,
            ylim=(-0.025, 1.025),
            # ylim=(-0.025, 0.625),
            xlabel="Time",
        )
        axes[i].set_title(rate_name, fontweight="bold", fontsize="large")
        axes[i].legend(ncol=2 if n_figures in [2, 4] else 1,
                       fontsize="medium")
        i += 1

    # Title
    if scoring_rule == "winkler":
        score_name = "WinklerScore"
        target_str = r"$W_t$" f"({name_p}, {name_q}); S=BrierScore"
    else:
        score_name = get_scoring_rule(scoring_rule).name
        target_str = r"$\Delta_t$" f"({name_p}, {name_q}); S={score_name}"
    # str_t = f"$T=10^{np.log10(T):.2g}$" if use_logx else f"T={T}"
    # fig.suptitle(f"{(1-alpha)*100:2.0f}% CS on {target_str}",
    #              fontweight="regular")
    if not no_title:
        fig.suptitle(target_str)
    if n_figures <= 3:
        fig.subplots_adjust(top=0.8, wspace=0.0, hspace=0.1)
        fig.tight_layout()
    else:
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        fig.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(
            os.path.join(plots_dir, f"comparecast_cs_{name_p}_{name_q}_{score_name}.{savefig_ext}"),
            dpi=savefig_dpi,
            transparent=True,
        )

    # Store auxiliary info and return the results data frame along with axes
    results.update({
        "y": ys,
        "p": ps,
        "q": qs,
    })
    if plot_diagnostics:
        for method in diagnostics:
            results[f"{diagnostics_fn}_{method}"] = diagnostics[method]

        if plots_dir:
            import pickle

            with open(os.path.join(plots_dir, "results.pkl"), "wb") as f:
                pickle.dump(results, f)
    return results, axes


def plot_pairwise_comparisons(
        data: Union[str, pd.DataFrame],
        forecasters: List[str],
        scoring_rule: str = "brier",
        lag: int = 1,
        aligned_outcomes: bool = True,
        plots_dir: str = "./plots",
        alpha: float = 0.05,
        boundary_type: str = "mixture",
        v_opt: float = 10,
        baselines: tuple = (),
        use_logx: bool = True,
        linewidth: int = 2,
        xlim: tuple = None,
        ylim_scale: float = 0.4,
        savefig_ext: str = "pdf",
        savefig_dpi: float = 300,
        **theme_kwargs
) -> plt.Axes:
    """Plot pairwise comparisons of forecasters in data.

    Args:
        data: pandas dataframe or path to a saved csv containing
            forecasts and data as columns
            (e.g., output of :py:func:`comparecast.forecasters.forecast`.)
        forecasters: list of forecasters to be compared against each other
        scoring_rule: name of the scoring rule to be used (default: brier)
        lag: forecast lag. Currently requires compute_cs is False if lag > 1. (default: 1)
        aligned_outcomes: whether the outcomes are aligned with the forecasts, if lag > 1.
            (default: True)
        plots_dir: directory to store all plots (default: ./plots)
        alpha: significance level for confidence sequences (default: 0.05)
        boundary_type: type of uniform boundary used for CS (default: mixture)
        v_opt: value of intrinsic time where the boundary is optimized.
            Default is 10; set to ``None`` in *post-hoc* analyses (only)
            to optimize the boundary at the last time step.
        baselines: compare with other baselines provided, including
            Hoeffding-style CS (``h``), fixed-time CI (``ci``), and
            asymptotic CS (``acs``). (default: ``()``)
        use_logx: if true, use the logarithmic scale on the x-axis.
        linewidth: line width. (default: 2)
        xlim: x-axis (time) limits as a tuple. (default: None)
        ylim_scale: scale of the y-axis limit for the CS plot. (default: 0.4)

    Returns:
        A ``matplotlib.pyplot.Axes`` object
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)
    for name in forecasters:
        assert name in data.columns, f"forecaster {name} not found in data"

    score = get_scoring_rule(scoring_rule)
    if scoring_rule == "winkler":
        raise ValueError(
            f"pairwise comparison not supported for winkler's score")
    else:
        a, b = score.bounds
        lo, hi = a - b, b - a
    y_radius = ylim_scale * hi if abs(lo) == hi else (ylim_scale / 4) * (hi - lo)
    ylim = (-y_radius, y_radius)

    n = len(forecasters)
    if n > 5:
        logging.warning("too many pairwise comparisons for %d forecasters", n)

    set_theme(**theme_kwargs)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 4 * n), facecolor="white")
    T = len(data)
    times = np.arange(1, T + 1)
    ys = data["y"].values

    # for global legend
    handles = []

    for i, name_p in enumerate(forecasters):
        for j, name_q in enumerate(forecasters):
            if i == j:
                axes[i][j].set_visible(False)
                continue

            ps = data[name_p].values
            qs = data[name_q].values

            # CS/CI calculations (TODO: refactor)
            results = compare_forecasts(
                data, name_p, name_q,
                scoring_rule=scoring_rule,
                lag=lag, aligned_outcomes=aligned_outcomes,
                alpha=alpha, boundary_type=boundary_type, v_opt=v_opt,
                compute_e=False)
            suffixes = {"cs": ""}
            if "h" in baselines:
                results_h = compare_forecasts(
                    data, name_p, name_q,
                    scoring_rule=scoring_rule,
                    lag=lag, aligned_outcomes=aligned_outcomes,
                    alpha=alpha, use_hoeffding=True,
                    boundary_type=boundary_type, v_opt=v_opt,
                    compute_e=False)
                suffixes["h"] = "_h"
                results = results.merge(results_h, on=["time"], suffixes=(None, suffixes["h"]))
            if "acs" in baselines:
                results_acs = compare_forecasts(
                    data, name_p, name_q,
                    scoring_rule=scoring_rule,
                    lag=lag, aligned_outcomes=aligned_outcomes,
                    alpha=alpha, use_asymptotic=True,
                    compute_e=False)
                suffixes["acs"] = "_acs"
                results = results.merge(results_acs, on=["time"], suffixes=(None, suffixes["acs"]))
            if "ci" in baselines:
                lcbs_ci, ucbs_ci = confint_lai(
                    ps, qs, ys, None, scoring_rule=scoring_rule, alpha=alpha)
                results_ci = pd.DataFrame({
                    "time": times,
                    "lcb": lcbs_ci,
                    "ucb": ucbs_ci,
                })
                suffixes["ci"] = "_ci"
                results = results.merge(results_ci, on=["time"], suffixes=(None, suffixes["ci"]))

            # Plot true deltas if true_probs or true_means exist in data
            if "true_probs" in data:
                results["true_means"] = data["true_probs"].values
            elif "true_means" in data:
                results["true_means"] = data["true_means"].values
            has_true_mean = "true_means" in results
            if has_true_mean:
                results["true_deltas"] = compute_true_deltas(
                    ps, qs, results["true_means"], scoring_rule)
                logging.info(f"True Delta [T=%d]: %.5f",
                             T, results.true_deltas.tail(1).item())

            # Plot the (i,j)-pair
            axes[i][j].axhline(y=0, color="black", alpha=0.5, linewidth=linewidth)
            if has_true_mean:
                true_deltas = compute_true_deltas(ps, qs, results["true_means"].values,
                                                  scoring_rule)
                lines = axes[i][j].plot(
                    times, true_deltas,
                    alpha=0.8, color=COLORS["true"],
                    linestyle=LINESTYLES["true"], linewidth=linewidth + 2,
                    label=LABELS["true"],
                )
                if (i, j) == (0, 1):
                    handles.append(lines[0])
            for name, suffix in suffixes.items():
                # ucb
                lines = axes[i][j].plot(
                    times, results["ucb" + suffix],
                    alpha=0.8, color=COLORS[name],
                    linestyle=LINESTYLES[name], linewidth=linewidth,
                    label=LABELS[name],
                )
                # lcb
                axes[i][j].plot(
                    times, results["lcb" + suffix],
                    alpha=0.8, color=COLORS[name],
                    linestyle=LINESTYLES[name], linewidth=linewidth,
                )
                if (i, j) == (0, 1):
                    handles.append(lines[0])
            axes[i][j].set_xlim(xlim)
            axes[i][j].set_ylim(ylim)
            if i == len(forecasters) - 1:
                axes[i][j].set_xlabel("Time")
            if j == 0:
                axes[i][j].set_ylabel(r"CS for $\Delta_t$")
            if use_logx:
                axes[i][j].set_xscale("log")
            lcb = results["lcb"].tail(1).item()
            ucb = results["ucb"].tail(1).item()
            axes[i][j].set_title(
                r"$\Delta_t$" f"({name_p}, {name_q}): ({lcb:.3f}, {ucb:.3f})"
            )
            # axes[i][j].legend(loc=("upper right"
            #                        if lcb + ucb < 0
            #                        else "lower right"),
            #                   fontsize="small")

    if scoring_rule == "winkler":
        score_name = "WinklerScore"
    else:
        score_name = get_scoring_rule(scoring_rule).name
    fig.suptitle(f"{(1 - alpha) * 100:2.0f}% "
                 r"Confidence Sequences on $\Delta_t$"
                 f"; S={score_name}",
                 fontsize="x-large")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    # global legend
    plt.figlegend(handles=handles, loc="center", fontsize="large",
                  bbox_to_anchor=(0.6/n, 1 - 0.6/n))
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(
            os.path.join(plots_dir, f"comparecast_cs_{n}x{n}_{score_name}.{savefig_ext}"),
            dpi=savefig_dpi,
            transparent=True,
        )

    return axes


def plot_ucbs(
        ucbs: pd.DataFrame,
        use_preset_theme: bool = True,
        save_filename: str = "confseq_ucbs.pdf",
        figsize: Tuple[int, int] = (8, 5),
        linewidth: int = 2,
        savefig_dpi: float = 300,
        **plot_kwargs
):
    """Plot the UCBs of multiple confidence sequences across time.

    Useful for quickly comparing the width of many CS.

    Args:
        ucbs: a ``pd.DataFrame`` with UCBs.
        use_preset_theme: whether to use preset theme (default: `True`).
        save_filename: filename to save the resulting plot.
        figsize: output figure size.
        linewidth: line width.
        **plot_kwargs: any other arguments to ``matplotlib.pyplot.Axes.set()``.

    Returns:
        None
    """
    if use_preset_theme:
        set_theme()
    if "figsize" in plot_kwargs:
        plt.figure(figsize=plot_kwargs["figsize"])
        del plot_kwargs["figsize"]
    ucbs["Time"] = np.arange(1, len(ucbs) + 1)
    df = ucbs.melt(id_vars=["Time"], value_vars=None,
                   var_name="Method", value_name="UCB")
    plt.figure(figsize=figsize)
    ax = sns.lineplot(x="Time", y="UCB", hue="Method", linewidth=linewidth, data=df)
    ax.set(**plot_kwargs)
    plt.tight_layout()
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=savefig_dpi, transparent=True)


def plot_mlb_forecasts(
        data: pd.DataFrame,
        team: str,
        years: List[int],
        forecasters: Iterable[str],
        no_playoffs: bool = False,
        window: Tuple[str, str] = ("2019-10-22", "2019-10-30"),
        colors: List = get_colors(),
        n_games: int = None,
        save_filename: str = None,
        savefig_dpi: float = 300,
):
    """Plotting function for MLB forecasts."""
    set_theme()
    plt.figure(figsize=(12, 5), facecolor="white")
    filtered_data = data[data.season.isin(years)]
    if len(years) == 1 and n_games is not None:
        filtered_data = filtered_data.tail(n_games)
    filtered_data = filtered_data.copy().reset_index()
    if no_playoffs:
        filtered_data = filtered_data[filtered_data.playoff.isna()]
    df = filtered_data[["time"] + [f for f in forecasters]].melt(
        "time", var_name="forecasts", value_name="probability")
    sns.set_palette(colors)
    ax = sns.lineplot(x="time", y="probability",
                      hue="forecasts", style="forecasts",
                      hue_order=forecasters,
                      linewidth=2, alpha=0.9, data=df)
    sns.scatterplot(x="time", y="win", color=COLORS["y"],
                    data=filtered_data, ax=ax)

    span0 = data[data.date == window[0]]["time"].item()
    span1 = data[data.date == window[1]]["time"].item()
    ax.axvspan(span0, span1, alpha=0.3, color='gray')

    team = "Home" if team == "MLB" else team
    if len(years) == 1:
        xticks = []
        xticklabels = []
        suffix = (str(years[0]) if n_games is None
                  else f"last {n_games} of {years[0]}")
    else:
        xticks = np.zeros_like(years)
        xticks[1:] = np.where(np.diff(filtered_data.season))[0]
        xticklabels = years
        suffix = f"{years[0]}-{years[-1]}"
    ax.set(
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel=f"Games ({suffix})",
        ylabel="Probability/Outcome",
        ylim=(-0.05, 1.05),
    )
    ax.set_title(
        f"{team} Team Win Probability Forecasts" +
        (f" (regular seasons only)" if no_playoffs else " (gray: playoffs)"),
    )
    ax.legend(loc="lower left", fontsize="small", ncol=2)
    plt.tight_layout()

    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=savefig_dpi, transparent=True)


def plot_weather_hz_evalues(
        evalues: pd.DataFrame,
        use_preset_theme: bool = True,
        save_filename: str = None,
        savefig_dpi: float = 300,
        **theme_kwargs,
):
    """Reproduces Figure 3 from Henzi & Ziegel (2021).

    A/B: A is not better than B under the null hypothesis
    """
    if use_preset_theme:
        set_theme(**theme_kwargs)

    fg = sns.relplot(
        x="Date",
        y="E-value",
        col="Hypothesis",
        hue="Airport",
        style="Airport",
        kind="line",
        linewidth=2,
        height=5,
        aspect=1,
        data=evalues,
    )
    for ax in fg.axes[0]:
        ax.set(
            xlabel="Year",
            yscale="log",
            ylim=(1e-2, 1e4)
        )
        ax.axhline(y=1, linewidth=1.5, linestyle="solid", color="black")
        ax.axhline(y=20, linewidth=1.5, linestyle="dotted", color="gray")

    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=savefig_dpi, transparent=True)


def plot_weather_comparison(
        data_dir: str = "eprob/replication_material/precip_fcs",
        lag: int = 1,
        lagged_null: str = "pw",
        scoring_rule: str = "brier",
        alpha: float = 0.1,
        v_opt: float = 0.5,
        c: float = 0.1,
        compute_cs: bool = True,
        compute_e: bool = True,
        no_calibration: bool = False,
        calibration_strategy: str = "mixture",
        use_hz: bool = False,
        alt_prob: float = 0.75,
        plots_dir: str = "./plots/weather",
        use_preset_theme: bool = True,
        ylim_scale: float = 0.05,
        savefig_ext: str = "pdf",
        savefig_dpi: float = 300,
        **theme_kwargs,
):
    """Produces CS and e-value plots for the weather forecast comparison.

    Also returns a dataframe containing the CS and the e-values.

    If use_hz is True, uses Henzi and Ziegel's e-process
    for conditional forecast dominance (strong null), and no CS is provided.
    """
    assert lag in LAGS, f"invalid lag {lag}, available: {LAGS}"

    if use_preset_theme:
        set_theme(**theme_kwargs)

    df_rows = []
    pairs = [("hclr", "idr"), ("idr", "hclr_noscale"), ("hclr", "hclr_noscale")]
    for airport in AIRPORTS:
        pop_fcs = read_precip_fcs(data_dir, pop_only=True)
        pop_fcs = pop_fcs[(pop_fcs["airport"] == airport) &
                          (pop_fcs["lag"] == lag)].sort_values(["date"], ascending=True)
        for name_p, name_q in pairs:
            try:
                if use_hz:
                    assert 0 < alt_prob < 1, "alt_prob should be within (0, 1)"
                    T = len(pop_fcs)
                    ps, qs, ys = [
                        pop_fcs[name].values for name in [name_p, name_q, "y"]
                    ]
                    e_pq = eprocess_hz(
                        ps, qs, ys,
                        aligned_outcomes=True,
                        scoring_rule=scoring_rule,
                        lag=lag,
                        alt_prob=alt_prob,
                    )
                    e_qp = eprocess_hz(
                        qs, ps, ys,
                        aligned_outcomes=True,
                        scoring_rule=scoring_rule,
                        lag=lag,
                        alt_prob=alt_prob,
                    )
                    results = pd.DataFrame({
                        "time": np.arange(1, T + 1),
                        "e_pq": e_pq,
                        "e_qp": e_qp,
                        "lcb": None,
                        "ucb": None,
                    })
                else:
                    results = compare_forecasts(
                        pop_fcs,
                        name_p,
                        name_q,
                        scoring_rule=scoring_rule,
                        lag=lag,
                        lagged_null=lagged_null,
                        aligned_outcomes=True,
                        compute_cs=compute_cs and lag == 1,
                        alpha=alpha,
                        compute_e=compute_e,
                        v_opt=v_opt,
                        c=c,
                        no_calibration=no_calibration,
                        calibration_strategy=calibration_strategy,
                    )
            except ValueError:
                raise ValueError(f"{airport}, {lag}, {name_p}, {name_q}")
            name_p, name_q = [
                {
                    "idr": "IDR",
                    "hclr": "HCLR",
                    "hclr_noscale": "HCLR_",
                }[name]
                for name in [name_p, name_q]
            ]
            for date, lcb, ucb, evalue in zip(
                    pop_fcs.date, results.lcb, results.ucb, results.e_pq):
                df_rows.extend([
                    {
                        "Airport": airport,
                        "Lag": lag,
                        "Date": date,
                        "Hypothesis": "/".join([name_p, name_q]),
                        "OutputType": otype,
                        "Value": value,
                    }
                    for otype, value in zip(["LCB", "UCB", "E-value"],
                                            [lcb, ucb, evalue])
                ])
    results = pd.DataFrame(df_rows)

    # Plot 1: CS
    if not use_hz and (compute_cs and lag == 1):
        fg = sns.relplot(
            x="Date",
            y="Value",
            col="Hypothesis",
            hue="Airport",
            style="Airport",
            size="OutputType",
            sizes=[2, 2],
            kind="line",
            height=5,
            aspect=1,
            linewidth=2,
            data=results[results["OutputType"] != "E-value"],
        )
        for ax, (name_p, name_q) in zip(fg.axes[0], pairs):
            name_p, name_q = [
                {
                    "idr": "IDR",
                    "hclr": "HCLR",
                    "hclr_noscale": "HCLR_",
                }[name]
                for name in [name_p, name_q]
            ]
            ax.set(
                xlabel="Year",
                ylabel=r"CS for $\Delta_t$",
                title=f"{100*(1-alpha):g}% CS on " r"$\Delta_t$"
                      f"({name_p}, {name_q})",
                ylim=(-ylim_scale, ylim_scale)
            )
            ax.axhline(y=0, linewidth=2, linestyle="-", color="black")

        # remove unnecessary legend items
        for handle, text in zip(fg._legend.legendHandles[5:],
                                fg._legend.texts[5:]):
            handle.set_data([], [])
            text.set_text("")

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(
                os.path.join(plots_dir, f"comparecast_cs_lag{lag}.{savefig_ext}"),
                dpi=savefig_dpi,
                bbox_inches="tight",
                transparent=True,
            )

    # Plot 2: e-processes
    fg = sns.relplot(
        x="Date",
        y="Value",
        col="Hypothesis",
        hue="Airport",
        style="Airport",
        kind="line",
        height=5,
        aspect=1,
        linewidth=2,
        data=results[results["OutputType"] == "E-value"],
    )
    for ax, (name_p, name_q) in zip(fg.axes[0], pairs):
        name_p, name_q = [
            {
                "idr": "IDR",
                "hclr": "HCLR",
                "hclr_noscale": "HCLR_",
            }[name]
            for name in [name_p, name_q]
        ]
        ax.set(
            xlabel="Year",
            ylabel=r"E-Process (log-scale)",
            title=r"$H_0: \delta_t$" f"({name_p}, {name_q})" r"$\leq 0,\; \forall t$" if use_hz
                   else r"$H_0: \Delta_t$" f"({name_p}, {name_q})" r"$\leq 0,\; \forall t$",
            yscale="log",
            ylim=(1e-2, 1e4),
        )
        ax.axhline(y=1, linewidth=1.5, linestyle="solid", color="black")
        ax.axhline(y=2/alpha, linewidth=1.5, linestyle="dotted", color="gray")

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        evalue_type = "hz" if use_hz else lagged_null
        plt.savefig(
            os.path.join(plots_dir, f"comparecast_evalues_{evalue_type}_lag{lag}.{savefig_ext}"),
            dpi=savefig_dpi,
            bbox_inches="tight",
            transparent=True,
        )

    return results
