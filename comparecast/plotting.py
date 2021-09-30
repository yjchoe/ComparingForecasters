"""
Some plotting utilities
"""

import logging
import os.path
from typing import Iterable, List, Tuple, Union
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from comparecast.comparecast import compare_forecasts
from comparecast.forecasters import FORECASTER_NAMES
from comparecast.scoring import get_scoring_rule
from comparecast.confint import confint_lai
from comparecast.diagnostics import compute_miscoverage, compute_fder
from comparecast.data_utils.weather import AIRPORTS, LAGS, read_precip_fcs


PYPLOT_DEFAULT_STYLE = "seaborn-colorblind"
SEABORN_DEFAULT_PALETTE = "colorblind"


def set_style(style_name: str = PYPLOT_DEFAULT_STYLE):
    """Set default plotting style."""
    plt.style.use(style_name)


def get_color_by_index(index: int, palette: str = SEABORN_DEFAULT_PALETTE):
    """Get color by integer indices."""
    colors = sns.color_palette(palette)
    return colors[index % len(colors)]


def get_colors(palette: str = SEABORN_DEFAULT_PALETTE):
    """Get colors defined by a seaborn palette."""
    return sns.color_palette(palette)


def plot_forecasts(
        data: pd.DataFrame,
        forecasters: Iterable[str],
        plots_dir: str = "./plots",
        use_logx: bool = True,
):
    """Plot forecasts along with the dataset.

    Also plots ``data`` or ``true_probs``, if those columns are available in
    ``data``.

    Args:
        data: A ``pd.DataFrame`` containing a set of forecasts as its columns.
            Must contain columns ``time`` and each element of ``forecasters``.
        forecasters: list of forecasters (columns of ``data``) to be plotted.
        plots_dir: directory to save the plots.
        use_logx: whether to use the log-scale on the time (x) axis.

    Returns: None
        Saves a plot to ``{plots_dir}/forecasters.pdf``.
    """
    if "all" in forecasters:
        forecasters = FORECASTER_NAMES
    for name in forecasters:
        assert name in data.columns, (
            f"invalid forecaster name {name}. "
            f"available: {data.columns.to_list()}"
        )

    set_style()
    colors = get_colors()
    normal_colors = colors[:3] + colors[4:7] + colors[8:]  # remove red, gray
    plt.figure(figsize=(12, 4), facecolor="white")
    if "true_probs" in data.columns:
        plt.scatter(data.time, data.true_probs, marker=".", alpha=0.7,
                    color=colors[7], label=r"reality ($r_t$)")
    elif "data" in data.columns:
        plt.scatter(data.time, data.data, marker=".", alpha=0.7,
                    color=colors[7], label=r"data ($y_t$)")
    for i, name in enumerate(forecasters):
        plt.plot(data.time, data[name], linewidth=1.5, alpha=0.9,
                 color=normal_colors[i % len(normal_colors)], label=name)
    plt.title("Forecasters", fontweight="regular", fontsize="13")
    plt.xlabel("Time")
    plt.ylabel("Probability/Outcome")
    plt.ylim(-0.05, 1.05)
    if use_logx:
        plt.xscale("log")
        plt.xlim(10, len(data))
    plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0), frameon=False)
    plt.tight_layout()
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "forecasters.pdf"))


LABELS = {
    "data": r"$\hat\Delta_t$",
    "true": r"$\Delta_t$",
    "cs": "EB CS",  # default
    "h": "Hoeffding CS",
    "acs": "Asymptotic CS",
    "ci": "Fixed-Time CI",
}
COLORS = {
    "data": get_color_by_index(7),   # gray
    "true": "darkred",               # darkred
    "cs": get_color_by_index(0),     # blue
    "h": get_color_by_index(9),      # skyblue
    "acs": get_color_by_index(2),    # green
    "ci": get_color_by_index(1),     # orange
}
LINESTYLES = {
    "data": "solid",
    "true": "solid",
    "cs": "solid",
    "h": "dashed",
    "acs": "dashdot",
    "ci": "dotted",
}


def plot_comparison(
        data: Union[str, pd.DataFrame],
        name_p: str,
        name_q: str,
        scoring_rule: str = "brier",
        plots_dir: str = "./plots",
        alpha: float = 0.05,
        lo: float = -1.,
        hi: float = 1.,
        boundary_type: str = "mixture",
        v_opt: float = 10,
        compare_baselines: tuple = (),
        plot_fder: bool = False,
        plot_miscoverage: bool = False,
        n_repeats: int = 10000,
        plot_width: bool = True,
        use_logx: bool = True,
) -> Tuple[pd.DataFrame, plt.Axes]:
    """Compare two forecasting strategies and plot confidence sequences on
    their average forecast score differentials.

    Produces up to three plots:
        1. Plot of confidence sequences/intervals on score differentials
        2. (Optional) Cumulative miscoverage rates or false decision rates
        3. (Optional) Widths of confidence sequences/intervals

    Args:
        data: pandas dataframe or path to a saved csv containing
            forecasts and data as columns
            (e.g., output of :py:func:`comparecast.forecasters.forecast`.)
        name_p: column name of the first forecaster
        name_q: column name of the second forecaster
        scoring_rule: name of the scoring rule to be used (default: brier)
        plots_dir: directory to store all plots (default: ./plots)
        alpha: significance level for confidence sequences (default: 0.05)
        lo: minimum value of score differentials (default: -1)
        hi: maximum value of score differentials (default: 1)
        boundary_type: type of uniform boundary used for CS (default: mixture)
        v_opt: value of intrinsic time where the boundary is optimized.
            Default is 10; set to ``None`` in *post-hoc* analyses (only)
            to optimize the boundary at the last time step.
        compare_baselines: compare with other baselines provided, including
            Hoeffding-style CS (``h``), asymptotic CS (``acs``), and
            fixed-time CI (``ci``). (default: ``()``)
        plot_fder: if true, also plot false decision rates (default: False)
        plot_miscoverage: if true, also plot cumulative miscoverage rates
            (default: False)
        n_repeats: number of repeated trials for miscoverage rate calculation
            (default: 10000)
        plot_width: if true, also plot the width of CS/CI
        use_logx: if true, use the logarithmic scale on the x-axis.

    Returns:
        A tuple of two items.
            1. A ``pd.DataFrame`` containing the score differentials (deltas)
               and the confidence sequences (cs)
            2. A ``matplotlib.pyplot.Axes`` object
    """
    assert not (plot_fder and plot_miscoverage), (
        "can only plot either FDeR or miscoverage"
    )

    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)

    T = len(data)
    ps, qs, ys, times = [
        data[name].values for name in [name_p, name_q, "data", "time"]
    ]
    score = get_scoring_rule(scoring_rule)

    # CS/CI calculations
    confseqs = OrderedDict()
    confseqs["cs"] = compare_forecasts(
        data, name_p, name_q, scoring_rule, alpha, lo, hi,
        use_asymptotic=False, boundary_type=boundary_type, v_opt=v_opt)
    if "h" in compare_baselines:
        confseqs["h"] = compare_forecasts(
            data, name_p, name_q, scoring_rule, alpha, lo, hi,
            use_hoeffding=True, boundary_type=boundary_type, v_opt=v_opt)
    if "acs" in compare_baselines:
        confseqs["acs"] = compare_forecasts(
            data, name_p, name_q, scoring_rule, alpha, lo, hi,
            use_asymptotic=True)
    if "ci" in compare_baselines:
        confseqs["ci"] = confint_lai(
            ps, qs, ys, None, scoring_rule=scoring_rule, alpha=alpha)

    # Plot true deltas if true_probs or true_means exist in data
    if "true_probs" in data:
        true_probs = data["true_probs"].values
    elif "true_means" in data:
        true_probs = data["true_means"].values
    else:
        true_probs = None
    if true_probs is not None:
        if scoring_rule == "winkler":
            true_deltas = np.cumsum(
                score(ps, qs, true_probs, base_score="brier")
            ) / times
        else:
            true_deltas = np.cumsum(
                score(ps, true_probs) - score(qs, true_probs)
            ) / times
        logging.info(f"True Delta [T={T}]: {true_deltas[-1]:.5f}")
    else:
        true_deltas = None

    if plot_fder:
        assert true_probs is not None
        assert "ci" in compare_baselines
        fder_cs, fder_ci = compute_fder(
            data, name_p, name_q, n_repeats=n_repeats,
            scoring_rule=scoring_rule, alpha=alpha, lo=lo, hi=hi)
    else:
        fder_cs, fder_ci = None, None
    if plot_miscoverage:
        assert true_probs is not None
        assert "ci" in compare_baselines
        miscov_cs, miscov_ci = compute_miscoverage(
            data, name_p, name_q, n_repeats=n_repeats,
            scoring_rule=scoring_rule, alpha=alpha, lo=lo, hi=hi)
    else:
        miscov_cs, miscov_ci = None, None

    # Plot
    n_figures = 1 + (plot_fder or plot_miscoverage) + plot_width
    figsize = [(8, 5), (12, 5), (16, 5)][n_figures - 1]
    i = 0

    set_style()
    fig, axes = plt.subplots(1, n_figures, figsize=figsize, facecolor="white")
    if n_figures == 1:
        axes = [axes]
    xscale = "log" if use_logx else "linear"
    xlim = (10, T) if use_logx else None
    y_rad = 0.75 * hi if abs(lo) == hi else 0.25 * (hi - lo) / 2
    ylim = (-y_rad, y_rad)

    # Plot 1: confidence sequences & intervals
    axes[i].axhline(color=COLORS["data"], alpha=0.5)
    if true_probs is not None:
        axes[i].plot(times, true_deltas, alpha=0.8, color=COLORS["true"],
                     linestyle=LINESTYLES["true"], label=LABELS["true"])
    for cs_type, (lcbs, ucbs) in confseqs.items():
        axes[i].plot(times, ucbs, alpha=0.8, color=COLORS[cs_type],
                     linestyle=LINESTYLES[cs_type], label=LABELS[cs_type])
        axes[i].plot(times, lcbs, alpha=0.8, color=COLORS[cs_type],
                     linestyle=LINESTYLES[cs_type])
    axes[i].set(
        xscale=xscale,
        xlim=xlim,
        ylim=ylim,
        xlabel="Time",
        ylabel=(("CS/CI" if "ci" in compare_baselines else "CS") +
                r" for $\Delta_t$"),
    )
    axes[i].legend(
        loc="upper right" if confseqs["cs"][1][-1] < 0 else "lower right")
    i += 1

    # Plot 2: either miscoverage or false decision rate
    if plot_miscoverage or plot_fder:
        rate_cs = miscov_cs if plot_miscoverage else fder_cs
        axes[i].plot(times, rate_cs, linestyle=LINESTYLES["cs"],
                     color=COLORS["cs"], label=LABELS["cs"])
        if "ci" in compare_baselines:
            rate_ci = miscov_ci if plot_miscoverage else fder_ci
            axes[i].plot(times, rate_ci, linestyle=LINESTYLES["ci"],
                         color=COLORS["ci"], label=LABELS["ci"])
        axes[i].plot(times, np.repeat(alpha, T), linestyle=LINESTYLES["data"],
                     color=COLORS["data"], label="Significance Level")
        axes[i].set(
            xscale=xscale,
            xlim=xlim,
            ylim=(0, 1),
            xlabel="Time",
            ylabel=("Cumulative Miscoverage Rate" if plot_miscoverage
                    else "False Decision Rate"),
        )
        axes[i].legend()
        i += 1

    # Plot 3: width
    if plot_width:
        for cs_type, (lcbs, ucbs) in confseqs.items():
            axes[i].plot(times, ucbs - lcbs, color=COLORS[cs_type],
                         linestyle=LINESTYLES[cs_type], label=LABELS[cs_type])
        axes[i].set(
            xscale=xscale,
            xlim=xlim,
            ylim=(0, hi - lo),
            xlabel="Time",
            ylabel="Width of CS" + ("/CI" if "ci" in compare_baselines else ""),
        )
        axes[i].legend()
        i += 1

    if scoring_rule == "winkler":
        target_str = r"$W_t$" f"({name_p}, {name_q}), S=brier"
    else:
        target_str = r"$\Delta_t$" f"({name_p}, {name_q}), S={scoring_rule}"

    # str_t = f"$T=10^{np.log10(T):.2g}$" if use_logx else f"T={T}"
    fig.suptitle(f"{(1-alpha)*100:2.0f}% Confidence Sequences on {target_str}",
                 fontsize=13, fontweight="regular")
    fig.tight_layout()
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(
            plots_dir, f"comparecast_cs_{name_p}_{name_q}_{scoring_rule}.pdf"))

    results = {
        "time": times,
        "data": ys,
        "p": ps,
        "q": qs,
        "true_delta": true_deltas,
    }
    for cs_type, (lcbs, ucbs) in confseqs.items():
        results.update({
            f"lcbs_{cs_type}": lcbs,
            f"ucbs_{cs_type}": ucbs,
        })
    if plot_fder:
        results.update({"fder_cs": fder_cs, "fder_ci": fder_ci})
    if plot_miscoverage:
        results.update({"miscov_cs": miscov_cs, "miscov_ci": miscov_ci})
    return pd.DataFrame(results), axes


def plot_pairwise_comparisons(
        data: Union[str, pd.DataFrame],
        forecasters: List[str],
        scoring_rule: str = "brier",
        plots_dir: str = "./plots",
        alpha: float = 0.05,
        lo: float = -1.,
        hi: float = 1.,
        boundary_type: str = "mixture",
        v_opt: float = 10,
        compare_baselines: tuple = (),
        use_logx: bool = True,
) -> plt.Axes:
    """Plot pairwise comparisons of forecasters in data.

    Args:
        data: pandas dataframe or path to a saved csv containing
            forecasts and data as columns
            (e.g., output of :py:func:`comparecast.forecasters.forecast`.)
        forecasters: list of forecasters to be compared against each other
        scoring_rule: name of the scoring rule to be used (default: brier)
        plots_dir: directory to store all plots (default: ./plots)
        alpha: significance level for confidence sequences (default: 0.05)
        lo: minimum value of score differentials (default: -1)
        hi: maximum value of score differentials (default: 1)
        boundary_type: type of uniform boundary used for CS (default: mixture)
        v_opt: value of intrinsic time where the boundary is optimized.
            Default is 10; set to ``None`` in *post-hoc* analyses (only)
            to optimize the boundary at the last time step.
        compare_baselines: compare with other baselines provided, including
            Hoeffding-style CS (``h``), asymptotic CS (``acs``), and
            fixed-time CI (``ci``). (default: ``()``)
        use_logx: if true, use the logarithmic scale on the x-axis.

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

    n = len(forecasters)
    if n > 5:
        logging.warning("too many pairwise comparisons for %d forecasters", n)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 4 * n), facecolor="white")
    T = len(data)
    times = np.arange(1, T + 1)
    ys = data["data"].values
    for i, name_p in enumerate(forecasters):
        for j, name_q in enumerate(forecasters):
            if i == j:
                axes[i][j].set_visible(False)
                continue

            ps = data[name_p].values
            qs = data[name_q].values

            # CS/CI calculations
            confseqs = OrderedDict()
            confseqs["cs"] = compare_forecasts(
                data, name_p, name_q, scoring_rule, alpha, lo, hi,
                boundary_type=boundary_type, v_opt=v_opt)
            if "h" in compare_baselines:
                confseqs["h"] = compare_forecasts(
                    data, name_p, name_q, scoring_rule, alpha, lo, hi,
                    boundary_type=boundary_type, v_opt=v_opt,
                    use_hoeffding=True)
            if "acs" in compare_baselines:
                confseqs["acs"] = compare_forecasts(
                    data, name_p, name_q, scoring_rule, alpha, lo, hi,
                    use_asymptotic=True)
            if "ci" in compare_baselines:
                confseqs["ci"] = confint_lai(
                    ps, qs, ys, None, scoring_rule=scoring_rule, alpha=alpha)

            axes[i][j].axhline(color=COLORS["data"], alpha=0.5)
            if "true_probs" in data.columns:
                true_probs = data["true_probs"].values
                pw_true_deltas = score(ps, true_probs) - score(qs, true_probs)
                true_deltas = np.cumsum(pw_true_deltas) / times
                axes[i][j].plot(times, true_deltas, color=COLORS["true"],
                                linestyle=LINESTYLES["true"],
                                label=LABELS["true"])
            for cs_type, (lcbs, ucbs) in confseqs.items():
                axes[i][j].plot(times, ucbs, alpha=0.8, color=COLORS[cs_type],
                                linestyle=LINESTYLES[cs_type],
                                label=LABELS[cs_type])
                axes[i][j].plot(times, lcbs, alpha=0.8, color=COLORS[cs_type],
                                linestyle=LINESTYLES[cs_type])
            axes[i][j].set_ylim(0.75 * lo, 0.75 * hi)
            if i == len(forecasters) - 1:
                axes[i][j].set_xlabel("Time")
            if j == 0:
                axes[i][j].set_ylabel(r"CS for $\Delta_t$")
            if use_logx:
                axes[i][j].set_xlim(10, T)
                axes[i][j].set_xscale("log")
            str_t = f"$T=10^{{{np.log10(T):.2g}}}$" if use_logx else f"T={T}"
            lcbs, ucbs = confseqs["cs"]
            axes[i][j].set_title(
                r"$\Delta_t$" f"({name_p}, {name_q})"
                f": ({lcbs[-1]:.3f}, {ucbs[-1]:.3f}) at {str_t}"
            )
            axes[i][j].legend(loc=("upper right" if ucbs[-1] + lcbs[-1] < 0
                                   else "lower right"))

    fig.suptitle(f"{(1 - alpha) * 100:2.0f}% "
                 r"Confidence Sequences on $\Delta_t$"
                 f", S={scoring_rule}",
                 fontsize=20, fontweight="regular")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir,
                                 f"comparecast_cs_{n}x{n}_{scoring_rule}.pdf"))

    return axes


def plot_ucbs(
        ucbs: pd.DataFrame,
        style_name: str = PYPLOT_DEFAULT_STYLE,
        palette: List = SEABORN_DEFAULT_PALETTE,
        save_filename: str = "confseq.pdf",
        **plot_kwargs
):
    """Plot the UCBs of multiple confidence sequences across time.

    Useful for quickly comparing the width of many CS.

    Args:
        ucbs: a ``pd.DataFrame`` with UCBs.
        style_name: matplotlib plot style.
        palette: list of colors to to seaborn's lineplot
        save_filename: filename to save the resulting plot.
        **plot_kwargs: any other arguments to ``matplotlib.pyplot.Axes.set()``.

    Returns:
        None
    """
    set_style(style_name)
    if "figsize" in plot_kwargs:
        plt.figure(figsize=plot_kwargs["figsize"])
        del plot_kwargs["figsize"]
    ucbs["Time"] = np.arange(1, len(ucbs) + 1)
    df = ucbs.melt(id_vars=["Time"], value_vars=None,
                   var_name="Method", value_name="UCB")
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(x="Time", y="UCB", hue="Method",
                      palette=palette,
                      data=df)
    ax.set(**plot_kwargs)
    plt.tight_layout()
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename)


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
):
    """Plotting function for MLB forecasts."""
    set_style()
    plt.figure(figsize=(15, 5), facecolor="white")
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
                      hue="forecasts", linewidth=2, alpha=0.7, data=df)
    sns.scatterplot(x="time", y="win", color=COLORS["data"],
                    data=filtered_data, ax=ax)

    span0 = data[data.date == window[0]]["time"].item()
    span1 = data[data.date == window[1]]["time"].item()
    ax.axvspan(span0, span1, alpha=0.3, color='gray')

    team = "Home" if team == "MLB" else team
    if len(years) == 1:
        suffix = (str(years[0]) if n_games is None
                  else f"{years[0]}, last {n_games} games")
        ax.set(
            xticks=[],
            xticklabels=[],
            xlabel="Game",
            ylabel="Probability",
            title=f"{team} Win Probability Forecasts ({suffix})" +
                  (f" (regular seasons only)" if no_playoffs else ""),
            ylim=(-0.05, 1.05),
        )
    else:
        xticks = np.zeros_like(years)
        xticks[1:] = np.where(np.diff(filtered_data.season))[0]
        suffix = f"{years[0]}-{years[-1]}"
        ax.set(
            xticks=xticks,
            xticklabels=years,
            xlabel="Year",
            ylabel="Probability",
            title=f"{team} Win Probability Forecasts ({suffix})" +
                  (f" (regular seasons only)" if no_playoffs else ""),
            ylim=(-0.05, 1.05),
        )
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, 0))
    plt.tight_layout()

    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename)


def plot_weather_hz_evalues(
        evalues: pd.DataFrame,
        save_filename: str = None,
):
    """Reproduces Figure 3 from Henzi & Ziegel (2021).

    A/B: A is not better than B under the null hypothesis
    """
    fg = sns.relplot(
        x="Date",
        y="E-value",
        col="Hypothesis",
        hue="Airport",
        style="Airport",
        kind="line",
        height=5,
        aspect=1,
        data=evalues,
    )
    for ax in fg.axes[0]:
        ax.set(
            xlabel="Year",
            yscale="log",
            ylim=(10 ** -4, 10 ** 4)
        )
        ax.axhline(y=1, linestyle="-", color="black")
        ax.axhline(y=10, linestyle="--", color="gray")
        ax.grid()

    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename)


def plot_weather_comparison(
        data_dir: str = "eprob/replication_material/precip_fcs",
        lag: int = 1,
        scoring_rule: str = "brier",
        alpha: float = 0.1,
        v_opt: float = 0.5,
        c: float = 0.1,
        plots_dir: str = "./plots/weather",
):
    """Produces CS and e-value plots for the weather forecast comparison.

    Also returns a dataframe containing the CS and the e-values.
    """
    assert lag in LAGS, f"invalid lag {lag}, available: {LAGS}"

    df_rows = []
    pairs = [("hclr", "idr"), ("idr", "hclr_noscale"), ("hclr", "hclr_noscale")]
    for airport in AIRPORTS:
        pop_fcs = read_precip_fcs(data_dir, pop_only=True)
        pop_fcs = pop_fcs[(pop_fcs["airport"] == airport)
                          & (pop_fcs["lag"] == lag)]
        for name_p, name_q in pairs:
            try:
                lcbs, ucbs, evalues = compare_forecasts(
                    pop_fcs,
                    name_p,
                    name_q,
                    scoring_rule=scoring_rule,
                    alpha=alpha,
                    compute_evalues=True,
                    v_opt=v_opt,
                    c=c,
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
                    pop_fcs.date, lcbs, ucbs, evalues):
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
    confseqs = pd.DataFrame(df_rows)

    # Plot 1: CS
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
        data=confseqs[confseqs["OutputType"] != "E-value"],
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
            ylim=(-.075, .075)
        )
        ax.axhline(y=0, linestyle="-", color="black")
        ax.grid()

    # remove unnecessary legend items
    for handle, text in zip(fg._legend.legendHandles[5:],
                            fg._legend.texts[5:]):
        handle.set_data([], [])
        text.set_text("")

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"comparecast_cs_lag{lag}.pdf"))

    # Plot 2: e-values
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
        data=confseqs[confseqs["OutputType"] == "E-value"],
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
            ylabel=r"E-values",
            title=(f"E-values against " r"$H_0: \Delta_t$" 
                   f"({name_p}, {name_q})" r"$\leq 0$"),
            yscale="log",
            ylim=(1e-4, 1e4),
        )
        ax.axhline(y=1, linestyle="-", color="black")
        ax.axhline(y=10, linestyle="--", color="gray")
        ax.grid()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(
            os.path.join(plots_dir, f"comparecast_evalues_lag{lag}.pdf"))

    return confseqs
