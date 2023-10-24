"""
Diagnostic functions for confidence sequences:
    miscoverage rate, false decision rate
"""

import logging
from typing import Union, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm

from comparecast.scoring import ScoringRule, get_scoring_rule, RelativeScore
from comparecast.confseq import confseq_eb, confseq_asymptotic
from comparecast.confint import confint_lai
from comparecast.utils import cumul_mean
from comparecast import data_utils


def compute_true_deltas(
        ps: np.ndarray,
        qs: np.ndarray,
        true_probs: np.ndarray,
        scoring_rule: Union[ScoringRule, str],
) -> np.ndarray:
    """Compute true deltas, Delta_t.

    This only works when the true probabilities (r_t) are known.
    """
    score = get_scoring_rule(scoring_rule)
    if isinstance(score, RelativeScore):
        true_deltas = cumul_mean(
            score.expected_score(ps, qs, true_probs)
        )
    else:
        true_deltas = cumul_mean(
            score.expected_score(ps, true_probs) - score.expected_score(qs, true_probs)
        )
    return true_deltas


def compute_miscoverage(lcbs, ucbs, true_params):
    """Compute the cumulative miscoverage rate of a sequence of confidence intervals.

    A "cumulative miscoverage" is defined as the event where
    the true parameter is not covered in at least one of the intervals up to time t.
    """
    return np.cumsum(
        np.logical_or(true_params < lcbs, ucbs < true_params)
    ) >= 1


def compute_fder(lcbs, ucbs, true_params):
    """Compute the false decision rate (FDeR) of a sequence of confidence intervals.

    A "false decision" is defined as the event where the confidence interval
    at time t does not include zero but has an opposite sign to the true delta.
    """
    return np.logical_or(
        np.logical_and(lcbs > 0, true_params < 0),  # pred p > q, true p < q
        np.logical_and(ucbs < 0, true_params > 0),  # pred p < q, true p > q
    )


def compute_cfdr(pvals_pq, pvals_qp=None, alpha=0.05):
    """Compute the cumulative false discovery rate (cFDR) of a sequence of p-values.

    Corresponds to the cumulative miscoverage rate for (1-alpha)-level CI/CSs.

    `pvals_qp` is optional; if not None, a two-sided cFDR is computed.
    """
    if pvals_qp is not None:
        return np.cumsum(
            np.logical_or(pvals_pq < alpha / 2, pvals_qp < alpha / 2)
        ) >= 1
    else:
        return np.cumsum(pvals_pq < alpha) >= 1


def compute_diagnostics(
        data: pd.DataFrame,
        name_p: str,
        name_q: str,
        diagnostics_fn: Union[str, Callable[[np.ndarray,
                                             np.ndarray,
                                             np.ndarray], float]] = "miscoverage",
        n_repeats: int = 100,
        interval: int = 100,
        scoring_rule: str = "brier",
        alpha: float = 0.05,
        boundary_type: str = "stitching",
        baselines: tuple = ("ci", ),  # "dm", "gw" require epftoolbox
):
    """Compute a diagnostics function for CI/CSs on
    forecast score differentials.

    diagnostic_fn is any function that takes (lcbs, ucbs, true_deltas)
    and computes a diagnostic metric up to time t (length of the inputs).
    """

    baselines = tuple(s.lower() for s in baselines)
    if "dm" in baselines or "gw" in baselines:
        assert diagnostics_fn == "cfdr"
        try:
            from epftoolbox.evaluation import DM
            from epftoolbox.evaluation.gw import gwtest
        except ImportError:
            raise ImportError("baselines 'dm' and 'gw' require package 'epftoolbox'")

    diagnostics_fn_pval = None
    if diagnostics_fn == "miscoverage":
        diagnostics_fn = compute_miscoverage
    elif diagnostics_fn == "fder":
        diagnostics_fn = compute_fder
    elif diagnostics_fn == "cfdr":
        diagnostics_fn = compute_miscoverage  # for CS
        diagnostics_fn_pval = compute_cfdr    # for p-values
    else:
        try:
            1 + diagnostics_fn(np.array([-1, -1]), np.array([1, -1]), np.array([0, 0]))
        except:
            raise ValueError("custom 'diagnostic_fn' must take "
                             "three np.ndarray inputs and return a number")

    if boundary_type != "stitching":
        logging.warning(
            f"miscoverage rate calculation may be slow"
            f" for boundary type {boundary_type}")

    ps, qs, ys, true_probs = [
        data[name].values
        for name in [name_p, name_q, "y", "true_probs"]
    ]
    T = len(data)

    # Scoring rule & bounds
    score = get_scoring_rule(scoring_rule)
    a, b = score.bounds
    lo, hi = a - b, b - a

    # True deltas
    true_deltas = compute_true_deltas(ps, qs, true_probs, scoring_rule)

    # Generate new ys for n_repeats times
    diagnostics = {method: np.zeros(T) for method in ("cs", ) + baselines}

    # Only compute every {interval} times for DM/GW
    intervals = np.arange(interval, T + 1, interval)
    unused = np.array([t for t in range(T) if t + 1 not in intervals])
    if "dm" in diagnostics:
        diagnostics["dm"][unused] = np.nan
    if "gw" in diagnostics:
        diagnostics["gw"][unused] = np.nan

    for _ in tqdm(range(n_repeats), total=n_repeats,
                  desc="calculating diagnostics under repeated sampling"):
        # new data from the same true probabilities
        ys = data_utils.synthetic.bernoulli(T, true_probs)

        # CS
        scores_p = score(ps, ys)
        scores_q = score(qs, ys)
        pw_deltas = scores_p - scores_q
        lcbs, ucbs = confseq_eb(pw_deltas, alpha, lo, hi,
                                boundary_type=boundary_type)
        diagnostics["cs"] += diagnostics_fn(lcbs, ucbs, true_deltas)

        # AsympCS
        if "acs" in diagnostics:
            lcbs_acs, ucbs_acs = confseq_asymptotic(pw_deltas, alpha, assume_iid=False)
            diagnostics["acs"] += diagnostics_fn(lcbs_acs, ucbs_acs, true_deltas)

        # CI
        if "ci" in diagnostics:
            lcbs_ci, ucbs_ci = confint_lai(ps, qs, ys, None,
                                           scoring_rule=scoring_rule, alpha=alpha)
            diagnostics["ci"] += diagnostics_fn(lcbs_ci, ucbs_ci, true_deltas)

        # hack to support 2 dimensions for DM/GW
        ys_2d, ps_2d, qs_2d = [np.vstack([arr, 1 - arr]).T for arr in [ys, ps, qs]]

        # DM test of equal unconditional predictive ability
        if "dm" in diagnostics:
            pvals_pq = np.array([
                DM(ys_2d[:t], qs_2d[:t], ps_2d[:t], norm=2)[0]
                for t in intervals
            ])
            pvals_qp = np.array([
                DM(ys_2d[:t], qs_2d[:t], ps_2d[:t], norm=2)[0]
                for t in intervals
            ])
            diagnostics["dm"][intervals - 1] += diagnostics_fn_pval(pvals_pq, pvals_qp, alpha=alpha)

        # GW test of equal conditional predictive ability
        if "gw" in diagnostics:
            loss_p, loss_q = 1 - scores_p, 1 - scores_q
            pvals_pq = np.array([
                gwtest(loss_q[:t], loss_p[:t], conditional=True)
                for t in intervals
            ])
            pvals_qp = np.array([
                gwtest(loss_p[:t], loss_q[:t], conditional=True)
                for t in intervals
            ])
            diagnostics["gw"][intervals - 1] += diagnostics_fn_pval(pvals_pq, pvals_qp, alpha=alpha)

    for method, rate in diagnostics.items():
        diagnostics[method] /= n_repeats

    return diagnostics

