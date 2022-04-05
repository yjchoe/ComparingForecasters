"""
Diagnostic functions for confidence sequences:
    miscoverage rate, false decision rate
"""

import logging
from typing import Union
import numpy as np
import pandas as pd
from tqdm import tqdm

from comparecast.scoring import ScoringRule, get_scoring_rule
from comparecast.confseq import confseq_eb
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
    if scoring_rule == "winkler":
        score = get_scoring_rule(scoring_rule)
        true_deltas = cumul_mean(score(ps, qs, true_probs, base_score="brier"))
    else:
        score = get_scoring_rule(scoring_rule)
        true_deltas = cumul_mean(
            score.expected_score(ps, true_probs) - score.expected_score(qs, true_probs)
        )
    return true_deltas


def compute_miscoverage(
        data: pd.DataFrame,
        name_p: str,
        name_q: str,
        n_repeats: int = 10000,
        scoring_rule: str = "brier",
        alpha: float = 0.05,
        boundary_type: str = "stitching",
):
    """Compute the cumulative miscoverage rate of
    the forecast score differentials between p and q."""

    if boundary_type != "stitching":
        logging.warning(
            f"miscoverage rate calculation may take too much time "
            f"for boundary type {boundary_type}")

    ps, qs, ys, true_probs = [
        data[name].values
        for name in [name_p, name_q, "data", "true_probs"]
    ]
    T = len(data)

    # Scoring rule & bounds
    score = get_scoring_rule(scoring_rule)
    a, b = score.bounds
    lo, hi = a - b, b - a

    # True deltas
    true_deltas = compute_true_deltas(ps, qs, true_probs, scoring_rule)

    # Generate new ys for n_repeats times; predictions are fixed for now
    # miscoverage = at least one miss up to time t (cf. Ville)
    miscov_cs = np.zeros(T)
    miscov_ci = np.zeros(T)
    for _ in tqdm(range(n_repeats), total=n_repeats,
                  desc="calculating miscoverage rate"):
        # new data from the same true probabilities
        ys = data_utils.synthetic.bernoulli(T, true_probs)

        # CS
        pw_deltas = score(ps, ys) - score(qs, ys)
        lcbs, ucbs = confseq_eb(pw_deltas, alpha, lo, hi,
                                boundary_type=boundary_type)
        miscov_cs += np.cumsum(np.logical_or(true_deltas < lcbs,
                                             ucbs < true_deltas)) >= 1

        # CI (unknown true prob)
        lcbs_ci, ucbs_ci = confint_lai(ps, qs, ys, None,
                                       scoring_rule=scoring_rule, alpha=alpha)
        miscov_ci += np.cumsum(np.logical_or(true_deltas < lcbs_ci,
                                             ucbs_ci < true_deltas)) >= 1
    miscov_cs /= n_repeats
    miscov_ci /= n_repeats
    return miscov_cs, miscov_ci


def compute_fder(
        data: pd.DataFrame,
        name_p: str,
        name_q: str,
        n_repeats: int = 10000,
        scoring_rule: str = "brier",
        alpha: float = 0.05,
        boundary_type: str = "stitching",
):
    """Compute the false decision rate (FDeR) for
    the forecast score differentials between p and q.

    A "false decision" is defined as the event where the confidence interval
    at time t does not include zero but has an opposite sign to the true delta.
    """
    if boundary_type != "stitching":
        logging.warning(
            "false decision rate calculation may take too much time "
            "for boundary type %s", boundary_type)

    ps, qs, ys, true_probs = [
        data[name].values
        for name in [name_p, name_q, "data", "true_probs"]
    ]
    T = len(data)

    # Scoring rule & bounds
    score = get_scoring_rule(scoring_rule)
    a, b = score.bounds
    lo, hi = a - b, b - a

    # True deltas
    true_deltas = compute_true_deltas(ps, qs, true_probs, scoring_rule)

    # Generate new ys for n_repeats times; predictions are fixed for now
    fder_cs = np.zeros(T)
    fder_ci = np.zeros(T)
    for _ in tqdm(range(n_repeats), total=n_repeats,
                  desc="calculating false decision rate"):
        # new data from the same true probabilities
        ys = data_utils.synthetic.bernoulli(T, true_probs)

        # CS
        pw_deltas = score(ps, ys) - score(qs, ys)
        lcbs, ucbs = confseq_eb(pw_deltas, alpha, lo, hi,
                                boundary_type=boundary_type)
        fder_cs += np.logical_or(
            np.logical_and(lcbs > 0, true_deltas < 0),  # pred p > q, true p < q
            np.logical_and(ucbs < 0, true_deltas > 0),  # pred p < q, true p > q
        )

        # CI (unknown true prob)
        lcbs_ci, ucbs_ci = confint_lai(ps, qs, ys, None,
                                       scoring_rule=scoring_rule, alpha=alpha)
        fder_ci += np.logical_or(
            np.logical_and(lcbs_ci > 0, true_deltas < 0),  # pred p > q, true p < q
            np.logical_and(ucbs_ci < 0, true_deltas > 0),  # pred p < q, true p > q
        )

    fder_cs /= n_repeats
    fder_ci /= n_repeats
    return fder_cs, fder_ci
