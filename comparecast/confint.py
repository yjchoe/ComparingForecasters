"""
Non-Time-Uniform Confidence Intervals for Score Differentials

We implement (Lai et al., 2011) as a baseline. Importantly,
these confidence intervals are *not* valid at an arbitrary stopping time.
"""

import logging
from typing import Tuple, Union
import numpy as np
import scipy.stats

from comparecast.scoring import ScoringRule, get_scoring_rule


def confint_lai(
        ps: np.ndarray,
        qs: np.ndarray = None,
        ys: np.ndarray = None,
        true_probs: np.ndarray = None,
        scoring_rule: Union[ScoringRule, str] = "brier",
        alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-time-uniform & asymptotic confidence intervals for
    forecast scores & score differentials.

    See Section 3.2 of (Lai et al., 2011).

    NOTE: true_probs is optionally used to better estimate the width;
          it is only available for synthetic data.
    """
    assert ys is not None, "data is not provided (third argument)"
    if qs is None:
        logging.info("computing fixed-time asymptotic CI for S(p, y)")
        raise NotImplementedError
    else:
        logging.info("computing fixed-time asymptotic CI for S(p, y) - S(q, y)")

    assert ps.shape == qs.shape == ys.shape
    assert 0 <= alpha <= 1

    T = len(ps)
    times = np.arange(1, T + 1)
    if scoring_rule == "winkler":
        score = get_scoring_rule("brier")
        skill_score = get_scoring_rule("winkler")
        fsds = np.cumsum(skill_score(ps, qs, ys, base_score=score)) / times
    else:
        score = get_scoring_rule(scoring_rule)
        fsds = np.cumsum(score(ps, ys) - score(qs, ys)) / times

    # variance: use true_probs if known
    all_zeros, all_ones = np.repeat(0, T), np.repeat(1, T)
    dsq = ((score(ps, all_ones) - score(ps, all_zeros)) -
           (score(qs, all_ones) - score(qs, all_zeros))) ** 2
    if scoring_rule == "winkler":
        # qs is assumed to be the baseline forecaster
        lsq = np.where(ps >= qs,
                       score(ps, all_ones) - score(qs, all_ones),
                       score(ps, all_zeros) - score(qs, all_zeros)) ** 2
        lsq = np.where(lsq != 0, lsq, 1e-8)
    else:
        lsq = 1.0
    if true_probs is not None:
        assert true_probs.shape == ys.shape
        ssq = np.cumsum(dsq * true_probs * (1 - true_probs) / lsq) / times
    else:
        ssq = np.cumsum(dsq * 0.25 / lsq) / times  # conservative estimate

    lcbs, ucbs = scipy.stats.norm.interval(1 - alpha,
                                           fsds,
                                           np.sqrt(ssq / times) + 1e-8)
    return lcbs, ucbs
