"""
Compare forecasts over time using confidence sequences
"""

import logging
from typing import Union
import numpy as np
import pandas as pd

from comparecast.scoring import ScoringRule, get_scoring_rule, WinklerScore
from comparecast.confseq import confseq_h, confseq_eb, confseq_asymptotic
from comparecast.eprocess import eprocess_expm


def compare_forecasts(
        data: Union[str, pd.DataFrame],
        name_p: str,
        name_q: str,
        scoring_rule: Union[ScoringRule, str] = "brier",
        lag: int = 1,
        aligned_outcomes: bool = True,
        compute_cs: bool = True,
        alpha: float = 0.05,
        use_hoeffding: bool = False,
        use_asymptotic: bool = False,
        lcb_only: bool = False,
        ucb_only: bool = False,
        compute_e: bool = True,
        **kwargs,
) -> pd.DataFrame:
    """Compare a pair of forecasts over time using
    time-uniform confidence sequences or e-processes.

    Args:
        data: pandas dataframe or path to a saved csv containing
            forecasts and data as columns
            (e.g., output of :py:func:`~comparecast.forecasters.forecast`.)
        name_p: column name of the first forecaster
        name_q: column name of the second forecaster
        scoring_rule: a :py:obj:`~comparecast.scoring.ScoringRule` object or its name
            (default: brier)
        lag: forecast lag. Currently requires compute_cs is False if lag > 1. (default: 1)
        aligned_outcomes: whether the outcomes are aligned with the forecasts, if lag > 1.
            (default: True)
        compute_cs: whether to compute a CS. (default: True)
        alpha: significance level for confidence sequences (default: 0.05)
        use_hoeffding: if True, use the Hoeffding-style CS instead.
            (default: False)
        use_asymptotic: if True, use the asymptotic CS instead.
            (default: False)
        lcb_only: Compute a one-sided CS with the lower confidence bound only.
            MUST provide the scale parameter `c`, as `hi` is unused.
        ucb_only: Compute a one-sided CS with the upper confidence bound only.
            MUST provide the scale parameter `c`, as `lo` is unused.
        compute_e: whether to compute an e-process against
        the both one-sided nulls corresponding to the CS (default: True)
        kwargs: hyperparameters to :py:func:`~comparecast.confseq.confseq_h`,
            :py:func:`~comparecast.confseq.confseq_eb`,
            or :py:func:`~comparecast.confseq.confseq_asymptotic`.
    Returns:
        pd.DataFrame containing columns (time, lcb, ucb, e_pq, e_qp).
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)

    ps, qs, ys = [
        data[name].values for name in [name_p, name_q, "y"]
    ]

    # Align forecasts and outcomes in case there are lags. e.g., lag = 3, T = 5
    T = len(data)
    times = np.arange(1, T + 1)
    if lag > 1:
        if compute_cs or not compute_e:
            raise NotImplementedError(
                "currently only supports e-processes for comparing multi-lag forecasts"
            )
        if not aligned_outcomes:
            # [0, 1, 2]
            ps, qs = [arr[:T - lag + 1] for arr in [ps, qs]]
            # [2, 3, 4]
            ys = ys[lag - 1:]
            # result times are outcome-aligned
            times = times[lag - 1:]
    result = pd.DataFrame({"time": times})

    # Compute pointwise score differentials
    score = get_scoring_rule(scoring_rule)
    if isinstance(score, WinklerScore):
        pw_deltas = score(ps, qs, ys)
        lo, hi = score.bounds
    else:
        pw_deltas = score(ps, ys) - score(qs, ys)
        a, b = score.bounds
        lo, hi = a - b, b - a  # lower/upper bounds on delta(p_t, q_t)
        assert np.isfinite(lo) and np.isfinite(hi), \
            "lower and upper bounds must be finite except when computing one-sided CS"

    # Check appropriate conditions
    if lcb_only or ucb_only:
        if use_hoeffding or use_asymptotic:
            raise NotImplementedError(
                "currently only supports lcb_only and ucb_only for EB CS"
            )
    else:
        assert np.isfinite(lo) and np.isfinite(hi), \
            "lower and upper bounds must be finite except when computing one-sided CS"

    # CS
    if compute_cs:
        if use_hoeffding:
            # Theorem 1, Choe and Ramdas (2021)
            lcbs, ucbs = confseq_h(pw_deltas, alpha, lo, hi, **kwargs)
            logging.info(f"{(1 - alpha) * 100:2.0f}% Hoeffding-style CS [T={T}]:"
                         f" ({lcbs[-1]:.5f}, {ucbs[-1]:.5f})")
        elif not use_asymptotic:
            # Theorem 2, Choe and Ramdas (2021)
            lcbs, ucbs = confseq_eb(pw_deltas, alpha, lo, hi,
                                    lcb_only=lcb_only,
                                    ucb_only=ucb_only,
                                    **kwargs)
            logging.info(f"{(1 - alpha) * 100:2.0f}% CS [T={T}]:"
                         f" ({lcbs[-1]:.5f}, {ucbs[-1]:.5f})")
        else:
            # Theorem 2.3, Waudby-Smith et al. (2021)
            lcbs, ucbs = confseq_asymptotic(pw_deltas, alpha, **kwargs)
            logging.info(f"{(1 - alpha) * 100:2.0f}% Asymptotic CS [T={T}]:"
                         f" ({lcbs[-1]:.5f}, {ucbs[-1]:.5f})")
        result["lcb"] = lcbs
        result["ucb"] = ucbs
    else:
        result["lcb"] = None
        result["ucb"] = None

    # E-process
    if compute_e:
        if use_asymptotic or use_hoeffding:
            raise NotImplementedError

        # corresponds to computing e_pq only
        if lcb_only:
            result["e_pq"] = eprocess_expm(pw_deltas, lag=lag, **kwargs)
            result["e_qp"] = None
        # corresponds to computing e_qp only
        elif ucb_only:
            kwargs.setdefault("alpha_opt", alpha)
            result["e_pq"] = None
            result["e_qp"] = eprocess_expm(-pw_deltas, lag=lag, **kwargs)
        else:
            kwargs.setdefault("c", hi - lo)
            kwargs.setdefault("alpha_opt", alpha / 2)  # each one-sided
            result["e_pq"] = eprocess_expm(pw_deltas, lag=lag, **kwargs)
            result["e_qp"] = eprocess_expm(-pw_deltas, lag=lag, **kwargs)
    else:
        result["e_pq"] = None
        result["e_qp"] = None

    return result
