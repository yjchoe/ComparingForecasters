"""
Compare forecasts over time using confidence sequences
"""

import logging
from typing import Union, Tuple
import numpy as np
import pandas as pd

from comparecast.scoring import get_scoring_rule
from comparecast.confseq import confseq_h, confseq_eb, confseq_asymptotic
from comparecast.evalues import evalue_expm


def compare_forecasts(
        data: Union[str, pd.DataFrame],
        name_p: str,
        name_q: str,
        scoring_rule: str = "brier",
        alpha: float = 0.05,
        lo: float = -1.,
        hi: float = 1.,
        use_hoeffding: bool = False,
        use_asymptotic: bool = False,
        compute_evalues: bool = False,
        **kwargs,
) -> Tuple[np.ndarray, ...]:
    """Compare a pair of forecasts over time using
    time-uniform confidence sequences.

    Args:
        data: pandas dataframe or path to a saved csv containing
            forecasts and data as columns
            (e.g., output of :py:func:`~comparecast.forecasters.forecast`.)
        name_p: column name of the first forecaster
        name_q: column name of the second forecaster
        scoring_rule: name of the scoring rule to be used (default: brier)
        alpha: significance level for confidence sequences (default: 0.05)
        lo: minimum value of score differentials (default: -1)
        hi: maximum value of score differentials (default: 1)
        use_hoeffding: if True, use the Hoeffding-style CS instead.
            (default: False)
        use_asymptotic: if True, use the asymptotic CS instead.
            (default: False)
        compute_evalues: if True, also compute e-values against
        the one-sided null corresponding to the CS (default: False)
        kwargs: hyperparameters to :py:func:`~comparecast.confseq.confseq_h`,
            :py:func:`~comparecast.confseq.confseq_eb`,
            or :py:func:`~comparecast.confseq.confseq_asymptotic`.
    Returns:
        Either (lcbs, ucbs) or (lcbs, ucbs, evalues).
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)

    T = len(data)
    ps, qs, ys = [
        data[name].values for name in [name_p, name_q, "data"]
    ]
    score = get_scoring_rule(scoring_rule)

    if scoring_rule == "winkler":
        logging.info(
            "computing winkler's score while treating %s as a baseline",
            name_q
        )
        # TODO: consider non-brier scores for winkler (need bounds)
        pw_deltas = score(ps, qs, ys, base_score="brier")
        q0 = min(min(qs), min(1 - qs))
        assert 0 < q0 < 1, \
            "baseline forecaster for winkler's score must be in range (0, 1)"
        lo, hi = 1 - 2 / q0, 1
    else:
        # pointwise deltas: delta(p_t, q_t) = S(p_t, y_t) - S(q_t, y_t)
        pw_deltas = score(ps, ys) - score(qs, ys)

    if use_hoeffding:
        # Theorem 3.1
        lcbs, ucbs = confseq_h(pw_deltas, alpha, lo, hi, **kwargs)
        logging.info(f"{(1 - alpha) * 100:2.0f}% Hoeffding-style CS [T={T}]:"
                     f" ({lcbs[-1]:.5f}, {ucbs[-1]:.5f})")
    elif not use_asymptotic:
        # Theorem 3.2
        lcbs, ucbs = confseq_eb(pw_deltas, alpha, lo, hi, **kwargs)
        logging.info(f"{(1 - alpha) * 100:2.0f}% CS [T={T}]:"
                     f" ({lcbs[-1]:.5f}, {ucbs[-1]:.5f})")
    else:
        # Waudby-Smith et al. (2021)
        lcbs, ucbs = confseq_asymptotic(pw_deltas, alpha, lo, hi, **kwargs)
        logging.info(f"{(1 - alpha) * 100:2.0f}% Asymptotic CS [T={T}]:"
                     f" ({lcbs[-1]:.5f}, {ucbs[-1]:.5f})")

    if compute_evalues:
        assert not use_asymptotic, \
            "e-values are only calculated in the non-asymptotic sense"
        kwargs.setdefault("c", hi - lo)
        kwargs.setdefault("alpha_opt", alpha / 2)  # one-sided
        evalues = evalue_expm(pw_deltas, **kwargs)
        return lcbs, ucbs, evalues

    return lcbs, ucbs
