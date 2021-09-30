"""
Proper scoring rules for binary forecasts.

In order to obtain valid confidence sequences for score differentials,
we require that the scoring rules have linear equivalents (Lai et al., 2011).

Following Gneiting & Raftery (2007), we follow the convention that
higher scores mean better forecasts.
"""

import numpy as np


def brier_score(ps: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Computes the pointwise Brier score S(p_t, y_t) = -(p_t - y_t)^2."""
    return -np.subtract(ps, ys) ** 2


def brier_score_lineq(ps: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Computes the linear equivalent (Lai et al., 2011) of
    the pointwise Brier score S(p_t, y_t) = 2*p_t*y_t - p_t^2."""
    return 2 * np.multiply(ps, ys) - np.square(ps)


def spherical_score(ps: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """S(p, y) = (p^Ty) / ||p||_2."""
    return np.multiply(ps, ys) / np.sqrt(np.square(ps) + (np.square(1 - ps)))


def zero_one_score(ps: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """S(p, y) = argmax(p)^Ty."""
    return np.multiply(ps >= 0.5, ys)


def logarithmic_score(
        ps: np.ndarray,
        ys: np.ndarray,
        clip: float = 0.01,
) -> np.ndarray:
    """S(p, y) = log(p^Ty).

    This scoring rule is unbounded for p, y in [0, 1].
    The clipping parameter `clip` (default: 0.01) clips p to the range
    `(clip, 1-clip)`.
    """
    if clip > 0.0:
        ps = np.clip(ps, clip, 1 - clip)
    return ys * np.log(ps + 1e-8) + (1 - ys) * np.log(1 - ps + 1e-8)


def absolute_score(ps: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """S(p, y) = -||p - y||_1. *Does not have a linear equivalent.*"""
    return -np.abs(np.subtract(ps, ys))


def winkler_score(
        ps: np.ndarray,
        qs: np.ndarray,
        ys: np.ndarray,
        base_score: str = "brier",
) -> np.ndarray:
    """Winkler (1994)'s normalized score (pointwise), which computes
    a skill score against a baseline forecaster (q)."""
    assert base_score != "winkler", \
        "can't use winkler score as a base score for winkler score!"
    assert np.logical_and(0 < qs, qs < 1).all()
    s = get_scoring_rule(base_score)
    numer = s(ps, ys) - s(qs, ys)
    denom = np.where(ps >= qs, s(ps, 1) - s(qs, 1), s(ps, 0) - s(qs, 0))
    denom = np.where(denom != 0, denom, 1e-8)
    return numer / denom


"""
Accessing scoring rules
"""

SCORING_RULES = {
    "brier": brier_score,
    "brier_lineq": brier_score_lineq,
    "quadratic": brier_score,
    "quadratic_lineq": brier_score_lineq,
    "spherical": spherical_score,
    "zero_one": zero_one_score,
    "logarithmic": logarithmic_score,
    "absolute": absolute_score,
    "winkler": winkler_score,
}


def get_scoring_rule(name: str):
    """Return a scoring rule as a function given its name."""
    try:
        return SCORING_RULES[name]
    except KeyError:
        raise KeyError(f"invalid scoring rule {name}, try: " +
                       ", ".join(SCORING_RULES.keys())) from None
