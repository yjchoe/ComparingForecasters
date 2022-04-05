"""
Utility functions

References:
    https://github.com/WannabeSmith/drconfseq
"""

from typing import Union, Tuple
from numpy.typing import ArrayLike
import numpy as np


def cumul_mean(
        xs: ArrayLike,
        n_init: int = 0,
        init_mean: Union[int, float] = 0,
) -> np.ndarray:
    """Compute the cumulate mean of `xs` at each time,
    with `n_init` initial points having mean `init_mean`.

    Set `n_init` to `0` to obtain the mean equivalent of :py:func:`np.cumsum`.
    """
    xs = np.array(xs)
    sums = np.cumsum(xs)
    times = np.arange(1, len(xs) + 1)
    return (sums + n_init * init_mean) / (times + n_init)


def cumul_var(
        xs: ArrayLike,
) -> np.ndarray:
    """Compute the cumulative sample variance of `xs` at each time."""
    xs = np.array(xs)
    times = np.arange(1, len(xs) + 1)
    scalar = times / np.insert(times, 0, 1)[:len(times)]
    return scalar * (cumul_mean(xs ** 2) - cumul_mean(xs) ** 2)


def check_bounds(
        xs: np.ndarray,
        lo: float = 0.,
        hi: float = 1.,
) -> int:
    """Check if input array is 1-dimensional and
    has values within the provided bounds.

    Returns 0 if bounds are met; otherwise throws an exception.
    """
    assert lo < hi, f"lower bound {lo} must be smaller than upper bound {hi}"
    assert len(xs.shape) == 1, \
        f"input array must be 1-dimensional, got shape {xs.shape}"
    assert np.logical_and(lo <= xs, xs <= hi).all(), \
        f"input array contains values outside ({lo}, {hi})"
    return 0


def preprocess_score_inputs(
        ps: ArrayLike,
        ys: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a forecast-outcome pair into two `(T, C)` arrays,
    representing C-categorical probabilities/outcomes."""
    assert len(ps) == len(ys), \
        f"input array lengths do not match: {len(ps)} != {len(ys)}"
    ps, ys = np.array(ps), np.array(ys)
    assert len(ps.shape) <= 2, \
        f"input forecasts must have <=2 dimensions, got {len(ps.shape)}"

    def _handle_probs(probs, n_classes_if_1d):
        """handle binary/categorical probabilities."""
        if len(probs.shape) == 1:
            probs = convert_to_onehot(probs, n_classes_if_1d)

        assert np.logical_and(0 <= probs, probs <= 1).all(), \
            "probabilities and one-hot vectors must take values within [0, 1]"
        assert np.allclose(probs.sum(axis=1), np.ones(probs.shape[0])), \
            "2-dim probabilities must sum to 1 along axis=1"
        return probs

    # if ps is 1-dim, then it must be an array of binary probabilities
    ps = _handle_probs(ps, 2)
    n_classes = ps.shape[1]

    # handle outcomes or outcome probabilities (one-hot matching ps)
    assert len(ys.shape) == 1 or ys.shape == ps.shape, \
        f"outcomes shape do not match expected shape, got {ys.shape}"
    ys = _handle_probs(ys, n_classes)

    return ps, ys


def check_lengths(
        *arrays: ArrayLike
) -> int:
    """Check whether two input array-like objects have the same length."""
    if len(arrays) < 1:
        return 0

    n0 = len(arrays[0])
    assert all(len(arr) == n0 for arr in arrays), (
        "input array lengths do not match: " 
        ", ".join([str(len(arr)) for arr in arrays])
    )
    return 0


def convert_to_onehot(
        xs: ArrayLike,
        n_classes: int = 0,
) -> np.ndarray:
    """Convert an integer-valued ordinal array to one-hot representation.

    Also supports converting binary probabilities to categorical ones,
    via the mapping p -> [1-p, p].

    If `n_classes == 0`, then
        it is inferred from the maximum integer value in `xs`.
    """
    xs = np.array(xs)
    assert len(xs.shape) == 1, "can only convert 1-dim arrays to one-hot"
    size = len(xs)
    n_classes = n_classes if n_classes >= 1 else max(xs) + 1
    if n_classes == 2:
        new_xs = np.vstack((1 - xs, xs)).T
    else:
        new_xs = np.zeros((size, n_classes))
        new_xs[np.arange(size), xs] = 1
    return new_xs


def convert_to_ordinal(
        xs: ArrayLike,
) -> np.ndarray:
    """Convert a one-hot array into an ordinal array.

    The last axis is assumed to be the one-hot dimension.
    """
    return np.argmax(xs, axis=-1)


def scale(xs: np.ndarray, lo: float = 0., hi: float = 1.):
    """Scale xs in [lo, hi] to zs in [0, 1] via

    z_t = (x_t - lo) / (hi - lo)."""
    return (xs - lo) / (hi - lo)


def unscale(zs: np.ndarray, lo: float = 0., hi: float = 1.):
    """Unscale zs in [0, 1] to xs in [lo, hi]."""
    return lo + (hi - lo) * zs
