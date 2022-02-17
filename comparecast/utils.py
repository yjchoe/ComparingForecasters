"""
Utility functions

References:
    https://github.com/WannabeSmith/drconfseq
"""

from typing import Union
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



def scale(xs: np.ndarray, lo: float = 0., hi: float = 1.):
    """Scale xs in [lo, hi] to zs in [0, 1] via

    z_t = (x_t - lo) / (hi - lo)."""
    return (xs - lo) / (hi - lo)


def unscale(zs: np.ndarray, lo: float = 0., hi: float = 1.):
    """Unscale zs in [0, 1] to xs in [lo, hi]."""
    return lo + (hi - lo) * zs
