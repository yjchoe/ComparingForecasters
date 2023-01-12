"""
Cumulant generating functions (CGFs) used for exponential supermartingales
"""

from typing import Union
import numpy as np
from numpy.typing import ArrayLike


def get_cgf(name: str):
    try:
        return CGFS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown CGF {name}")


def _check_lambda_bound(
        lambdas: ArrayLike,
        lo: float = 0,
        hi: float = np.inf,
):
    """Check lo <= lambdas < hi."""
    lambdas, lo, hi = [np.atleast_1d(inp) for inp in [lambdas, lo, hi]]
    assert (lo <= lambdas).all() and (lambdas <= hi).all(), (
        f"lambdas out of bounds: {min(lambdas)}, {max(lambdas)}"
    )


def cgf_bernoulli(lambdas: ArrayLike, g: float, h: float, **kwargs):
    """Scaled CGF of a centered random variable supported on -g and h."""
    assert g > 0 and h > 0
    _check_lambda_bound(lambdas, 0, np.inf)
    return np.log(
        (g * np.exp(h * lambdas) + h * np.exp(-g * lambdas)) / (g + h)
    ) / (g * h)


def cgf_gaussian(lambdas: ArrayLike, **kwargs):
    """CGF of a standard Gaussian random variable."""
    _check_lambda_bound(lambdas, 0, np.inf)
    return 0.5 * lambdas ** 2


def cgf_poisson(lambdas: ArrayLike, c: float = 1, **kwargs):
    """CGF of a centered Poisson random variable with scale parameter c."""
    _check_lambda_bound(lambdas, 0, np.inf)
    return (np.exp(c * lambdas) - c * lambdas - 1) / (c ** 2)


def cgf_exponential(lambdas: Union[float, ArrayLike],
                    c: Union[float, ArrayLike] = 0.5,
                    **kwargs):
    """CGF of a centered exponential random variable with scale parameter c."""
    _check_lambda_bound(lambdas, 0, 1 / np.where(c > 0, c, np.inf))
    return (-np.log(1 - c * lambdas) - c * lambdas) / (c ** 2)


def cgf_gamma(lambdas: ArrayLike, c: float = 1, **kwargs):
    """CGF of a sub-gamma random variable with scale parameter c."""
    _check_lambda_bound(lambdas, 0, 1 / c if c > 0 else np.inf)
    return lambdas ** 2 / (2 * (1 - c * lambdas))


CGFS = {
    "bernoulli": cgf_bernoulli,
    "exponential": cgf_exponential,
    "gamma": cgf_gamma,
    "gaussian": cgf_gaussian,
    "normal": cgf_gaussian,
    "poisson": cgf_poisson,
}
