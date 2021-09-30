"""
Confidence sequences for probability forecasts
"""

from typing import Tuple
import numpy as np

from confseq import boundaries


"""
Time-Uniform, Nonparametric, Nonasymptotic Confidence Sequences 
(Howard et al., 2021)

These bounds are centered around the sample mean of observations.
"""


def _check_xs(
        xs: np.ndarray,
        lo: float = 0.,
        hi: float = 1.,
):
    """Check if input array is 1-dimensional and
    has values within the provided bounds."""
    assert lo < hi, f"lower bound {lo} must be smaller than upper bound {hi}"
    assert len(xs.shape) == 1, \
        f"input array must be 1-dimensional, got shape {xs.shape}"
    assert np.logical_and(lo <= xs, xs <= hi).all(), \
        f"input array contains values outside ({lo}, {hi})"
    return 0


def confseq_h(
        xs: np.ndarray,
        alpha: float = 0.05,
        lo: float = 0.,
        hi: float = 1.,
        boundary_type: str = "mixture",
        v_opt: float = 10.,
        s: float = 1.4,
        eta: float = 2.,
        c: float = None,
        is_one_sided: bool = False,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the Hoeffding-style confidence sequences
     for the time-varying mean of a sequence of bounded random variables.

    Corresponds to Theorem 3.1 in our paper.

    Args:
        xs: Input sequence of bounded random variables.
        alpha: Significance level.
        lo: Lower bound on the input random variables.
        hi: Upper bound on the input random variables.
        boundary_type: Type of sub-Gaussian uniform boundary.
            Either ``stitching`` or ``mixture`` (default).
            Note that ``stitching`` may yield wider intervals than ``mixture``.
        v_opt: value of intrinsic time where the boundary is optimized.
            Default is 10; set to ``None`` in *post-hoc* analyses (only)
            to optimize the boundary at the last time step.
        s: (``stitching`` only) controls how crossing probability is
            distributed over epochs
        eta: (``stitching`` only) controls the spacing of epochs
        c: (``stitching`` only) scale parameter. Default (set to None): hi - lo.
        is_one_sided: (``mixture`` only) whether to use the one-sided
        normal mixture boundary. Default: False.

    Returns:
        A tuple with the lower and upper confidence bounds.
    """
    _check_xs(xs, lo, hi)

    # Time & sample mean (centers)
    ts = np.arange(1, len(xs) + 1)
    mus = np.cumsum(xs) / ts

    # Intrinsic time as t * sigma^2, where sigma is the sub-Gaussian parameter
    sigma = (hi - lo) / 2
    vs = sigma * ts

    # Sub-Gaussian uniform boundary
    v_opt = v_opt if v_opt is not None else vs[-1]
    if boundary_type.lower() == "stitching":
        c = c if c is not None else hi - lo
        radii = boundaries.poly_stitching_bound(
            vs, alpha / 2, v_opt, c=c, s=s, eta=eta,
        ) / ts
    elif boundary_type.lower() == "mixture":
        if is_one_sided:
            # One-sided variant
            radii = boundaries.normal_mixture_bound(
                vs, alpha / 2, v_opt, alpha_opt=alpha / 2, is_one_sided=True,
            ) / ts
        else:
            # Default (two-sided)
            radii = boundaries.normal_mixture_bound(
                vs, alpha, v_opt, alpha_opt=alpha, is_one_sided=False,
            ) / ts
    else:
        raise ValueError(
            f"boundary_type must be either 'stitching' or 'mixture'"
            f" (given: {boundary_type})"
        )
    return mus - radii, mus + radii


def confseq_eb(
        xs: np.ndarray,
        alpha: float = 0.05,
        lo: float = 0.,
        hi: float = 1.,
        boundary_type: str = "mixture",
        v_opt: float = 10.,
        s: float = 1.4,
        eta: float = 2.,
        c: float = None,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the empirical Bernstein confidence sequences
     for the time-varying mean of a sequence of bounded random variables.

    Corresponds to Theorem 3.2 in our paper.

    Args:
        xs: Input sequence of bounded random variables.
        alpha: Significance level.
        lo: Lower bound on the input random variables.
        hi: Upper bound on the input random variables.
        boundary_type: Type of sub-exponential uniform boundary.
            Either ``stitching`` or ``mixture`` (default).
            Note that ``stitching`` can be computed in closed-form but
            may yield wider intervals than ``mixture``.
        v_opt: value of intrinsic time where the boundary is optimized.
            Default is 10; set to ``None`` in *post-hoc* analyses (only)
            to optimize the boundary at the last time step.
        s: (``stitching`` only) controls how crossing probability is
            distributed over epochs
        eta: (``stitching`` only) controls the spacing of epochs
        c: scale parameter for the exponential CGF.
            By default (None), uses hi - lo.

    Returns:
        A tuple with the lower and upper confidence bounds.
    """
    _check_xs(xs, lo, hi)

    # Sample mean (centers)
    ts = np.arange(1, len(xs) + 1)
    mus = np.cumsum(xs) / ts

    # Sample variance (estimate of intrinsic time)
    shifted_mus = mus.copy()
    shifted_mus[1:], shifted_mus[0] = mus[:-1], mus[0]
    vs = np.maximum(1., np.cumsum((xs - shifted_mus) ** 2))

    # Sub-exponential uniform boundary (scale c)
    c = c if c is not None else hi - lo
    v_opt = v_opt if v_opt is not None else vs[-1]
    if boundary_type.lower() == "stitching":
        radii = boundaries.poly_stitching_bound(
            vs, alpha / 2, v_opt, c=c, s=s, eta=eta,
        ) / ts
    elif boundary_type.lower() == "mixture":
        radii = boundaries.gamma_exponential_mixture_bound(
            vs, alpha / 2, v_opt, c=c, alpha_opt=alpha / 2,
        ) / ts
    else:
        raise ValueError(
            f"boundary_type must be either 'stitching' or 'mixture'"
            f" (given: {boundary_type})"
        )
    return mus - radii, mus + radii


"""
Predictably-Mixed Confidence Sequences (Waudby-Smith and Ramdas, 2020)

These bounds are centered around a *weighted* mean of observations.
"""


def scale(xs: np.ndarray, lo: float = 0., hi: float = 1.):
    """Scale xs in [lo, hi] to zs in [0, 1] via

    z_t = (x_t - lo) / (hi - lo)."""
    return (xs - lo) / (hi - lo)


def unscale(zs: np.ndarray, lo: float = 0., hi: float = 1.):
    """Unscale zs in [0, 1] to xs in [lo, hi]."""
    return lo + (hi - lo) * zs


def confseq_pm_h(
        xs: np.ndarray,
        alpha: float = 0.05,
        lo: float = 0.,
        hi: float = 1.,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the (1-alpha)-confidence sequence given by
    a predictably-mixed Hoeffding process.

    We use the predictable mixture recommended by the original paper.
    (Waudby-Smith and Ramdas, 2020, equation 13)

    Returns a tuple of (LCBs, UCBs) for all timesteps.
    """
    _check_xs(xs, lo, hi)

    # Scale to [0, 1]
    zs = scale(xs, lo, hi)

    # Get the (recommended) predictable mixture lambda_t's
    t = len(zs)
    lambdas = np.sqrt(8 * np.log(2. / alpha) / (
        np.arange(1, t + 1) * np.log(np.arange(2, t + 2))
    ))  # 1..t
    lambdas = np.minimum(lambdas, 1.)

    # Compute psi_H(lambda_i)
    psis = 0.125 * lambdas ** 2

    # Compute CS for each time t
    lambdas_sum = np.cumsum(lambdas)
    centers = np.cumsum(zs * lambdas) / lambdas_sum
    radii = (np.log(2. / alpha) + np.cumsum(psis)) / lambdas_sum

    # Unscale to [lo, hi]
    centers = unscale(centers, lo, hi)
    radii = (hi - lo) * radii
    return centers - radii, centers + radii


def confseq_pm_eb(
        xs: np.ndarray,
        alpha: float = 0.05,
        c: float = 0.5,
        lo: float = 0.,
        hi: float = 1.,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the (1-alpha)-confidence sequence given by
    a predictably-mixed empirical Bernstein process.

    We use the predictable mixture recommended by the original paper.
    (Waudby-Smith and Ramdas, 2020, equation 16)

    Returns a tuple of (LCBs, UCBs) for all timesteps.

    TODO: double-check initial values of lambda_t and mu_t.
    """
    _check_xs(xs, lo, hi)
    assert 0 < c < 1, f"parameter c must be between 0 and 1 (given: {c})"

    # Scale to [0, 1]
    zs = scale(xs, lo, hi)

    # Get the (recommended) predictable mixture lambda_t's
    t = len(zs)
    mus = (0.5 + np.cumsum(zs)) / np.arange(2, t + 2)  # 1..t
    sigmas = (0.25 + np.cumsum((zs - mus) ** 2)) / np.arange(2, t + 2)  # 1..t
    lambdas = np.sqrt(2 * np.log(2. / alpha) / (
            sigmas * np.arange(2, t + 2) * np.log(np.arange(3, t + 3))
    ))  # 2..t+1
    lambdas = np.minimum(lambdas, c)
    # shift right by 1 so that lambdas only depend on the past
    lambdas[1:], lambdas[0] = lambdas[:-1], c

    # Compute v_i and psi_E(lambda_i)
    psis = -0.25 * (np.log(1. - lambdas) + lambdas)
    # shift mus by 1 for v_i calculation
    mus[1:], mus[0] = mus[:-1], 0.5
    vs = 4 * (zs - mus) ** 2

    # Compute CS for each time t
    lambdas_sum = np.cumsum(lambdas)
    centers = np.cumsum(zs * lambdas) / lambdas_sum
    radii = (np.log(2. / alpha) + np.cumsum(vs * psis)) / lambdas_sum

    # Unscale to [lo, hi]
    centers = unscale(centers, lo, hi)
    radii = (hi - lo) * radii
    return centers - radii, centers + radii


"""
Asymptotic Confidence Sequences (Waudby-Smith et al., 2021)

These bounds are centered around the sample mean of observations.
They only hold asymptotic guarantees and require independence of observations,
but they tend to work well in practice.
"""


def confseq_asymptotic(
        xs: np.ndarray,
        alpha: float = 0.05,
        lo: float = 0.,
        hi: float = 1.,
        t_star: int = None,
        assume_iid: bool = False,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the asymptotic (1-alpha)-confidence sequence given by
    Waudby-Smith et al. (2021).

    The width of the CS is optimized at t_star.
    For the non-iid case, we use an upper bound of the variance that works for
    bounded random variables (see Section C.2).
    """
    _check_xs(xs, lo, hi)
    t = len(xs)
    if t_star is None:
        # heuristic: midpoint in log-scale
        t_star = 10 ** (np.log10(t) / 2)

    # optimal width at t_star
    # rhosq = (2 * np.log(1 / alpha) + np.log(1 + 2 * np.log(1 / alpha))) / t_star
    rhosq = (-alpha ** 2 - 2 * np.log(alpha)
             + np.log(-2 * np.log(alpha) + 1 - alpha ** 2)) / t_star

    # Sample means at each time
    times = np.arange(1, t + 1)
    mus = np.cumsum(xs) / times

    # Use sample variance (with running average as centers) for both cases
    shifted_mus = mus.copy()
    shifted_mus[1:], shifted_mus[0] = mus[:-1], mus[0]
    vs = np.maximum(1., np.cumsum((xs - shifted_mus) ** 2)) / times
    # IID: apply Theorem 1
    if assume_iid:
        radii = np.sqrt(
            vs * 2 * (rhosq * times + 1) / (rhosq * times ** 2) *
            np.log(np.sqrt(rhosq * times + 1) / alpha)
        )
    # Non-IID: apply Theorem 6
    else:
        assert lo < hi
        # vs = np.repeat((hi - lo) ** 2 / 4, t)  # simple bound; not tight
        radii = np.sqrt(
            2 * (rhosq * times * vs + 1) / (rhosq * times ** 2) *
            np.log(np.sqrt(rhosq * times * vs + 1) / alpha)
        )

    # Radii of the confidence sequence
    centers = mus
    return centers - radii, centers + radii
