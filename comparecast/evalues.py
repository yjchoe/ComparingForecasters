"""
E-values & anytime-valid p-values corresponding to the sub-exponential CSs
"""

import logging
import numpy as np
from scipy.special import loggamma, gammainc
from numpy.typing import ArrayLike

from comparecast.cgfs import get_cgf


def gamma_exponential_log_mixture(
        sums: np.ndarray,
        vs: np.ndarray,
        rho: float,
        c: float,
) -> np.ndarray:
    """Computes the gamma-exponential mixture in logarithm.

    Extends :py:func:`confseq.boundaries.gamma_exponential_log_mixture` to
    allow for negative sums.
    """
    csq = c ** 2
    rho_csq = rho / csq
    v_rho_csq = (vs + rho) / csq
    cs_v_csq = (c * sums + vs) / csq
    cs_v_rho_csq = cs_v_csq + rho_csq

    leading_constant = (
        rho_csq * np.log(rho_csq)
        - loggamma(rho_csq)
        - np.log(gammainc(rho_csq, rho_csq))
    )
    return np.where(
        cs_v_rho_csq > 0,
        (
            leading_constant
            + loggamma(v_rho_csq)
            + np.log(gammainc(v_rho_csq, np.maximum(1e-8, cs_v_rho_csq)))
            - v_rho_csq * np.log(np.maximum(1e-8, cs_v_rho_csq))
            + cs_v_csq
        ),
        leading_constant - rho_csq - np.log(v_rho_csq)  # upper bound (App. D)
    )


def evalue_expm(
        xs: ArrayLike,
        vs: ArrayLike = None,
        v_opt: float = 1.,
        c: float = 0.1,
        alpha_opt: float = 0.05,
        clip_min: float = None,
        clip_max: float = 10000,
        lambda_: float = None,
        cgf: str = "exponential",
        **cgf_kwargs,
) -> np.ndarray:
    """Compute the e-values for the weak one-sided null, i.e.,

        H_0^w: (1/t) * sum_{i=1,...,t} x_i <= 0

    The e-value corresponds to the realization of an e-process, defined as:

        L_t(lambda) = exp( lambda * sum_{i=1..t} x_i - cgf(lambda) * V_t )

    where V_t represents the intrinsic time
    (variance process for the sums of x_i minus their unknown means).

    By default, we use the gamma-exponential conjugate mixture (CM) to obtain
    an optimal value for lambda.
    This can be overridden by setting lambda_ to a specific constant
    (and possibly also modifying cgf).

    Args:
        xs: input sequence
        vs: (optional) variance process for the sum of xs.
            Default (by setting to None) is the running empirical variance.
        v_opt: value of intrinsic time for which lambda is optimized
        c: A parameter that controls how aggressively one can bet against H_0.
            Also used as the parameter to the exponential/gamma/poisson CGFs.
        alpha_opt: a "significance level" for which the parameters to
            the gamma mixture density are optimized.
            Note that the e-value is one-sided, so this value should be
            half of the value used for the corresponding CS.
        clip_min: lower bound on the output e-values.
        clip_max: upper bound on the output e-values.
            Due to possible overflows, clip_max is set to 10000 by default.
        lambda_: (optional) if provided, use this (fixed) lambda instead of
            the one obtained by a gamma-exponential mixture. (Default: None)
        cgf: which CGF to use when using a custom lambda_.
            Options include bernoulli, gaussian, poisson, gamma, and
            exponential (default), reflecting assumptions made on x_i.
    Returns: np.ndarray
        sequence of e-values for H_0
    """

    # Sample mean (centers)
    t = len(xs)
    times = np.arange(1, t + 1)
    sums = np.cumsum(xs)

    # Sample variance (estimate of intrinsic time)
    if vs is None:
        mus = sums / times
        shifted_mus = mus.copy()
        shifted_mus[1:], shifted_mus[0] = mus[:-1], 0.0
        vs = np.cumsum((xs - shifted_mus) ** 2)

    # Get log(E_t) either with the conjugate mixture (default) ...
    if lambda_ is None:
        # "best rho" given by Proposition 3(a), Howard et al. (2021)
        rho = v_opt / (2 * np.log(1 / (2 * alpha_opt))
                       + np.log(1 + 2 * np.log(1 / (2 * alpha_opt))))
        log_e = gamma_exponential_log_mixture(sums, vs, rho, c)
    # ... or with a user-provided lambda & cgf
    else:
        cgf_fn = get_cgf(cgf)
        logging.info("using cgf %s with fixed lambda %g", cgf, lambda_)
        log_e = lambda_ * sums - cgf_fn(lambda_, c=c, **cgf_kwargs) * vs

    # prevent overflow by first computing log(E) and clipping large values
    log_e = np.clip(
        log_e,
        a_min=np.log(clip_min) if clip_min is not None else None,
        a_max=np.log(clip_max) if clip_max is not None else None,
    )
    evalues = np.exp(log_e)
    return evalues
