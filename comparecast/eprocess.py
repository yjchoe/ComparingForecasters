"""
E-values & anytime-valid p-values corresponding to the sub-exponential CSs
"""

import logging
import numpy as np
from scipy.special import loggamma, gammainc
from numpy.typing import ArrayLike

from comparecast.cgfs import get_cgf
from comparecast.scoring import get_scoring_rule
from comparecast.utils import preprocess_score_inputs


# For the normal mixture (Robbins, 1970), simply use the following
from confseq.boundaries import normal_log_mixture


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
            # unused when cs_v_rho_csq <= 0.
            # third term could still become small v_rho_csq grows large.
            leading_constant
            + loggamma(v_rho_csq)
            + np.log(gammainc(v_rho_csq, np.maximum(1e-8, cs_v_rho_csq)), where=cs_v_rho_csq > 0)
            - v_rho_csq * np.log(cs_v_rho_csq, where=cs_v_rho_csq > 0)
            + cs_v_csq
        ),
        leading_constant - rho_csq - np.log(v_rho_csq)  # upper bound (App. D)
    )


def calibrate_p_to_e(ps: np.ndarray, strategy: str = "mixture"):
    """Calibrate p-to-e, following Vovk and Wang (2020).

    [strategy]
    simple:
        f(p) = 1 / (2 * sqrt(p))
    mixture:
        f(p) = (1 - p + p * log(p)) / (p * log(p)^2).
    """
    assert np.logical_and(0 <= ps, ps <= 1).all(), \
        "Input array must only contain values in [0, 1]"

    if strategy == "simple":
        return 1 / (2 * np.sqrt(ps) + 1e-8)
    else:
        return (1 - ps + ps * np.log(ps + 1e-16)) / (ps * np.log(ps + 1e-16) ** 2 + 1e-8)


def eprocess_expm(
        xs: ArrayLike,
        vs: ArrayLike = None,
        lag: int = 1,
        lagged_null: str = "pw",
        v_opt: float = 10.,
        c: float = 0.1,
        alpha_opt: float = 0.05,
        clip_min: float = None,
        clip_max: float = 10000000,
        lambda_: float = None,
        cgf: str = "exponential",
        no_calibration: bool = False,
        calibration_strategy: str = "mixture",
        compute_p: bool = False,
        **cgf_kwargs,
) -> np.ndarray:
    """Compute the e-process for the weak one-sided null, i.e.,

        H_0^w: (1/t) * sum_{i=1,...,t} x_i <= 0

    The e-process (bounded by an exponential supermartingale) defined as:

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
        lag: forecast lag in the data. (default: 1)
        lagged_null: which weak null to test in the case of lag > 1.
            Options are: pw (easier to reject; default), w (harder to reject).
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
        no_calibration: if True, return unstopped sequential e-values before calibration.
        calibration_strategy: p-to-e calibration strategy. (Default: mixture)
        compute_p: if True, also returns an anytime-valid p-process by
            taking 1/max(E_i).
    Returns: np.ndarray
        e-process for H_0
    """
    T = len(xs)
    assert lag >= 1, f"forecast lag must be >= 1, got {lag}"
    evalues_per_lag = []

    # A pooled sample variance estimator
    times = np.arange(1, T + 1)
    sums = np.cumsum(xs)

    # Predictable means as centers of intrinsic time
    if vs is None:
        means = sums / times
        pred_means = means.copy()
        pred_means[lag:] = means[:T - lag]
        pred_means[:lag] = 0.0
    else:
        pred_means = None
        if lag > 1:
            logging.warning("custom input `vs[t]` must be `G[t-lag]`-measurable")

    for k in range(1, lag + 1):
        # Pointwise e-values (for "p is no better than q" under null)
        start, end = k - 1, T
        xs_k = xs[start:end:lag]
        sums_k = np.cumsum(xs_k)

        # Intrinsic time / sample variance (for each k)
        if vs is None:
            pred_means_k = pred_means[start:end:lag]
            vs_k = np.cumsum((xs_k - pred_means_k) ** 2)
        else:
            vs_k = vs[start:end:lag]
        v_opt = v_opt if v_opt is not None else vs_k[-1]

        if lag > 1:
            # Un-roll sums & vs to the original scale (some padding back and front)
            def _unroll(seq):
                """e.g., [1,2,3,4] -> [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]."""
                return np.concatenate([np.zeros(start), np.repeat(seq, lag)])[:T]

            sums_k = _unroll(sums_k)
            vs_k = _unroll(vs_k)
            assert len(sums_k) == len(vs_k) == T, (
                f"lagged sums/vs length mismatch: {len(sums_k)}, {len(vs_k)}, {T}"
            )

        log_e = np.zeros(T)
        # Get log(E_t) either with the conjugate mixture (default) ...
        if lambda_ is None:
            # "best rho" given by Proposition 3(a), Howard et al. (2021)
            # Code reference: OneSidedNormalMixture::best_rho in confseq/uniform_boundaries.h
            # Multi-lag case: since we're taking the minimum, the same alpha can be used
            alpha_os = 2 * alpha_opt
            rho = v_opt / (
                2 * np.log(1 / alpha_os) + np.log(1 + 2 * np.log(1 / alpha_os))
            )
            log_e[start:] = gamma_exponential_log_mixture(sums_k[start:], vs_k[start:], rho, c)
        # ... or with a user-provided lambda & cgf
        else:
            cgf_fn = get_cgf(cgf)
            logging.info("using cgf %s with fixed lambda %g", cgf, lambda_)
            log_e[start:] = lambda_ * sums_k[start:] - cgf_fn(lambda_, c=c, **cgf_kwargs) * vs_k[start:]

        # prevent overflow by first computing log(E) and clipping large values
        log_e = np.clip(
            log_e,
            a_min=np.log(clip_min) if clip_min is not None else None,
            a_max=np.log(clip_max) if clip_max is not None else None,
        )
        evalues = np.exp(log_e)

        evalues_per_lag.append(evalues)

    if lag == 1:
        return evalues_per_lag[0]
    elif no_calibration:
        # sequential e-values (not stopped)
        if lagged_null == "pw":
            logging.info("computing e-process for the *period-wise* weak null for lagged forecasts")
            return np.mean(evalues_per_lag, axis=0)
        else:
            return np.min(evalues_per_lag, axis=0)
    else:
        # e-to-p, combine p, and then p-to-e
        if lagged_null == "pw":
            logging.info("computing e-process for the *period-wise* weak null for lagged forecasts")
            mean_max_e = np.maximum.accumulate(evalues_per_lag, axis=1).mean(axis=0)
            combined_p = np.minimum(1, np.exp(1) * np.log(lag) / mean_max_e)
        else:
            # Need to take the minimum for the weak null
            pvalues_per_lag = np.minimum(1, 1 / np.maximum.accumulate(evalues_per_lag, axis=1))
            combined_p = np.max(pvalues_per_lag, axis=0)
        return calibrate_p_to_e(combined_p, strategy=calibration_strategy)

    # LEGACY CODE: lag-1 case only
    #
    # # Sample mean (centers)
    # t = len(xs)
    # times = np.arange(1, t + 1)
    # sums = np.cumsum(xs)
    #
    # # Sample variance (estimate of intrinsic time)
    # if vs is None:
    #     mus = sums / times
    #     shifted_mus = mus.copy()
    #     shifted_mus[1:], shifted_mus[0] = mus[:-1], 0.0
    #     vs = np.cumsum((xs - shifted_mus) ** 2)
    # v_opt = v_opt if v_opt is not None else vs[-1]
    #
    # # Get log(E_t) either with the conjugate mixture (default) ...
    # if lambda_ is None:
    #     # "best rho" given by Proposition 3(a), Howard et al. (2021)
    #     # Code reference: OneSidedNormalMixture::best_rho in confseq/uniform_boundaries.h
    #     alpha_os = 2 * alpha_opt
    #     rho = v_opt / (
    #         2 * np.log(1 / alpha_os) + np.log(1 + 2 * np.log(1 / alpha_os))
    #     )
    #     log_e = gamma_exponential_log_mixture(sums, vs, rho, c)
    # # ... or with a user-provided lambda & cgf
    # else:
    #     cgf_fn = get_cgf(cgf)
    #     logging.info("using cgf %s with fixed lambda %g", cgf, lambda_)
    #     log_e = lambda_ * sums - cgf_fn(lambda_, c=c, **cgf_kwargs) * vs
    #
    # # prevent overflow by first computing log(E) and clipping large values
    # log_e = np.clip(
    #     log_e,
    #     a_min=np.log(clip_min) if clip_min is not None else None,
    #     a_max=np.log(clip_max) if clip_max is not None else None,
    # )
    # evalues = np.exp(log_e)
    # return evalues


def boundary_csf(
        ps: ArrayLike,
        qs: ArrayLike,
        scoring_rule: str = "brier",
) -> np.ndarray:
    """The boundary of the strong null under consistent scoring functions:

        kappa_nu(min{p,q}, max{p,q}).

    Required for GROW e-values developed by Henzi and Ziegel (2021).
    (Implementation ported from their R version.)
    The specific form depends on which scoring rule is used.
    """
    m = np.minimum(ps, qs)
    M = np.maximum(ps, qs)
    if scoring_rule == "brier":
        return (m + M) / 2
    elif scoring_rule == "logarithmic":
        return (np.log(1 - m) - np.log(1 - M)) / (
            np.log(M) - np.log(1 - M) - np.log(m) + np.log(1 - m)
        )
    elif scoring_rule == "spherical":
        nm = np.sqrt(2 * m ** 2 - 2 * m + 1)
        nM = np.sqrt(2 * M ** 2 - 2 * M + 1)
        return ((M - 1) * nm - (m - 1) * nM) / ((2 * M - 1) * nm - (2 * m - 1) * nM)
    elif scoring_rule == "dominance":
        return ps
    else:
        raise ValueError("invalid scoring rule for boundary_csf "
                         "(brier, logarithmic, spherical, or dominance)")


def lagged_pred_indices(
        T: int,
        lag: int,
        offset: int,
) -> np.ndarray:
    """Compute the indices for each of the h subsequences I_{T,k},
    where k is an offset {1, 2, ..., lag}."""
    assert 1 <= offset <= lag
    assert lag <= T

    return np.arange(offset, np.floor((T - offset) / lag) - 1, lag)


def eprocess_hz(
        ps: ArrayLike,
        qs: ArrayLike,
        ys: ArrayLike,
        aligned_outcomes: bool = True,
        scoring_rule: str = "brier",
        lag: int = 1,
        lambda_: float = None,
        alt_prob: float = 0.75,
        clip_min: float = None,
        clip_max: float = 10000000,
) -> np.ndarray:
    """E-process for conditional forecast dominance by Henzi and Ziegel (2021).

    Tests the null that "p is no better than q at all times" (**opposite** of the original paper).
    Applies for binary forecasts only.

    *For lagged forecasts, outcomes (ys) are assumed to be aligned with forecasts already.*

    *These are NOT the stopped e-values, which require a correction for lag > 1;
     see original implementation & paper for details.*

    Args:
        ps: probability forecasts of the first forecaster (better under the null)
        qs: probability forecasts of the second forecaster (worse under the null)
        ys: binary outcomes (aligned or not; see `aligned_outcomes`)
        aligned_outcomes: whether outcomes are aligned with the forecasts or not. (default: `True`)
        scoring_rule: type of scoring rule (brier, logarithmic, spherical, or dominance)
        lag: forecast lag (default: 1)
        lambda_: choice of lambda. if None (default), choose the GROW lambda (Thm 2).
        alt_prob: alternative weight used for the GROW e-value.
        clip_min: lower bound on the output e-values.
        clip_max: upper bound on the output e-values.
            Due to possible overflows, clip_max is set to 10000000 by default.

    Returns:
        An array of (non-stopped) e-values.
    """
    # Setup
    ps, ys = preprocess_score_inputs(ps, ys)
    qs, _ = preprocess_score_inputs(qs, ys)
    assert ys.shape[1] == 2, "HZ's e-process only supports binary outcomes"
    ps, qs, ys = [arr[:, 1] for arr in [ps, qs, ys]]

    # Alternatives & boundaries (pointwise)
    pis = alt_prob * ps + (1 - alt_prob) * qs
    kappas = boundary_csf(qs, ps, scoring_rule)

    T = len(ys)
    score_fn = get_scoring_rule(scoring_rule)

    # Computed separately for each lag offset and summed over
    # (full sequence is processed directly if lag == 1)
    # evalues_per_lag = []

    combined_e = np.zeros(T)
    indices = np.tile(np.arange(1, lag + 1), int(np.ceil(T / lag)))[:T]  # 1, 2, 3, 1, 2, 3, ..., 1
    for k in range(1, lag + 1):
        # Pointwise e-values (for "p is no better than q" under null)
        if aligned_outcomes:
            start, end = k - 1, T
            ps_k, qs_k, pis_k, kappas_k = [fc[start:end:lag] for fc in [ps, qs, pis, kappas]]
            ys_k = ys[start:end:lag]
        else:
            # untested
            start, end = k - 1, T - lag + 1
            ps_k, qs_k, pis_k, kappas_k = [fc[start:end:lag] for fc in [ps, qs, pis, kappas]]
            ys_k = ys[start+lag-1::lag]
            assert len(ps_k) == len(ys_k), f"sliced forecasts and outcomes length mismatch: {len(ps_k)} != {len(ys_k)}"
        if lambda_ is None:
            # GROW
            evalues = (1 - ys_k - pis_k) / (1 - ys_k - kappas_k)
        else:
            scores = score_fn(qs_k, ys_k) - score_fn(ps_k, ys_k)
            i_qp = qs_k > ps_k
            scores_iqp = score_fn(qs_k, i_qp) - score_fn(ps_k, i_qp)
            evalues = 1 + lambda_ * scores / np.abs(scores_iqp)

        # Take product (or sum in log scale)
        # (prevent overflow by first computing log(E) and clipping large values)
        log_e = np.cumsum(np.log(evalues))
        log_e = np.clip(
            log_e,
            a_min=np.log(clip_min) if clip_min is not None else None,
            a_max=np.log(clip_max) if clip_max is not None else None,
        )
        evalues = np.exp(log_e)

        # take additive increments on each k and put back in the original array
        e_increments = np.diff(evalues, prepend=0)
        assert len(e_increments) == sum(indices == k)
        combined_e[indices == k] = e_increments

    return np.cumsum(combined_e) / lag

        # Revert to original timescale (some padding back and front)
        # evalues = np.concatenate([np.ones(start),
        #                           np.repeat(evalues, lag),
        #                           evalues[-1] * np.ones((T - start) % lag)])
        # assert len(evalues) == T, f"lagged evalue length mismatch: {len(evalues)} != {T}"
        # evalues_per_lag.append(evalues)

    # return np.mean(evalues_per_lag, axis=0)
