"""
Forecast datasets & data generators
"""

import os.path
from typing import Union, List
import numpy as np
from numpy.random import default_rng
import pandas as pd


"""
Synthetic sequences of (non-iid) true probs/means
"""


def bernoulli(
        n: int,
        p: Union[float, List, np.ndarray] = 0.5,
        rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Return a sequence of Bernoulli random variables."""
    return rng.binomial(1, p, size=n)


def zeros_then_ones(
        n_zeros: int,
        n_ones: int,
) -> np.ndarray:
    """Return a sequence of `n_zeros` 0's followed by `n_ones` 1's."""
    return np.concatenate([np.zeros((n_zeros, )), np.ones((n_ones, ))])


def zeros_then_ones_repeated(
        n: int,
        n_spans: int,
        roll: int = 0,
) -> np.ndarray:
    """Return a repeating sequence of 0's and 1's."""
    assert 1 <= n_spans <= n
    span = n // n_spans
    ys = np.concatenate([
        zeros_then_ones(span, span)
        for _ in range((n_spans + 1) // 2)
    ])[:n]
    return np.roll(ys, roll)


def randoms_zeros_then_ones(
        n_randoms: int,
        n_zeros: int,
        n_ones: int,
        p: float = 0.5,
        rng: np.random.Generator = default_rng(),
) -> np.ndarray:
    """Return a sequence of `n_randoms` Bernoulli(p) random variables,
    followed by `n_zeros` 0's and `n_ones` 1's."""
    return np.concatenate([rng.binomial(1, p, size=n_randoms),
                           np.zeros((n_zeros, )),
                           np.ones((n_ones, ))])


def default(
        n: int,
        n_spans: int = 5,
):
    """Default setting for simulated experiments in the paper.

    First span is random; the next `n_spans - 1` spans alternate between constant zeros/ones.
    Odd number of spans (for even n) is preferred.
    """
    assert n >= n_spans, f"default setting requires n > n_spans (given: {n} > {n_spans})"

    span_length = n // n_spans
    seqs = [np.repeat(0.5, span_length)]
    for span in range(1, n_spans):
        if span % 2 == 0:
            seqs.append(np.zeros((span_length, )))
        else:
            seqs.append(np.ones((span_length, )))
    return np.concatenate(seqs)


def default_logt(
        n: int,
):
    """Default setting but in log scale of time.

    Random for the first 100, and then repeated zeros-then-ones in
    each log-scale span ([101, 1000], [1001, 10000], ...).
    """
    n_spans = int(np.log10(n))
    assert n_spans >= 2, f"default setting requires n > 100 (given: {n})"

    seqs = [np.repeat(0.5, 100)]
    for span in range(2, n_spans):
        r = 10 ** (span + 1) - 10 ** span
        seqs.append(zeros_then_ones(r // 4, r // 4))
        seqs.append(zeros_then_ones(r // 4, r // 4)[::-1])
    return np.concatenate(seqs)


def sigmoid(
        n: int,
        changepoint: float = 0.25,
) -> np.ndarray:
    """Return a sequence of values between [0, 1] that follow a sigmoid fn."""
    grid = 20. * (np.linspace(0, 1, num=n) - changepoint)  # [-10, 10]
    return 1. / (1. + np.exp(-grid))


"""
Presets: 
    binary: pd.DataFrame(time, data, true_probs)
    continuous: pd.DataFrame(time, data, true_means, true_params)
"""


def make_preset(
        true_probs: np.ndarray,
        rng: np.random.Generator = default_rng(),
):
    """A helper function that makes binary data given true probabilities."""
    n = len(true_probs)
    ys = bernoulli(n, true_probs, rng=rng)
    return pd.DataFrame({
        "time": np.arange(1, n + 1),
        "y": ys,
        "true_probs": true_probs,
    })


def preset_default(
        n: int,
        noise: float = 0.1,
        rng: np.random.Generator = default_rng(),
        n_spans: int = 5,
) -> pd.DataFrame:
    """Default synthetic data without log-scale.

    Generated from a noisy version of
    2000 1/2s, 2000 1s, 2000 0s, 2000 1s, and 2000 0s.
    """
    pattern = default(n, n_spans=n_spans)
    true_probs = 0.8 * pattern + 0.2 * (1 - pattern)
    true_probs = np.clip(true_probs + rng.normal(0, noise, n), 0, 1)
    return make_preset(true_probs, rng)


def preset_default_logt(
        n: int,
        noise: float = 0.1,
        rng: np.random.Generator = default_rng(),
) -> pd.DataFrame:
    """Default synthetic data.

    Generated from a noisy version of
    100 1/2s, 1000 1s, 1000 0s, 1000 1s, 1000 0s, ..., 1000 1s, and 500 0s."""
    pattern = default_logt(n)
    true_probs = 0.8 * pattern + 0.2 * (1 - pattern)
    true_probs = np.clip(true_probs + rng.normal(0, noise, n), 0, 1)
    return make_preset(true_probs, rng)


def preset_random(
        n: int,
        noise: float = 0.1,
        rng: np.random.Generator = default_rng(),
) -> pd.DataFrame:
    """Random synthetic data: true_prob == 0.5 + noise for all rounds."""
    true_probs = np.repeat(0.5, n)
    true_probs = np.clip(true_probs + rng.normal(0, noise, n), 0, 1)
    return make_preset(true_probs, rng)


def preset_sigmoid(
        n: int,
        noise: float = 0.25,
        rng: np.random.Generator = default_rng(),
        changepoint: float = 0.25,  # between [0, 1]
) -> pd.DataFrame:
    """A smoothly increasing function with a changepoint + sinusoidal noise."""
    pattern = sigmoid(n, changepoint)
    sine_noise = np.sin(0.1 * np.arange(n)) + rng.normal(0, 1, n)
    true_probs = np.clip(pattern + noise * sine_noise, 0, 1)
    return make_preset(true_probs, rng)


def make_preset_beta(
        true_means: np.ndarray,
        rng: np.random.Generator = default_rng(),
) -> pd.DataFrame:
    """A helper function that makes continuous data given true means, where
    y_t ~ Beta(r_t, 1 - r_t)."""
    n = len(true_means)
    true_params = [true_means, 1. - true_means]
    ys = rng.beta(*true_params)
    out = {
        "time": np.arange(1, n + 1),
        "y": ys,
        "true_means": true_means,
        "true_dist": ["beta" for _ in range(n)],
    }
    out.update({
        f"true_param{i}": true_param
        for i, true_param in enumerate(true_params)
    })
    return pd.DataFrame(out)


def preset_beta(
        n: int,
        noise: float = 0.1,
        rng: np.random.Generator = default_rng(),
) -> pd.DataFrame:
    """Synthetic data with continuous outcomes taking values in [-1, 1].

        z_t ~ Beta(r_t, 1 - r_t)
        y_t = 2 * z_t - 1
    """
    pattern = sigmoid(n, changepoint=0.25)
    true_means = 0.8 * pattern + 0.2 * (1 - pattern)
    true_means = np.clip(true_means + rng.normal(0, noise, n), 0.01, 0.99)
    return make_preset_beta(true_means, rng)


# pd.DataFrame(time, data, true_probs)
PRESETS = {
    "default": preset_default,
    "default_logt": preset_default_logt,
    "random": preset_random,
    "sigmoid": preset_sigmoid,
    "beta": preset_beta,
}


def get_data(
        data_name: str,
        size: int = 0,
        noise: float = 0.1,
        rng: Union[int, np.random.Generator] = default_rng(),
) -> pd.DataFrame:
    """Get data from its name or filename, up to n_rounds."""
    if os.path.exists(data_name):
        data = pd.read_csv(data_name)
        if size > 0:
            data = data[:size]
    else:
        try:
            if isinstance(rng, int):
                rng = default_rng(rng)
            assert size > 0, f"specify data size for synthetic data generation"
            data = PRESETS[data_name](size, noise, rng)
        except KeyError:
            raise KeyError(
                f"data name {data_name} is not one of the presets, "
                f"available: " + " ".join(list(PRESETS.keys()))) from None
    return data
