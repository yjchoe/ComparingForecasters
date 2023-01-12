"""
Implementation of standard forecasters and forecasting utilities
"""

from collections import OrderedDict
import logging
import os.path
from typing import List, Tuple, Union, Collection, Callable
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from tqdm import trange

from comparecast.kernels import get_kernel


def forecast(
        data: Union[ArrayLike, pd.DataFrame],
        forecasters: Union[str, List[str]],
        out_file: str = None,
):
    """Generate forecasts on data and store them in out_file."""
    if out_file is not None and os.path.exists(out_file):
        raise ValueError(f"output file {out_file} already exists")

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame({"time": np.arange(1, len(data) + 1),
                             "y": np.ndarray(data)})

    if "all" in forecasters:
        forecasters = FORECASTERS_DEFAULT

    logging.info("forecasters:", ", ".join(forecasters))
    for name in forecasters:
        forecaster = get_forecaster(name)
        data[name] = forecaster(data["y"].values)

    if out_file is not None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        data.to_csv(out_file, index=False)
    return data


def shift_forecasts(
        ps: ArrayLike,
        initial_value: float = 0.5,
) -> np.ndarray:
    """Forecasts (p_2, ..., p_{T+1}) are made using the data (y_1, ..., y_T),
    so we shift the forecasts 'forward' to return (p_1, ..., p_T).

    p_1 is determined by initial_value; p_{T+1} is dropped.
    """
    return np.insert(np.array(ps)[:-1], 0, initial_value)


def forecast_oracle(ys: ArrayLike, slack: float = 0.) -> np.ndarray:
    """The oracle forecaster with slack parameter,
    p_t = y_t * (1 - slack) + (1 - y_t) * slack.

    E.g., ys = [0, 1, 1, 0], slack = 0.0: ps = [0.0, 1.0, 1.0, 0.0]. (oracle)
    E.g., ys = [0, 1, 1, 0], slack = 0.2: ps = [0.2, 0.8, 0.8, 0.2].
    E.g., ys = [0, 1, 1, 0], slack = 0.5: ps = [0.5, 0.5, 0.5, 0.5]. (random)
    """
    assert 0. <= slack <= 1.
    return (1. - slack) * ys + slack * (1. - ys)


def forecast_laplace(ys: ArrayLike, c: float = 0.5) -> np.ndarray:
    """p_t = (k + c)/(t + 1), where k is the number of 1's observed so far
    and c is an initial baseline.

    E.g., ys = [0, 1, 1, 0], c = 0.5: ps = [0.5, 0.25, 0.5, 0.625].
    """
    assert 0. <= c <= 1.
    ps = (np.cumsum(ys) + c) / (np.arange(1, len(ys) + 1) + 1)
    return shift_forecasts(ps, initial_value=c)


def forecast_constant(ys: ArrayLike, c: float = 0.5) -> np.ndarray:
    """p_t = const, e.g., random, always-zero, and always-one forecasters."""
    ps = np.repeat(c, len(ys))
    return shift_forecasts(ps, initial_value=c)


def forecast_random(
        ys: ArrayLike,
        u: float = 0.0,
        rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """p_t = u or 1-u with equal probability."""
    ps = rng.uniform(0, 1, len(ys))
    return (1 - u) * ps + u * (1 - ps)


def interleave(
        ps: ArrayLike,
        qs: ArrayLike,
        replace_every: int = 2,
) -> np.ndarray:
    """Interleave qs onto ps, replacing every other value.

    E.g., ps = [1.0, 1.0, 1.0, 1.0, 1.0], qs = [0.5, 0.5, 0.5, 0.5, 0.5].
    replace_every=1: [0.5, 0.5, 0.5, 0.5, 0.5]
    replace_every=2: [0.5, 1.0, 0.5, 1.0, 0.5]
    replace_every=4: [0.5, 1.0, 1.0, 1.0, 0.5]
    replace_every=5: [1.0, 1.0, 1.0, 1.0, 1.0]
    """
    assert ps.shape == qs.shape
    assert 1 <= replace_every <= len(ps)
    out = ps.copy()
    if replace_every < len(ps):
        out[::replace_every] = qs[::replace_every]
    return out


def forecast_seasonal(
        ys: ArrayLike,
        season_starts: Collection[int],
        forecast_fn: Callable[[ArrayLike, float], np.ndarray] = None,
        baseline: float = 0.5,
        reversion_factor: float = 1 / 3,
) -> np.ndarray:
    """Forecast outcomes within each season.

    Seasons are delimited by indices in `season_starts`."""
    seasons = np.insert(season_starts, len(season_starts), len(ys))
    forecasts = []
    for start, end in zip(seasons[:-1], seasons[1:]):
        ps = forecast_fn(ys[start:end], baseline)
        forecasts.append(ps)
        baseline = (1 - reversion_factor) * ps[-1] + reversion_factor * 0.5
    return np.concatenate(forecasts)


"""
K29 defensive forecasting algorithms
"""


def find_root(fn, method="bisect", maxiter=10):
    """Use scipy's root finder to find a root of fn, defined on [0, 1].

    Return (1 + sign(fn)) / 2 if root does not exist."""
    try:
        sol = root_scalar(fn, method=method, bracket=[0, 1], maxiter=maxiter)
        return sol.root
    except ValueError:
        return (1 + np.sign(fn(0))) / 2.


def forecast_k29(
        ys: ArrayLike,
        kernel_params: Tuple[str, float] = ("gaussian", 0.1),
        prev_forecasts: ArrayLike = None,
        verbose: bool = True,
) -> np.ndarray:
    """Make binary forecasts using the K29 defensive forecasting algorithm
    (Vovk 2005).

    Reference: http://onlineprediction.net/index.html?n=Main.K29
    """
    # Get kernel functions
    kernel_fn = get_kernel(*kernel_params)

    # K29 defensive forecasting
    if prev_forecasts is not None:
        assert len(prev_forecasts) < len(ys)
        forecasts = list(prev_forecasts)
    else:
        forecasts = []
    if verbose:
        iters = trange(len(forecasts), len(ys),
                       desc="forecast_k29_{}{}".format(*kernel_params))
    else:
        iters = range(len(forecasts), len(ys))
    for _ in iters:
        # Skeptic's bet: kernelized sum of previous (y - forecast)s
        def _bet(p):
            return sum([kernel_fn(p, p_i) * (y_i - p_i)
                        for y_i, p_i in zip(ys, forecasts)])
        p_t = find_root(_bet)
        forecasts.append(p_t)

    return shift_forecasts(np.array(forecasts))


"""
Preset forecasters as functions
"""

FORECASTERS = OrderedDict({
    "laplace": lambda ys: forecast_laplace(ys),
    "k29_poly3": lambda ys: forecast_k29(ys, ("poly", 3)),
    "k29_rbf0.01": lambda ys: forecast_k29(ys, ("rbf", 0.01)),
    "k29_epa": lambda ys: forecast_k29(ys, ("epa", 2.0)),
    "always_0.5": lambda ys: forecast_constant(ys, 0.5),
    "always_0": lambda ys: forecast_constant(ys, 0.0),
    "always_1": lambda ys: forecast_constant(ys, 1.0),
    "constant_0.5": lambda ys: forecast_constant(ys, 0.5),
    "constant_0": lambda ys: forecast_constant(ys, 0.0),
    "constant_1": lambda ys: forecast_constant(ys, 1.0),
    "random": lambda ys: forecast_random(ys),
})
FORECASTERS_ALL = list(FORECASTERS.keys())
FORECASTERS_DEFAULT = ["laplace", "k29_poly3", "k29_rbf0.01",
                       "constant_0.5", "constant_0", "constant_1", "random"]


def get_forecaster(name: str):
    """Return a forecaster as a function given their name."""
    try:
        return FORECASTERS[name]
    except KeyError:
        raise KeyError(f"invalid forecaster name {name}, try one of: "
                       ", ".join(FORECASTERS_ALL)) from None
