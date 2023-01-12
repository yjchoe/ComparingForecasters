"""
Proper scoring rules for probabilistic forecasts.

General usage:

    import comparecast as cc

    brier = cc.BrierScore()

    scores = brier([0.7, 0.1, 0.5], [1, 0, 1])  # binary forecasts
    scores = brier(
        [[0.2, 0.1, 0.7], [0.9, 0.1, 0.0]],
        [2, 0],
    )  # categorical forecasts

    expected_scores = brier.expected_score([0.7, 0.1, 0.5], [0.9, 0.2, 0.7])
        # binary forecasts scored on on bernoulli outcomes
    expected_scores = brier.expected_score(
        [[0.2, 0.1, 0.7], [0.9, 0.1, 0.0]],
        [[0.0, 0.1, 0.9], [0.8, 0.1, 0.1]],
    )  # categorical forecasts scored on unknown categorical probabilities

Generally speaking, we implement both the score S(p, y) and its expected score
S(p; r) = E_r[S(p, y)], where the latter can be computed as
a linear transformation of the score if the score has a
linear equivalent (Lai et al., 2011).

Following Gneiting & Raftery (2007), we follow the convention that
higher scores mean better forecasts.

TODO: real-valued outcomes (some expected scores may be different)
"""

from typing import Tuple, Union, Callable
import numpy as np
from numpy.typing import ArrayLike

from comparecast.utils import preprocess_score_inputs, convert_to_onehot


"""
Scoring rules: S(p, y) measures the quality of forecast p given outcome y.
"""


class ScoringRule:
    """A generic scoring rule object.

    Child class should implement `score()` and `expected_score()`.

    Attributes:
        is_proper: whether the scoring rule is proper
        bounds (a `@property`): lower and upper bounds of the score function
        name: name of the scoring rule
    Methods:
        __init__, score, expected_score
    """
    def __init__(
            self,
            is_proper: bool,
            bounds: Tuple[float, float] = (-np.inf, np.inf),
            name: str = None,
    ):
        self.is_proper = is_proper
        self._bounds = bounds
        self.name = "ScoringRule" if name is None else name

    def __repr__(self):
        return self.name

    def __call__(self, ps: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Equivalent to `self.score()`.

        Child classes should override the `score` method.
        """
        return self.score(ps, ys)

    @property
    def bounds(self) -> Tuple[float, float]:
        """Lower and upper bounds on the score.

        If unknown, set to `(-np.inf, np.inf)` by default."""
        return self._bounds

    @bounds.setter
    def bounds(self, value: Tuple[float, float]):
        self._bounds = value

    def score(self, ps: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        raise NotImplementedError

    def expected_score(self, ps: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner.

        Same as `self.score()` if scoring rule is linear in y.
        """
        raise NotImplementedError


class BrierScore(ScoringRule):
    """The Brier score for binary or categorical outcomes:

        S(p, y) = 1 - (1/2) * ||p - y||_2^2.

    The Brier score is strictly proper.
    Note that the (1/2) factor scales the range of the score to [0, 1].
    """
    def __init__(self):
        super().__init__(
            is_proper=True,
            bounds=(0, 1),
            name="BrierScore",
        )

    def score(self, ps: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        ps, ys = preprocess_score_inputs(ps, ys)
        return 1 - 0.5 * np.square(ps - ys).sum(axis=1)
        # return 0.5 * np.square(ps - ys).sum(axis=1)

    def expected_score(self, ps: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner.

        The Brier score has a different expression for the expected score:
            S(p; r) = 1/2 - (1/2) * ||p||^2 + <p, r>.
        """
        ps, rs = preprocess_score_inputs(ps, rs)
        return 0.5 - 0.5 * (ps * ps).sum(axis=1) + (ps * rs).sum(axis=1)
        # return 0.5 * ((ps * ps).sum(axis=1) - 2 * (ps * rs).sum(axis=1) + 1)


class LogarithmicScore(ScoringRule):
    """The (truncated) logarithmic score for binary or categorical outcomes:

        S(p, y) = <y, log(p)>.

    The truncation of forecasts is determined by the `eps` parameter:
    ```
        ps = np.clip(ps, eps, 1 - eps)
    ```

    The score is strictly proper if and only if `eps == 0`.
    In the implementation, if `eps` is no greater than `1e-8` (default),
    then the score is treated as improper.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        is_proper = self.eps <= 1e-8
        bounds = (np.log(eps) if eps > 0 else -np.inf, 0)
        super().__init__(
            is_proper=is_proper,
            bounds=bounds,
            name="LogarithmicScore",
        )

    def score(self, ps: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        ps, ys = preprocess_score_inputs(ps, ys)
        ps = np.clip(ps, self.eps, 1 - self.eps)
        return (ys * np.log(ps)).sum(axis=1)

    def expected_score(self, ps: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner."""
        return self.score(ps, rs)


class SphericalScore(ScoringRule):
    """The spherical score for binary or categorical outcomes:

        S(p, y) = <p, y> / ||p||_2.

    The spherical score is strictly proper.
    """

    def __init__(self):
        super().__init__(
            is_proper=True,
            bounds=(0, 1),
            name="SphericalScore",
        )

    def score(self, ps: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        ps, ys = preprocess_score_inputs(ps, ys)
        return (
            (ps * ys).sum(axis=1) /
            np.maximum(np.sqrt((ps * ps).sum(axis=1)), 1e-8)
        )

    def expected_score(self, ps: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner."""
        return self.score(ps, rs)


class ZeroOneScore(ScoringRule):
    """The zero-one score for binary or categorical outcomes:

        S(p, y) = <argmax(p), y>,

    where the argmax operator returns a one-hot vector.

    The zero-one score is proper but not strictly proper.
    """

    def __init__(self):
        super().__init__(
            is_proper=True,
            bounds=(0, 1),
            name="ZeroOneScore",
        )

    def score(self, ps: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        ps, ys = preprocess_score_inputs(ps, ys)
        ps = convert_to_onehot(ps.argmax(axis=1), ps.shape[1])
        return (ps * ys).sum(axis=1)

    def expected_score(self, ps: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner."""
        return self.score(ps, rs)


class AbsoluteScore(ScoringRule):
    """The absolute score for binary or categorical outcomes:

        S(p, y) = 1 - (1/2) * ||p - y||_1.

    The absolute score is *improper* for probability forecasts.
    Note that the (1/2) factor scales the range of the score to [0, 1].
    """
    def __init__(self):
        super().__init__(
            is_proper=False,
            bounds=(0, 1),
            name="AbsoluteScore",
        )

    def score(self, ps: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        ps, ys = preprocess_score_inputs(ps, ys)
        return 1 - 0.5 * np.abs(ps - ys).sum(axis=1)
        # return 0.5 * np.abs(ps - ys).sum(axis=1)

    def expected_score(self, ps: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner.

        The absolute score has a different expression for the expected score:
            S(p; r) = <r, p>.
        """
        ps, rs = preprocess_score_inputs(ps, rs)
        return (rs * ps).sum(axis=1)
        # return 1 - (rs * ps).sum(axis=1)


class RelativeScore:
    """A generic object for relative scores w.r.t. a reference forecaster.

    Unlike :py:obj:ScoringRule:, it takes _two_ forecast arrays,
     in addition to the outcome array.
    Relative scores typically use a "base" scoring rule S.

    Attributes:
        is_proper: whether the scoring rule is proper
        base_score: a base scoring rule that is used to evaluate
            absolute forecasting skill (optional)
        bounds (a `@property`): lower and upper bounds of the score function
        name: name of the scoring rule
    Methods:
        __init__, score, expected_score
    """
    def __init__(
            self,
            is_proper: bool,
            base_score: Union[str, ScoringRule] = None,
            bounds: Tuple[float, float] = (-np.inf, np.inf),
            name: str = None,
    ):
        self.is_proper = is_proper
        self.base_score = get_scoring_rule(base_score)
        self._bounds = bounds
        self.name = f"SkillScore({base_score})" if name is None else name

    def __repr__(self):
        return self.name

    def __call__(self, ps: ArrayLike, qs: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Equivalent to `self.score()`.

        Child classes should override the `score` method.
        """
        return self.score(ps, qs, ys)

    @property
    def bounds(self) -> Tuple[float, float]:
        """Lower and upper bounds on the score.

        If unknown, set to `(-np.inf, np.inf)` by default."""
        return self._bounds

    @bounds.setter
    def bounds(self, value: Tuple[float, float]):
        self._bounds = value

    def score(self, ps: ArrayLike, qs: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        raise NotImplementedError

    def expected_score(self, ps: ArrayLike, qs: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner.

        Same as `self.score()` if scoring rule is linear in y.
        """
        raise NotImplementedError


class WinklerScore(RelativeScore):
    """A generalized form of Winkler (1994)'s normalized score for binary forecasts.

        w(p, q; y) = [S(p, y) - S(q, y)] / T(p, q)

    where S is a "base" scoring rule and

        T(p, q) = S(p, 1) - S(q, 1)  if p >= q
                  S(p, 0) - S(q, 0)  if p <  q.

    The score is a normalized variant of pointwise score differentials,
    and it can be useful when the magnitude of the base scores can vary significantly
    (e.g., for the logarithmic score).
    The Winkler score is (strictly) proper if the base score if (strictly) proper.

    A lower bound is 1 - 2/q_0, where q_0 is how close q can get to either 0 or 1.
    """
    def __init__(
            self,
            base_score: Union[str, ScoringRule] = "logarithmic",
            bounds: Tuple[float, float] = (1 - 2 / 1e-8, 1),
    ):
        base_score = get_scoring_rule(base_score)
        super().__init__(
            is_proper=base_score.is_proper,
            base_score=base_score,
            bounds=bounds,
            name="WinklerScore({})".format(base_score.name.split("Score")[0]),
        )

    def score(self, ps: ArrayLike, qs: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Compute pointwise scores in a vectorized manner."""
        assert len(ps.shape) <= 2, \
            "Winkler's score is only defined for binary outcomes/forecasts"
        assert np.logical_and(0 < qs, qs < 1).all()

        T = len(ps)
        zeros, ones = np.zeros(T), np.ones(T)
        if len(ps.shape) == 2:
            zeros = convert_to_onehot(zeros, 2)
            ones = convert_to_onehot(ones, 2)

        difference = self.base_score(ps, ys) - self.base_score(qs, ys)
        normalizer = np.where(
            ps >= qs,
            self.base_score(ps, ones) - self.base_score(qs, ones),
            self.base_score(ps, zeros) - self.base_score(qs, zeros),
        )
        # difference is also zero when normalizer is zero
        normalizer = np.where(normalizer != 0, normalizer, 1e-8)
        return difference / normalizer

    def expected_score(self, ps: ArrayLike, qs: ArrayLike, rs: ArrayLike) -> np.ndarray:
        """Compute pointwise *expected* scores in a vectorized manner."""
        return self.score(ps, qs, rs)


"""
Functional API
"""

SCORING_RULES = {
    "brier": BrierScore,
    "quadratic": BrierScore,
    "logarithmic": LogarithmicScore,
    "spherical": SphericalScore,
    "zero_one": ZeroOneScore,
    "absolute": AbsoluteScore,
    "winkler": WinklerScore,
}


def get_scoring_rule(name: Union[ScoringRule, str], **kwargs) -> Union[ScoringRule, Callable]:
    """Return a scoring rule as a function given its name and optional keyword arguments."""
    if isinstance(name, ScoringRule) or isinstance(name, RelativeScore):
        return name
    try:
        return SCORING_RULES[name](**kwargs)
    except KeyError:
        raise KeyError(f"invalid scoring rule {name}, try: " + ", ".join(SCORING_RULES.keys())) from None


def get_expected_scoring_rule(name: str, **kwargs) -> Callable:
    """Return the expected scoring rule function of a given scoring rule.

    (For backwards compatibility.)
    """
    print("WARNING: get_expected_scoring_rule() is deprecated."
          "use ScoringRule.expected_score() instead.")
    if name in ["brier", "quadratic", "absolute"]:
        score_obj = get_scoring_rule(name, **kwargs)

        def expected_score(*args):
            return score_obj.expected_score(*args)

        return expected_score
    else:
        return get_scoring_rule(name, **kwargs)
