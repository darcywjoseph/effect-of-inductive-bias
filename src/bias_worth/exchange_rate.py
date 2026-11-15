"""Exchange rates read off learning curves.

The rule, fixed before any final run:

1. Average the test score over seeds at each data size.
2. Make the curve non-decreasing with isotonic regression.
3. Interpolate linearly between grid points against log2 of the data size.
4. N(s) is where the curve first reaches the target s.

E(s) = N_reference(s) / N_treatment(s). A curve that never reaches s, or
already exceeds it at the smallest size, gives a censored N, and E is then
reported as a bound. Nothing is extrapolated beyond the grid.
"""

import dataclasses
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Censor = Literal["none", "left", "right"]
Kind = Literal["exact", "lower", "upper", "undefined"]


@dataclasses.dataclass(frozen=True)
class Crossing:
    """The data a curve needs to reach a target.

    Attributes:
        n: Labelled examples needed. For a censored crossing this is the
          grid end that bounds the true value.
        censor: "none" if the curve crosses inside the grid, "left" if it
          already meets the target at the smallest size (true N <= n), and
          "right" if it never reaches the target (true N > n).
    """

    n: float
    censor: Censor


@dataclasses.dataclass(frozen=True)
class Estimate:
    """An exchange rate, possibly only a bound.

    Attributes:
        value: The ratio, or the bound when kind is not "exact".
        kind: "exact", "lower" (true E > value), "upper" (true E < value),
          or "undefined" when the censoring leaves no bound.
    """

    value: float
    kind: Kind


def isotonic(values: npt.ArrayLike) -> FloatArray:
    """Returns the least-squares non-decreasing fit (pool adjacent violators).

    Args:
        values: Scores in order of increasing data size.

    Returns:
        The fitted, non-decreasing scores.
    """
    blocks: list[list[float]] = []  # Each block is [mean, count].
    for value in np.asarray(values, dtype=np.float64):
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            mean_b, count_b = blocks.pop()
            mean_a, count_a = blocks[-1]
            total = count_a + count_b
            blocks[-1] = [(mean_a * count_a + mean_b * count_b) / total, total]
    return np.concatenate([np.full(int(count), mean) for mean, count in blocks])


def data_needed(sizes: npt.ArrayLike, scores: npt.ArrayLike, target: float) -> Crossing:
    """Reads N(target) off one seed-averaged learning curve.

    Args:
        sizes: Labelled examples at each grid point, strictly increasing.
        scores: Seed-averaged score at each grid point.
        target: The score to reach.

    Returns:
        Where the isotonic curve first reaches the target.

    Raises:
        ValueError: If sizes are not strictly increasing and positive, or
          the arrays differ in length.
    """
    x = np.asarray(sizes, dtype=np.float64)
    y = isotonic(scores)
    if x.shape != y.shape or x.ndim != 1 or not x.size:
        raise ValueError(f"sizes and scores must be equal-length 1-D: {x.shape=}")
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError(f"sizes must be positive and increasing: {x=}")
    reached = np.flatnonzero(y >= target)
    if not reached.size:
        return Crossing(n=float(x[-1]), censor="right")
    i = int(reached[0])
    if i == 0:
        censor: Censor = "none" if y[0] == target else "left"
        return Crossing(n=float(x[0]), censor=censor)
    t = (target - y[i - 1]) / (y[i] - y[i - 1])
    log_n = np.log2(x[i - 1]) + t * (np.log2(x[i]) - np.log2(x[i - 1]))
    return Crossing(n=float(2.0**log_n), censor="none")


def ratio(reference: Crossing, treatment: Crossing) -> Estimate:
    """Combines two crossings into E = N_reference / N_treatment.

    Args:
        reference: Crossing for the reference setup.
        treatment: Crossing for the treatment setup.

    Returns:
        The exchange rate, or the bound the censoring allows.
    """
    value = reference.n / treatment.n
    # A reference that never gets there, or a treatment already there at
    # the smallest size, can only make E larger than the computed ratio.
    pushes_up = reference.censor == "right" or treatment.censor == "left"
    pushes_down = reference.censor == "left" or treatment.censor == "right"
    if pushes_up and pushes_down:
        return Estimate(value=float("nan"), kind="undefined")
    if pushes_up:
        return Estimate(value=value, kind="lower")
    if pushes_down:
        return Estimate(value=value, kind="upper")
    return Estimate(value=value, kind="exact")


def exchange_rate(
    ref_sizes: npt.ArrayLike,
    ref_scores: npt.ArrayLike,
    treat_sizes: npt.ArrayLike,
    treat_scores: npt.ArrayLike,
    target: float,
) -> Estimate:
    """Computes E(target) from two seed-averaged curves.

    Args:
        ref_sizes: Grid sizes of the reference curve.
        ref_scores: Seed-averaged reference scores.
        treat_sizes: Grid sizes of the treatment curve.
        treat_scores: Seed-averaged treatment scores.
        target: The score both setups must reach.

    Returns:
        The exchange rate estimate.
    """
    return ratio(
        data_needed(ref_sizes, ref_scores, target),
        data_needed(treat_sizes, treat_scores, target),
    )


@dataclasses.dataclass(frozen=True)
class Interval:
    """A bootstrap percentile interval for E.

    Attributes:
        low: 2.5th percentile. Zero means upper-bounded draws reached it.
        high: 97.5th percentile. Infinity means lower-bounded draws reached it.
        share_censored: Share of draws that were a bound, not a value.
        share_undefined: Share of draws with no usable bound.
    """

    low: float
    high: float
    share_censored: float
    share_undefined: float


def bootstrap_interval(
    ref_sizes: npt.ArrayLike,
    ref_seed_scores: npt.ArrayLike,
    treat_sizes: npt.ArrayLike,
    treat_seed_scores: npt.ArrayLike,
    target: float,
    draws: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Resamples seeds within each setup and returns a percentile interval.

    A seed's whole curve is one unit, because the seed fixes both the
    nested subsets and the initialisation. Draws that come out as a lower
    bound count as +inf and upper bounds as 0, which widens the interval in
    the honest direction. Undefined draws are dropped and counted.

    Args:
        ref_sizes: Grid sizes of the reference curve.
        ref_seed_scores: Reference scores, shape (seeds, sizes).
        treat_sizes: Grid sizes of the treatment curve.
        treat_seed_scores: Treatment scores, shape (seeds, sizes).
        target: The score both setups must reach.
        draws: Number of bootstrap draws.
        seed: Seed for the resampling.
        level: Coverage of the interval.

    Returns:
        The interval and how much of it rests on censored draws.
    """
    ref = np.asarray(ref_seed_scores, dtype=np.float64)
    treat = np.asarray(treat_seed_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    censored = 0
    for d in range(draws):
        ref_mean = ref[rng.integers(0, len(ref), len(ref))].mean(axis=0)
        treat_mean = treat[rng.integers(0, len(treat), len(treat))].mean(axis=0)
        est = exchange_rate(ref_sizes, ref_mean, treat_sizes, treat_mean, target)
        censored += est.kind in ("lower", "upper")
        values[d] = {
            "exact": est.value,
            "lower": np.inf,
            "upper": 0.0,
            "undefined": np.nan,
        }[est.kind]
    defined = values[~np.isnan(values)]
    tail = (1.0 - level) / 2.0
    if defined.size:
        low, high = np.quantile(defined, [tail, 1.0 - tail], method="inverted_cdf")
    else:
        low, high = np.nan, np.nan
    return Interval(
        low=float(low),
        high=float(high),
        share_censored=censored / draws,
        share_undefined=(draws - defined.size) / draws,
    )
