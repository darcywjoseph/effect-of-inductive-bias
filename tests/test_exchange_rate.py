import math

import numpy as np
import pytest

from bias_worth import exchange_rate as er

SIZES = np.array([100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)


def curve(sizes: np.ndarray, shift: float) -> np.ndarray:
    """A curve linear in log2(n), shifted right by log2(shift)."""
    result: np.ndarray = 0.1 * (np.log2(sizes) - np.log2(shift))
    return result


def test_isotonic_pools_violators() -> None:
    np.testing.assert_allclose(er.isotonic([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
    np.testing.assert_allclose(er.isotonic([3, 2, 1]), [2, 2, 2])
    np.testing.assert_allclose(er.isotonic([1, 2, 3]), [1, 2, 3])


def test_known_exchange_rate_is_recovered() -> None:
    # The reference needs 8x the data for any score on a log-linear curve.
    ref = curve(SIZES, shift=8.0)
    treat = curve(SIZES, shift=1.0)
    target = float(curve(np.array([500.0]), shift=1.0)[0])
    est = er.exchange_rate(SIZES, ref, SIZES, treat, target)
    assert est.kind == "exact"
    assert est.value == pytest.approx(8.0)


def test_crossing_on_grid_point_and_between() -> None:
    assert er.data_needed(SIZES, curve(SIZES, 1.0), float(curve(SIZES, 1.0)[3])).n == (
        pytest.approx(1000.0)
    )
    mid = (curve(SIZES, 1.0)[3] + curve(SIZES, 1.0)[4]) / 2
    assert er.data_needed(SIZES, curve(SIZES, 1.0), float(mid)).n == pytest.approx(
        math.sqrt(1000 * 2000)
    )


def test_non_monotone_curve_uses_first_crossing_of_isotonic_fit() -> None:
    scores = np.array([0.1, 0.5, 0.3, 0.6, 0.7, 0.8, 0.9])
    # Isotonic fit pools 0.5 and 0.3 to 0.4, so 0.45 is crossed between
    # index 2 (0.4) and index 3 (0.6), not at index 1.
    crossing = er.data_needed(SIZES, scores, 0.45)
    assert crossing.censor == "none"
    assert 500 < crossing.n < 1000


def test_reference_never_reaching_target_gives_lower_bound() -> None:
    ref = np.full(len(SIZES), 0.2)
    treat = np.linspace(0.1, 0.9, len(SIZES))
    est = er.exchange_rate(SIZES, ref, SIZES, treat, 0.5)
    assert est.kind == "lower"
    assert est.value == pytest.approx(10000 / er.data_needed(SIZES, treat, 0.5).n)


def test_treatment_never_reaching_target_gives_upper_bound() -> None:
    ref = np.linspace(0.1, 0.9, len(SIZES))
    est = er.exchange_rate(SIZES, ref, SIZES, np.full(len(SIZES), 0.2), 0.5)
    assert est.kind == "upper"


def test_treatment_above_target_at_smallest_size_gives_lower_bound() -> None:
    ref = np.linspace(0.1, 0.9, len(SIZES))
    est = er.exchange_rate(SIZES, ref, SIZES, np.full(len(SIZES), 0.95), 0.5)
    assert est.kind == "lower"


def test_neither_reaching_target_is_undefined() -> None:
    flat = np.full(len(SIZES), 0.2)
    assert er.exchange_rate(SIZES, flat, SIZES, flat, 0.5).kind == "undefined"


def test_bad_grid_raises() -> None:
    with pytest.raises(ValueError):
        er.data_needed([10, 5], [0.1, 0.2], 0.15)


def test_bootstrap_interval_covers_truth_under_noise() -> None:
    rng = np.random.default_rng(1)
    ref = curve(SIZES, 8.0) + rng.normal(0, 0.01, (3, len(SIZES)))
    treat = curve(SIZES, 1.0) + rng.normal(0, 0.01, (3, len(SIZES)))
    target = float(curve(np.array([500.0]), 1.0)[0])
    interval = er.bootstrap_interval(SIZES, ref, SIZES, treat, target, draws=2000)
    assert interval.low < 8.0 < interval.high
    assert interval.share_undefined == 0.0


def test_bootstrap_without_noise_collapses_to_point() -> None:
    ref = np.tile(curve(SIZES, 4.0), (3, 1))
    treat = np.tile(curve(SIZES, 1.0), (3, 1))
    target = float(curve(np.array([500.0]), 1.0)[0])
    interval = er.bootstrap_interval(SIZES, ref, SIZES, treat, target, draws=200)
    assert interval.low == pytest.approx(4.0)
    assert interval.high == pytest.approx(4.0)
