import numpy as np
import pytest

from bias_worth.data import subsets

LABELS = np.repeat(np.arange(5), [100, 50, 20, 7, 3])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fraction_subsets_are_nested(seed: int) -> None:
    grid = ["1%", "2%", "5%", "10%", "20%", "50%", "100%"]
    drawn = [set(subsets.draw_subset(LABELS, size, seed).indices) for size in grid]
    for small, large in zip(drawn, drawn[1:], strict=False):
        assert small <= large
    assert drawn[-1] == set(range(len(LABELS)))


def test_per_class_subsets_are_nested_and_capped() -> None:
    small = subsets.draw_subset(LABELS, "5pc", 0)
    large = subsets.draw_subset(LABELS, "20pc", 0)
    assert set(small.indices) <= set(large.indices)
    assert large.per_class == {0: 20, 1: 20, 2: 20, 3: 7, 4: 3}
    assert large.capped_classes == [3, 4]


def test_fraction_keeps_at_least_one_per_class() -> None:
    sub = subsets.draw_subset(LABELS, "1%", 0)
    assert min(sub.per_class.values()) == 1


def test_seeds_differ() -> None:
    a = subsets.draw_subset(LABELS, "10%", 0).indices
    b = subsets.draw_subset(LABELS, "10%", 1).indices
    assert not np.array_equal(a, b)


def test_bad_size_raises() -> None:
    with pytest.raises(ValueError):
        subsets.draw_subset(LABELS, "ten", 0)
