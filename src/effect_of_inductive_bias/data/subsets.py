"""Nested, per-class labelled subsets.

Each class gets one random ordering per seed. A subset of any size takes a
prefix of every class's ordering, so for a fixed seed each smaller subset
is contained in every larger one.
"""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclasses.dataclass(frozen=True)
class Subset:
    """A labelled subset drawn from a pool.

    Attributes:
        indices: Pool indices in the subset, sorted ascending.
        per_class: Number of examples taken from each class.
        capped_classes: Classes that had fewer examples than requested.
    """

    indices: IntArray
    per_class: dict[int, int]
    capped_classes: list[int]


def class_orderings(labels: npt.ArrayLike, seed: int) -> dict[int, IntArray]:
    """Returns one random ordering of pool indices per class.

    Args:
        labels: Class label of every pool example.
        seed: Seed for the orderings. The same seed always gives the same
          orderings, whatever subset sizes are drawn later.

    Returns:
        A dict mapping each class to its pool indices in random order.
    """
    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)
    orderings: dict[int, IntArray] = {}
    for cls in np.unique(labels_arr):
        members = np.flatnonzero(labels_arr == cls).astype(np.int64)
        orderings[int(cls)] = rng.permutation(members)
    return orderings


def _take(orderings: dict[int, IntArray], counts: dict[int, int]) -> Subset:
    picked: list[IntArray] = []
    per_class: dict[int, int] = {}
    capped: list[int] = []
    for cls, order in orderings.items():
        want = counts[cls]
        if want > len(order):
            capped.append(cls)
        got = min(want, len(order))
        picked.append(order[:got])
        per_class[cls] = got
    indices = np.sort(np.concatenate(picked)) if picked else np.empty(0, np.int64)
    return Subset(indices=indices, per_class=per_class, capped_classes=capped)


def fraction_subset(labels: npt.ArrayLike, fraction: float, seed: int) -> Subset:
    """Draws a per-class fraction of the pool.

    Each class contributes round(fraction * class size) examples, with a
    minimum of one, so class proportions match the pool.

    Args:
        labels: Class label of every pool example.
        fraction: Share of each class to keep, in (0, 1].
        seed: Seed for the nested ordering.

    Returns:
        The subset.

    Raises:
        ValueError: If fraction is outside (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]: {fraction=}")
    orderings = class_orderings(labels, seed)
    counts = {
        cls: max(1, math.floor(fraction * len(order) + 0.5))
        for cls, order in orderings.items()
    }
    return _take(orderings, counts)


def per_class_subset(labels: npt.ArrayLike, count: int, seed: int) -> Subset:
    """Draws a fixed number of examples from every class.

    Classes with fewer than count examples contribute all they have and are
    listed in Subset.capped_classes.

    Args:
        labels: Class label of every pool example.
        count: Examples wanted per class.
        seed: Seed for the nested ordering.

    Returns:
        The subset.
    """
    orderings = class_orderings(labels, seed)
    return _take(orderings, dict.fromkeys(orderings, count))


def draw_subset(labels: Sequence[int] | npt.ArrayLike, size: str, seed: int) -> Subset:
    """Draws a subset from a grid entry written as a string.

    Args:
        labels: Class label of every pool example.
        size: "10%" for a per-class fraction, "20pc" for 20 per class, or
          "all" for the whole pool.
        seed: Seed for the nested ordering.

    Returns:
        The subset.

    Raises:
        ValueError: If size is not in one of the three forms.
    """
    if size == "all":
        return fraction_subset(labels, 1.0, seed)
    if size.endswith("%"):
        return fraction_subset(labels, float(size[:-1]) / 100.0, seed)
    if size.endswith("pc"):
        return per_class_subset(labels, int(size[:-2]), seed)
    raise ValueError(f"unknown subset size: {size=}")
