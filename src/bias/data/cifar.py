"""CIFAR-100 with a fixed validation hold-out."""

import dataclasses
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torchvision

UInt8Array = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int64]


@dataclasses.dataclass(frozen=True)
class Splits:
    """Images (N, 32, 32, 3) uint8 and labels for each split.

    Attributes:
        pool_x: Labelled pool images; subsets are drawn from here.
        pool_y: Labelled pool labels.
        val_x: Validation images, outside the label budget.
        val_y: Validation labels.
        test_x: Official test images.
        test_y: Official test labels.
    """

    pool_x: UInt8Array
    pool_y: IntArray
    val_x: UInt8Array
    val_y: IntArray
    test_x: UInt8Array
    test_y: IntArray


def load(root: Path, val_per_class: int = 50, split_seed: int = 0) -> Splits:
    """Loads CIFAR-100 and holds out a stratified validation set.

    Args:
        root: Data root that torchvision downloaded into.
        val_per_class: Training images per class moved to validation.
        split_seed: Seed for the hold-out.

    Returns:
        The pool, validation and test splits.
    """
    train = torchvision.datasets.CIFAR100(str(root), train=True)
    test = torchvision.datasets.CIFAR100(str(root), train=False)
    train_y = np.asarray(train.targets, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    val_mask = np.zeros(len(train_y), dtype=bool)
    for cls in np.unique(train_y):
        members = np.flatnonzero(train_y == cls)
        val_mask[rng.choice(members, val_per_class, replace=False)] = True
    return Splits(
        pool_x=train.data[~val_mask],
        pool_y=train_y[~val_mask],
        val_x=train.data[val_mask],
        val_y=train_y[val_mask],
        test_x=test.data,
        test_y=np.asarray(test.targets, dtype=np.int64),
    )
