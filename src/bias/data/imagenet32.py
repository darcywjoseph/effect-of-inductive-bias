"""ImageNet-32 (ImageNet-1k at 32x32) with a fixed validation hold-out.

Reads the cache written by scripts/imagenet32_to_npz.py. The official
ImageNet validation set (50 per class) has labels and serves as the test
set; validation images are held out of the training set instead.
"""

from pathlib import Path

import numpy as np

from bias.data.cifar import Splits

# The usual ImageNet channel statistics.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load(root: Path, val_per_class: int = 10, split_seed: int = 0) -> Splits:
    """Loads ImageNet-32 and holds out a stratified validation set.

    Args:
        root: Data root holding imagenet32/imagenet32.npz.
        val_per_class: Training images per class moved to validation.
        split_seed: Seed for the hold-out.

    Returns:
        The pool, validation and test splits.
    """
    cache = np.load(root / "imagenet32" / "imagenet32.npz")
    train_x, train_y = cache["train_x"], cache["train_y"]
    rng = np.random.default_rng(split_seed)
    val_mask = np.zeros(len(train_y), dtype=bool)
    for cls in np.unique(train_y):
        members = np.flatnonzero(train_y == cls)
        val_mask[rng.choice(members, val_per_class, replace=False)] = True
    return Splits(
        pool_x=train_x[~val_mask],
        pool_y=train_y[~val_mask],
        val_x=train_x[val_mask],
        val_y=train_y[val_mask],
        test_x=cache["val_x"],
        test_y=cache["val_y"],
    )
