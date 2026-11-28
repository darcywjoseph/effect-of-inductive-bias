from pathlib import Path

import numpy as np
import torch

from bias.data import imagenet32
from bias.models import registry


def test_imagenet32_holds_out_validation_per_class(tmp_path: Path) -> None:
    (tmp_path / "imagenet32").mkdir()
    train_y = np.repeat(np.arange(3), 20)
    np.savez(
        tmp_path / "imagenet32" / "imagenet32.npz",
        train_x=np.zeros((60, 32, 32, 3), np.uint8),
        train_y=train_y,
        val_x=np.zeros((6, 32, 32, 3), np.uint8),
        val_y=np.repeat(np.arange(3), 2),
    )
    splits = imagenet32.load(tmp_path, val_per_class=5)
    assert np.bincount(splits.val_y).tolist() == [5, 5, 5]
    assert np.bincount(splits.pool_y).tolist() == [15, 15, 15]
    assert len(splits.test_y) == 6


def test_models_take_a_class_count() -> None:
    for name in ("vit", "vit_patch8", "group_grid"):
        model = registry.build(name, num_classes=1000).eval()
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 1000)
