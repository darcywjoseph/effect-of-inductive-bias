"""CIFAR-100 as model-ready tensors, and how to make more of them.

The labelled pool is kept twice: as model inputs on the device, for the
no-augmentation path, and raw on the CPU, for augmentation and LeJEPA
views. A raw batch becomes model input through prepare.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from bias import augment
from bias.data import cifar

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)

BatchFn = Callable[[torch.Tensor], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class TensorData:
    """The dataset's tensors and its augmentation.

    Attributes:
        pool_x: Labelled pool as model inputs, on the device.
        pool_y: Labelled pool labels, on the device.
        val_x: Validation inputs.
        val_y: Validation labels.
        test_x: Test inputs.
        test_y: Test labels.
        num_classes: Number of classes.
        pool_raw: Labelled pool before prepare, on the CPU.
        prepare: Turns a raw CPU batch into model inputs on the device.
        augment_example: Per-example augmentation of a raw example, run by
          CPU workers.
        ssl_view: Per-example transform that makes one LeJEPA view.
    """

    pool_x: torch.Tensor
    pool_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int
    pool_raw: torch.Tensor
    prepare: BatchFn
    augment_example: augment.Transform
    ssl_view: augment.Transform


def _in_chunks(prepare: BatchFn, raw: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([prepare(raw[i : i + chunk]) for i in range(0, len(raw), chunk)])


class _ImagePrepare:
    def __init__(self, device: torch.device) -> None:
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.to(self.mean.device, non_blocking=True).float().div(255.0)
        return (x - self.mean) / self.std


def load_tensors(config: dict[str, Any], device: torch.device) -> TensorData:
    """Loads CIFAR-100 onto the device with its augmentation.

    Args:
        config: Parsed config.
        device: Where the model inputs live.

    Returns:
        The dataset.
    """
    data = config["data"]
    splits = cifar.load(
        Path(config["data_root"]), data["val_per_class"], data["split_seed"]
    )

    def nchw(x: npt.NDArray[np.uint8]) -> torch.Tensor:
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()

    prepare = _ImagePrepare(device)
    pool_raw = nchw(splits.pool_x)
    return TensorData(
        pool_x=_in_chunks(prepare, pool_raw),
        pool_y=torch.from_numpy(splits.pool_y).to(device),
        val_x=prepare(nchw(splits.val_x)),
        val_y=torch.from_numpy(splits.val_y).to(device),
        test_x=prepare(nchw(splits.test_x)),
        test_y=torch.from_numpy(splits.test_y).to(device),
        num_classes=100,
        pool_raw=pool_raw,
        prepare=prepare,
        augment_example=augment.cifar_transform(),
        ssl_view=augment.cifar_ssl_view(),
    )
