"""CIFAR-100 or ImageNet-32 as model-ready tensors, and how to make more.

The labelled pool is kept raw on the CPU, for augmentation and LeJEPA
views, and for CIFAR-100 also as model inputs on the device, for the
no-augmentation path. ImageNet-32 is too big for that copy, so its pool_x
is the raw CPU pool and only augmented training works on it. A raw batch
becomes model input through prepare.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from bias import augment
from bias.data import cifar, imagenet32

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)

BatchFn = Callable[[torch.Tensor], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class TensorData:
    """The dataset's tensors and its augmentation.

    Attributes:
        pool_x: Labelled pool as model inputs, on the device (ImageNet-32:
          the raw pool, on the CPU).
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
    def __init__(
        self,
        device: torch.device,
        mean: tuple[float, ...] = CIFAR_MEAN,
        std: tuple[float, ...] = CIFAR_STD,
    ) -> None:
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.to(self.mean.device, non_blocking=True).float().div(255.0)
        return (x - self.mean) / self.std


def load_tensors(config: dict[str, Any], device: torch.device) -> TensorData:
    """Loads the config's dataset onto the device with its augmentation.

    Args:
        config: Parsed config; its dataset is "cifar100" or "imagenet32".
        device: Where the model inputs live.

    Returns:
        The dataset.
    """
    data = config["data"]
    big = config["dataset"] == "imagenet32"
    loader = imagenet32.load if big else cifar.load
    splits = loader(
        Path(config["data_root"]), data["val_per_class"], data["split_seed"]
    )

    def nchw(x: npt.NDArray[np.uint8]) -> torch.Tensor:
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()

    prepare = (
        _ImagePrepare(device, imagenet32.MEAN, imagenet32.STD)
        if big
        else _ImagePrepare(device)
    )
    pool_raw = nchw(splits.pool_x)
    return TensorData(
        pool_x=pool_raw if big else _in_chunks(prepare, pool_raw),
        pool_y=torch.from_numpy(splits.pool_y).to(device),
        val_x=prepare(nchw(splits.val_x)),
        val_y=torch.from_numpy(splits.val_y).to(device),
        test_x=prepare(nchw(splits.test_x)),
        test_y=torch.from_numpy(splits.test_y).to(device),
        num_classes=int(splits.pool_y.max()) + 1,
        pool_raw=pool_raw,
        prepare=prepare,
        augment_example=augment.cifar_transform(),
        ssl_view=augment.cifar_ssl_view(),
    )
