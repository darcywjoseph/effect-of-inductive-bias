"""Model-ready tensors for every dataset, and how to make more of them.

Each dataset keeps its labelled pool twice: as model inputs on the device,
for the no-augmentation path, and raw on the CPU, for augmentation and
LeJEPA views. A raw batch becomes model input through prepare, and the
dataset says which per-example and per-batch augmentation it uses.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from bias import augment
from bias.data import cifar, hsi, speech_commands

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)

BatchFn = Callable[[torch.Tensor], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class TensorData:
    """One dataset's tensors and its augmentation.

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
        augment_batch: Augmentation of a prepared batch on the device; the
          identity for datasets that have none.
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
    augment_batch: BatchFn
    ssl_view: augment.Transform


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _in_chunks(prepare: BatchFn, raw: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([prepare(raw[i : i + chunk]) for i in range(0, len(raw), chunk)])


class _ImagePrepare:
    def __init__(self, device: torch.device) -> None:
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = raw.to(self.mean.device, non_blocking=True).float().div(255.0)
        return (x - self.mean) / self.std


def _cifar(config: dict[str, Any], device: torch.device) -> TensorData:
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
        augment_batch=_identity,
        ssl_view=augment.cifar_ssl_view(),
    )


class _AudioPrepare:
    """Waveforms to normalised log-mels cut to a whole number of patches."""

    def __init__(
        self, data: dict[str, Any], frames: int, device: torch.device
    ) -> None:
        self.logmel = speech_commands.LogMel(
            data["n_mels"], data["win_ms"], data["hop_ms"]
        ).to(device)
        self.frames = frames
        self.device = device
        self.mean, self.std = 0.0, 1.0

    def fit(self, raw: torch.Tensor) -> None:
        """Sets the scalar mean and std from a sample of raw pool clips.

        Args:
            raw: int16 waveforms from the labelled pool.
        """
        x = _in_chunks(self, raw)
        self.mean, self.std = float(x.mean()), float(x.std())

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        waves = raw.to(self.device, non_blocking=True).float()
        if raw.dtype == torch.int16:
            waves = waves / 32768.0
        x: torch.Tensor = self.logmel(waves)[..., : self.frames]
        return (x - self.mean) / self.std


def _speech_commands(config: dict[str, Any], device: torch.device) -> TensorData:
    root = Path(config["data_root"])
    data = config["data"]
    splits = {s: speech_commands.load_split(root, s) for s in ("pool", "val", "test")}
    prepare = _AudioPrepare(data, data["frames"], device)
    pool_raw = torch.from_numpy(splits["pool"].waves)
    sample = torch.from_numpy(np.random.default_rng(0).permutation(len(pool_raw))[:4096])
    prepare.fit(pool_raw[sample])
    wave_augment = speech_commands.WaveAugment(
        speech_commands.background_noise(root), **data["wave_augment"]
    )

    def labels(split: str) -> torch.Tensor:
        return torch.from_numpy(splits[split].labels).to(device)

    def inputs(split: str) -> torch.Tensor:
        return _in_chunks(prepare, torch.from_numpy(splits[split].waves))

    return TensorData(
        pool_x=_in_chunks(prepare, pool_raw),
        pool_y=labels("pool"),
        val_x=inputs("val"),
        val_y=labels("val"),
        test_x=inputs("test"),
        test_y=labels("test"),
        num_classes=len(speech_commands.class_names(root)),
        pool_raw=pool_raw,
        prepare=prepare,
        augment_example=wave_augment,
        augment_batch=speech_commands.SpecAugment(**data["spec_augment"]),
        ssl_view=wave_augment,
    )


class _ToDevice:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.to(self.device, non_blocking=True).float()


def _hsi(config: dict[str, Any], device: torch.device) -> TensorData:
    data = config["data"]
    scene = hsi.load_scene(Path(config["data_root"]), config["dataset"])
    if data.get("split", "block") == "random":
        # The leaky protocol, used only by the leakage stage.
        split = hsi.random_split(
            scene.labels, tuple(data["fractions"]), data["split_seed"]
        )
    else:
        split = hsi.block_split(
            scene.labels,
            data["block"],
            data["patch_radius"],
            tuple(data["fractions"]),
            data["split_seed"],
            data["tries"],
        ).split
    mean, std = hsi.band_stats(scene.cube, split)
    cube = (scene.cube - mean) / std

    def part(k: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = np.nonzero(split == k)
        patches = hsi.extract_patches(cube, rows, cols, data["patch_radius"])
        return torch.from_numpy(patches), torch.from_numpy(scene.labels[rows, cols])

    (pool_raw, pool_y), (val_x, val_y), (test_x, test_y) = (
        part(k) for k in (hsi.TRAIN, hsi.VAL, hsi.TEST)
    )
    prepare = _ToDevice(device)
    patch_augment = hsi.PatchAugment(data["noise_std"])
    return TensorData(
        pool_x=prepare(pool_raw),
        pool_y=pool_y.to(device),
        val_x=prepare(val_x),
        val_y=val_y.to(device),
        test_x=prepare(test_x),
        test_y=test_y.to(device),
        num_classes=int(scene.labels.max()) + 1,
        pool_raw=pool_raw,
        prepare=prepare,
        augment_example=patch_augment,
        augment_batch=_identity,
        ssl_view=patch_augment,
    )


LOADERS = {
    "cifar100": _cifar,
    "speech_commands_v2": _speech_commands,
    "salinas": _hsi,
    "pavia_university": _hsi,
}


def load_tensors(config: dict[str, Any], device: torch.device) -> TensorData:
    """Loads a dataset onto the device with its augmentation.

    Args:
        config: Parsed config.
        device: Where the model inputs live.

    Returns:
        The dataset.

    Raises:
        ValueError: If the dataset has no loader.
    """
    if config["dataset"] not in LOADERS:
        raise ValueError(f"no tensor loader for {config['dataset']=}")
    return LOADERS[config["dataset"]](config, device)
