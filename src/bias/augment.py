"""Standard augmentation, applied per example by CPU workers.

Images get a random crop with 4 pixels of reflect padding, a horizontal
flip and RandAugment (2 ops, magnitude 9), the usual CIFAR recipe.
"""

from collections.abc import Callable, Iterator

import torch
import torch.utils.data
from torchvision.transforms import v2

Transform = Callable[[torch.Tensor], torch.Tensor]


def cifar_transform() -> Transform:
    """Returns the CIFAR augmentation for one uint8 (3, 32, 32) image.

    Returns:
        A transform from uint8 image to uint8 image.
    """
    transform: Transform = v2.Compose(
        [
            v2.RandomCrop(32, padding=4, padding_mode="reflect"),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=9),
        ]
    )
    return transform


def cifar_ssl_view() -> Transform:
    """Returns the LeJEPA view transform for one uint8 (3, 32, 32) image.

    A 32-pixel random resized crop, a flip, colour jitter, greyscale and
    solarisation, without Gaussian blur.

    Returns:
        A transform from uint8 image to uint8 image.
    """
    transform: Transform = v2.Compose(
        [
            v2.RandomResizedCrop(32, scale=(0.08, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomSolarize(threshold=128, p=0.2),
        ]
    )
    return transform


class _Augmented(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform: Transform) -> None:
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.x[i]), self.y[i]


class _EpochSampler(torch.utils.data.Sampler[int]):
    """Yields whole shuffled epochs until num_samples indices are out."""

    def __init__(self, n: int, num_samples: int, seed: int) -> None:
        self.n, self.num_samples, self.seed = n, num_samples, seed

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed)
        given = 0
        while given < self.num_samples:
            for i in torch.randperm(self.n, generator=generator).tolist():
                if given == self.num_samples:
                    return
                yield i
                given += 1


def augmented_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    transform: Transform,
    batch: int,
    steps: int,
    seed: int,
    workers: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields augmented batches for a fixed number of steps.

    Batches walk through shuffled epochs of the subset, as in the
    no-augmentation path, and every example is augmented afresh each time
    it is drawn.

    Args:
        x: Subset inputs on the CPU, in the dtype the transform expects.
        y: Subset labels on the CPU.
        transform: Per-example augmentation.
        batch: Batch size.
        steps: Number of batches to yield.
        seed: Seed for the order and for the workers' random draws.
        workers: CPU worker processes; 0 augments in the main process.

    Yields:
        (inputs, labels) batches, still on the CPU.
    """
    loader = torch.utils.data.DataLoader(
        _Augmented(x, y, transform),
        batch_size=batch,
        sampler=_EpochSampler(len(x), steps * batch, seed),
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        prefetch_factor=4 if workers > 0 else None,
    )
    yield from loader
