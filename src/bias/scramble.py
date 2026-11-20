"""Trains the models on scrambled CIFAR-100.

One fixed scramble serves every image and is applied after augmentation:

- "pixels": the 1,024 pixel positions are reordered.
- "patches": the 64 patches of 4x4 pixels change places, and the 16
  positions inside every patch are reordered the same way.
- "local<b>": pixels are reordered inside every b x b block, each block
  its own way.
- "jigsaw<b>": the b x b blocks change places with their insides intact.

Usage:
    python -m bias.scramble configs/images.yaml
"""

import argparse
import dataclasses
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from bias import config as config_lib
from bias import grid, train
from bias.data import tensors
from bias.models import vit

SIDE, PATCH = vit.IMG_SIZE, vit.PATCH_SIZE
SIZE, SEED = "10%", 0
_LOCAL = ("local2", "local4", "local8", "local16")
# The scrambles each model is trained on, with augmentation.
RUNS = {
    "mlp": ("pixels", "patches"),
    "vit": ("pixels", "patches", *_LOCAL, "jigsaw2"),
    "cnn": ("pixels", "patches", *_LOCAL, "jigsaw2", "jigsaw4", "jigsaw8", "jigsaw16"),
}


def permutation(kind: str, seed: int = 0) -> npt.NDArray[np.int64]:
    """Returns the source position of every pixel in the scrambled image.

    Args:
        kind: "pixels", "patches", "local<b>" or "jigsaw<b>".
        seed: Seed of the scramble.

    Returns:
        An array of SIDE * SIDE indices into the flattened image:
        scrambled.flat[i] = image.flat[result[i]].

    Raises:
        ValueError: If the kind is unknown or b does not divide SIDE.
    """
    rng = np.random.default_rng(seed)
    if kind == "pixels":
        return rng.permutation(SIDE * SIDE)
    match = re.fullmatch(r"(local|jigsaw)([1-9]\d*)|patches", kind)
    if not match:
        raise ValueError(f"unknown scramble: {kind=}")
    family, block = match[1] or "patches", int(match[2] or PATCH)
    if SIDE % block:
        raise ValueError(f"block side must divide {SIDE}: {kind=}")
    per_side = SIDE // block
    grid = np.arange(SIDE * SIDE).reshape(per_side, block, per_side, block)
    # blocks[b, i] is the flat position of pixel i of block b.
    blocks = grid.transpose(0, 2, 1, 3).reshape(per_side**2, block**2)
    if family == "local":
        blocks = np.stack([b[rng.permutation(block**2)] for b in blocks])
    else:
        blocks = blocks[rng.permutation(per_side**2)]
    if family == "patches":
        blocks = blocks[:, rng.permutation(block**2)]
    source = blocks.reshape(per_side, per_side, block, block).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(source).reshape(-1)


def apply(x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Scrambles a batch of images.

    Args:
        x: Images of shape (batch, channels, SIDE, SIDE).
        source: Source positions from permutation(), on x's device.

    Returns:
        The scrambled images, same shape.
    """
    return x.flatten(2)[:, :, source].reshape(x.shape)


def scrambled(data: tensors.TensorData, kind: str) -> tensors.TensorData:
    """Returns the dataset with every split scrambled the same way.

    Args:
        data: CIFAR-100 tensors.
        kind: A scramble that permutation() knows.

    Returns:
        A copy whose model inputs are scrambled. Augmentation still sees
        the real image, and the scramble follows it.
    """
    source = torch.from_numpy(permutation(kind)).to(data.pool_x.device)
    prepare = data.prepare
    return dataclasses.replace(
        data,
        pool_x=apply(data.pool_x, source),
        val_x=apply(data.val_x, source),
        test_x=apply(data.test_x, source),
        prepare=lambda raw: apply(prepare(raw), source),
    )


def main() -> None:
    """Trains every model on its scrambles, skipping the runs already logged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = config_lib.load_config(parser.parse_args().config)
    settings = grid.best_settings(grid.read_runs(config, "tune"))
    device = train.pick_device()
    data = tensors.load_tensors(config, device)
    log = Path(config["results_dir"]) / f"{config['dataset']}_scramble.csv"
    for model, kinds in RUNS.items():
        tuned = grid.tuned(config, settings, model, "aug", SIZE)
        job = train.Job(model, "aug", SIZE, SEED, *tuned)
        for kind in kinds:
            stage = f"scramble_{kind}"
            if grid.unlogged(config, stage, [job]):
                grid.run_jobs(config, scrambled(data, kind), [job], stage, log, device)


if __name__ == "__main__":
    main()
