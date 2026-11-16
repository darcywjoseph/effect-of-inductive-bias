"""Hyperspectral scenes and spatially disjoint block splits.

A scene is cut into square blocks and whole blocks go to train, validation
or test. A pixel is kept only if every pixel within one patch radius lies
in a block of the same split, so a kept pixel's patch never contains a
pixel of another split. This is the block-plus-buffer protocol of
arXiv:2606.06684 and arXiv:2609.01786.
"""

import dataclasses
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy.io
import scipy.ndimage

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAMES = ("train", "val", "test")

SCENES = {
    "salinas": (
        "Salinas_corrected.mat",
        "salinas_corrected",
        "Salinas_gt.mat",
        "salinas_gt",
    ),
    "pavia_university": ("PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt"),
}


@dataclasses.dataclass(frozen=True)
class Scene:
    """A hyperspectral image with its ground truth.

    Attributes:
        cube: Reflectance, shape (height, width, bands).
        labels: Class per pixel, shape (height, width); -1 means unlabelled
          and classes run from 0.
    """

    cube: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]


def load_scene(root: Path, name: str) -> Scene:
    """Loads a scene from the .mat files under root/hsi.

    Args:
        root: Data root holding the hsi directory.
        name: "salinas" or "pavia_university".

    Returns:
        The scene, with labels shifted so 0 is the first class.
    """
    cube_file, cube_key, gt_file, gt_key = SCENES[name]
    cube = scipy.io.loadmat(root / "hsi" / cube_file)[cube_key].astype(np.float32)
    gt = scipy.io.loadmat(root / "hsi" / gt_file)[gt_key].astype(np.int64)
    return Scene(cube=cube, labels=gt - 1)


@dataclasses.dataclass(frozen=True)
class BlockSplit:
    """A block assignment and the pixels it keeps.

    Attributes:
        block_split: Split of every pixel's block, shape (height, width).
        split: Split of every kept labelled pixel; -1 for unlabelled,
          buffered-out, or unassigned pixels.
        covered: Classes with kept pixels in all three splits.
        score: The worst ratio, over covered classes and splits, of a
          split's share of a class to its target share. 1.0 is perfect.
        kept: Share of labelled pixels that survive the buffer.
    """

    block_split: npt.NDArray[np.int8]
    split: npt.NDArray[np.int8]
    covered: int
    score: float
    kept: float

    @property
    def rank(self) -> tuple[int, float, float]:
        """The search's sort key: coverage first, then balance, then kept."""
        return (self.covered, self.score, self.kept)


def _apply_buffer(
    block_split: npt.NDArray[np.int8],
    labels: npt.NDArray[np.int64],
    radius: int,
) -> npt.NDArray[np.int8]:
    window = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    split = np.full(labels.shape, -1, dtype=np.int8)
    for k in (TRAIN, VAL, TEST):
        # Image edges are not another split, so pad with True.
        inside = scipy.ndimage.binary_erosion(
            block_split == k, structure=window, border_value=1
        )
        split[inside & (labels >= 0)] = k
    return split


def _coverage_and_balance(
    split: npt.NDArray[np.int8],
    labels: npt.NDArray[np.int64],
    fractions: tuple[float, float, float],
    classes: npt.NDArray[np.int64],
) -> tuple[int, float]:
    covered = 0
    worst = np.inf
    for cls in classes:
        total = np.count_nonzero(labels == cls)
        shares = [
            np.count_nonzero((split == k) & (labels == cls)) / total
            for k in (TRAIN, VAL, TEST)
        ]
        if min(shares) > 0:
            covered += 1
            worst = min(worst, *(s / t for s, t in zip(shares, fractions, strict=True)))
    return covered, float(worst) if covered else 0.0


def block_split(
    labels: npt.NDArray[np.int64],
    block: int,
    radius: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    tries: int = 2000,
) -> BlockSplit:
    """Searches random block assignments for the best class coverage.

    Each try deals the blocks that hold labelled pixels to the three splits
    in proportion to fractions (by labelled pixels) and applies the buffer.
    Tries are ranked by how many classes reach all three splits, then by
    how evenly those classes are shared out, then by pixels kept. The
    search sees labels only, never a model or a score.

    Args:
        labels: Class per pixel, -1 for unlabelled.
        block: Block side in pixels.
        radius: Patch radius; also the buffer width.
        fractions: Target share of labelled pixels for train, val and test.
        seed: Seed for the search. Fixed per scene in the config.
        tries: Random assignments to try.

    Returns:
        The best assignment found.
    """
    height, width = labels.shape
    rows, cols = -(-height // block), -(-width // block)
    block_id = (np.arange(height)[:, None] // block) * cols + (
        np.arange(width)[None, :] // block
    )
    counts = np.bincount(block_id[labels >= 0], minlength=rows * cols)
    labelled_blocks = np.flatnonzero(counts)
    n_labelled = int(counts.sum())
    classes = np.unique(labels[labels >= 0])
    edges = np.cumsum(fractions)[:-1] * counts.sum()
    rng = np.random.default_rng(seed)

    best: BlockSplit | None = None
    for _ in range(tries):
        order = rng.permutation(labelled_blocks)
        cumulative = np.cumsum(counts[order]) - counts[order] / 2
        assignment = np.full(rows * cols, TRAIN, dtype=np.int8)
        assignment[order] = np.searchsorted(edges, cumulative).astype(np.int8)
        block_map = assignment[block_id]
        split = _apply_buffer(block_map, labels, radius)
        covered, score = _coverage_and_balance(split, labels, fractions, classes)
        candidate = BlockSplit(
            block_split=block_map,
            split=split,
            covered=covered,
            score=score,
            kept=np.count_nonzero(split >= 0) / n_labelled,
        )
        if best is None or candidate.rank > best.rank:
            best = candidate
    if best is None:
        raise ValueError(f"tries must be positive: {tries=}")
    return best


def random_split(
    labels: npt.NDArray[np.int64],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> npt.NDArray[np.int8]:
    """Splits labelled pixels at random, per class, with no buffer.

    This is the common but leaky protocol, kept only for the leakage
    comparison.

    Args:
        labels: Class per pixel, -1 for unlabelled.
        fractions: Share of each class for train, val and test.
        seed: Seed for the shuffle.

    Returns:
        Split of every labelled pixel, -1 elsewhere.
    """
    rng = np.random.default_rng(seed)
    split = np.full(labels.shape, -1, dtype=np.int8)
    flat = split.reshape(-1)
    flat_labels = labels.reshape(-1)
    for cls in np.unique(flat_labels[flat_labels >= 0]):
        members = rng.permutation(np.flatnonzero(flat_labels == cls))
        cut = np.round(np.cumsum(fractions)[:-1] * len(members)).astype(int)
        for k, part in enumerate(np.split(members, cut)):
            flat[part] = k
    return split


def band_stats(
    cube: npt.NDArray[np.float32], split: npt.NDArray[np.int8]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Returns per-band mean and std over training pixels only.

    Args:
        cube: Reflectance, shape (height, width, bands).
        split: Split map from block_split or random_split.

    Returns:
        A (mean, std) pair, each of shape (bands,).
    """
    pixels = cube[split == TRAIN]
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def extract_patches(
    cube: npt.NDArray[np.float32],
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    radius: int,
) -> npt.NDArray[np.float32]:
    """Cuts square patches around pixels, reflecting at the image edge.

    Args:
        cube: Normalised reflectance, shape (height, width, bands).
        rows: Row of each centre pixel.
        cols: Column of each centre pixel.
        radius: Patch radius; patches are (2 * radius + 1) wide.

    Returns:
        Patches of shape (n, bands, side, side).
    """
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)), "reflect")
    side = 2 * radius + 1
    offsets = np.arange(side)
    patches = padded[
        rows[:, None, None] + offsets[None, :, None],
        cols[:, None, None] + offsets[None, None, :],
    ]
    return np.ascontiguousarray(patches.transpose(0, 3, 1, 2))
