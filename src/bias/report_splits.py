"""Writes per-class split sizes for a config, and a block map for scenes.

Usage: python -m bias.report_splits configs/images.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from bias import config as config_lib
from bias.data import cifar, hsi, speech_commands, subsets


def _table(
    pool_y: npt.NDArray[np.int64],
    val_y: npt.NDArray[np.int64],
    test_y: npt.NDArray[np.int64],
    grid: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    classes = np.unique(np.concatenate([pool_y, val_y, test_y]))
    frame = pd.DataFrame(index=pd.Index(classes, name="class"))
    frame["pool"] = np.bincount(pool_y, minlength=len(classes))[classes]
    frame["val"] = np.bincount(val_y, minlength=len(classes))[classes]
    frame["test"] = np.bincount(test_y, minlength=len(classes))[classes]
    capped: dict[str, list[int]] = {}
    for size in grid:
        sub = subsets.draw_subset(pool_y, size, seed)
        frame[size] = [sub.per_class.get(int(c), 0) for c in classes]
        capped[size] = sub.capped_classes
    return frame, capped


def _plot_blocks(scene: hsi.Scene, split: hsi.BlockSplit, name: str, out: Path) -> None:
    colours = matplotlib.colors.ListedColormap(["#4C78A8", "#F58518", "#E45756"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    axes[0].imshow(
        np.ma.masked_less(scene.labels, 0), cmap="tab20", interpolation="nearest"
    )
    axes[0].set_title("Ground truth")
    axes[1].imshow(
        split.block_split, cmap=colours, vmin=0, vmax=2, interpolation="nearest"
    )
    axes[1].set_title("Block assignment")
    kept = np.ma.masked_less(split.split, 0)
    axes[2].imshow(scene.labels >= 0, cmap="Greys", alpha=0.25, interpolation="nearest")
    axes[2].imshow(kept, cmap=colours, vmin=0, vmax=2, interpolation="nearest")
    axes[2].set_title("Kept labelled pixels (grey: lost to buffer)")
    handles = [
        matplotlib.patches.Rectangle((0, 0), 1, 1, color=colours(k)) for k in range(3)
    ]
    fig.legend(handles, hsi.SPLIT_NAMES, loc="lower center", ncol=3)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}: block split, score {split.score:.2f}")
    fig.savefig(out, dpi=120)
    plt.close(fig)


def report(config: dict[str, Any], out_dir: Path, fig_dir: Path) -> pd.DataFrame:
    """Builds the split table for one config and writes it out.

    Args:
        config: Parsed config.
        out_dir: Directory for the per-class CSV.
        fig_dir: Directory for the hyperspectral block figure.

    Returns:
        The per-class table.
    """
    root = Path(config["data_root"])
    name = config["dataset"]
    data = config["data"]
    seed = int(config["seeds"][0])
    ys: list[npt.NDArray[np.int64]]
    if name == "cifar100":
        splits = cifar.load(root, data["val_per_class"], data["split_seed"])
        ys = [splits.pool_y, splits.val_y, splits.test_y]
    elif name == "speech_commands_v2":
        ys = [
            speech_commands.load_split(root, s).labels for s in ("pool", "val", "test")
        ]
    else:
        scene = hsi.load_scene(root, name)
        split = hsi.block_split(
            scene.labels,
            block=data["block"],
            radius=data["patch_radius"],
            fractions=tuple(data["fractions"]),
            seed=data["split_seed"],
            tries=data["tries"],
        )
        ys = [scene.labels[split.split == k] for k in (hsi.TRAIN, hsi.VAL, hsi.TEST)]
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_blocks(scene, split, name, fig_dir / f"hsi_blocks_{name}.png")
        lost = np.count_nonzero((scene.labels >= 0) & (split.split < 0))
        print(
            f"{name}: buffer drops {lost} of {np.count_nonzero(scene.labels >= 0)} "
            f"labelled pixels; balance score {split.score:.3f}"
        )
    frame, capped = _table(ys[0], ys[1], ys[2], config["grid"], seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"splits_{name}.csv")
    for size, classes in capped.items():
        if classes:
            print(f"{name} {size}: capped classes {classes}")
    return frame


def main() -> None:
    """Runs the report for each config given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/splits"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    for path in args.configs:
        frame = report(config_lib.load_config(path), args.out, args.figures)
        totals = frame.sum()
        print(f"{path.name}: {len(frame)} classes")
        print(totals.to_string())


if __name__ == "__main__":
    main()
