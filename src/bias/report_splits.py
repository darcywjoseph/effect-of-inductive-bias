"""Writes per-class split sizes at every grid point for a config.

Usage: python -m bias.report_splits configs/images.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from bias import config as config_lib
from bias.data import cifar, subsets


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


def report(config: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    """Builds the split table for one config and writes it out.

    Args:
        config: Parsed config.
        out_dir: Directory for the per-class CSV.

    Returns:
        The per-class table.
    """
    name = config["dataset"]
    data = config["data"]
    splits = cifar.load(
        Path(config["data_root"]), data["val_per_class"], data["split_seed"]
    )
    frame, capped = _table(
        splits.pool_y,
        splits.val_y,
        splits.test_y,
        config["grid"],
        int(config["seeds"][0]),
    )
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
    args = parser.parse_args()
    for path in args.configs:
        frame = report(config_lib.load_config(path), args.out)
        print(f"{path.name}: {len(frame)} classes")
        print(frame.sum().to_string())


if __name__ == "__main__":
    main()
