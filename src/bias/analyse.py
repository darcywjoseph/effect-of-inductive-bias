"""Exchange-rate tables from the final runs.

For each dataset and each of its two targets, every comparison in the
study gets a point estimate of E and a seed-bootstrap interval. Targets
are the seed-mean test scores of the CNN with augmentation at the sizes
fixed in the config. Where a comparison needs "the ViT with its best
treatment", best means the highest seed-mean validation score averaged
over the grid, so the test set chooses nothing.

Usage:
    python -m bias.analyse configs/images.yaml
"""

import argparse
import dataclasses
import glob
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from bias import config as config_lib
from bias import exchange_rate

Setup = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class Comparison:
    """One row of the study's comparison table.

    Attributes:
        question: What the row answers.
        reference: (model, treatment) whose data need is the numerator.
        treatment: (model, treatment) whose data need is the denominator.
    """

    question: str
    reference: Setup
    treatment: Setup


BEST_VIT = ("vit", "best")

COMPARISONS = [
    Comparison("CNN bias vs ViT, no aug", ("vit", "none"), ("cnn", "none")),
    Comparison("CNN bias vs MLP, no aug", ("mlp", "none"), ("cnn", "none")),
    Comparison("CNN bias vs ViT, with aug", ("vit", "aug"), ("cnn", "aug")),
    Comparison("CNN bias vs MLP, with aug", ("mlp", "aug"), ("cnn", "aug")),
    Comparison("Augmentation, ViT", ("vit", "none"), ("vit", "aug")),
    Comparison("Augmentation, MLP", ("mlp", "none"), ("mlp", "aug")),
    Comparison("Augmentation, CNN", ("cnn", "none"), ("cnn", "aug")),
    Comparison("Self-supervision, ViT", ("vit", "aug"), ("vit", "lejepa")),
    Comparison("Self-supervision, MLP", ("mlp", "aug"), ("mlp", "lejepa")),
    Comparison("Self-supervision, CNN", ("cnn", "aug"), ("cnn", "lejepa")),
    Comparison("Pretrained start, ViT", ("vit", "aug"), ("vit", "imagenet")),
    Comparison("CNN lead left over best ViT", BEST_VIT, ("cnn", "aug")),
    Comparison("CNN bias with LeJEPA for both", ("vit", "lejepa"), ("cnn", "lejepa")),
]


@dataclasses.dataclass(frozen=True)
class Curve:
    """Seed-level learning curve of one setup.

    Attributes:
        sizes: Labelled examples at each grid point.
        test: Test scores, shape (seeds, sizes).
        val: Validation scores, shape (seeds, sizes).
    """

    sizes: npt.NDArray[np.float64]
    test: npt.NDArray[np.float64]
    val: npt.NDArray[np.float64]


def load_finals(config: dict[str, Any]) -> pd.DataFrame:
    """Reads every final run of the config's dataset.

    Args:
        config: Parsed config.

    Returns:
        One row per final run.

    Raises:
        ValueError: If there are no final runs, or a run appears twice.
    """
    pattern = str(Path(config["results_dir"]) / f"{config['dataset']}_final_*.csv")
    frames = [pd.read_csv(path) for path in sorted(glob.glob(pattern))]
    if not frames:
        raise ValueError(f"no final runs match {pattern}")
    rows = pd.concat(frames, ignore_index=True)
    key = ["model", "treatment", "size", "seed"]
    if rows.duplicated(key).any():
        raise ValueError(f"duplicate final runs:\n{rows[rows.duplicated(key)]}")
    return rows


def curve(rows: pd.DataFrame, setup: Setup, grid: list[str]) -> Curve:
    """Returns one setup's curve over the grid.

    Args:
        rows: Final runs.
        setup: (model, treatment).
        grid: The config's grid, smallest first.

    Returns:
        The curve.

    Raises:
        ValueError: If a grid point or seed is missing.
    """
    model, treatment = setup
    mine = rows[(rows.model == model) & (rows.treatment == treatment)]
    test = mine.pivot(index="seed", columns="size", values="test")
    val = mine.pivot(index="seed", columns="size", values="best_val")
    if set(test.columns) != set(grid) or test.isna().any().any():
        raise ValueError(f"{setup} is missing runs: has {sorted(test.columns)}")
    n = mine.groupby("size").n_labelled.mean()
    return Curve(
        sizes=n[grid].to_numpy(dtype=np.float64),
        test=test[grid].to_numpy(dtype=np.float64),
        val=val[grid].to_numpy(dtype=np.float64),
    )


def best_vit(rows: pd.DataFrame, grid: list[str]) -> Setup:
    """Returns the ViT setup with the best mean validation score.

    Args:
        rows: Final runs.
        grid: The config's grid.

    Returns:
        (model, treatment) of the ViT treatment chosen by validation.
    """
    treatments = sorted(set(rows[rows.model == "vit"].treatment))
    return max(
        (("vit", t) for t in treatments),
        key=lambda setup: float(curve(rows, setup, grid).val.mean()),
    )


def targets(config: dict[str, Any], rows: pd.DataFrame) -> dict[str, float]:
    """Returns the fixed targets: reference test scores at the target sizes.

    Args:
        config: Parsed config.
        rows: Final runs.

    Returns:
        A map from target size to target score.
    """
    spec = config["targets"]
    ref = (spec["reference"]["model"], spec["reference"]["treatment"])
    c = curve(rows, ref, config["grid"])
    means = c.test.mean(axis=0)
    return {size: float(means[config["grid"].index(size)]) for size in spec["sizes"]}


def table(config: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    """Computes every comparison at every target.

    Args:
        config: Parsed config.
        rows: Final runs.

    Returns:
        One row per comparison and target.
    """
    grid = config["grid"]
    boot = config["bootstrap"]
    setups = set(zip(rows.model, rows.treatment, strict=True))
    best = best_vit(rows, grid)
    out = []
    for target_size, target in targets(config, rows).items():
        for comp in COMPARISONS:
            ref = best if comp.reference == BEST_VIT else comp.reference
            if ref not in setups or comp.treatment not in setups:
                continue
            r, t = curve(rows, ref, grid), curve(rows, comp.treatment, grid)
            est = exchange_rate.exchange_rate(
                r.sizes, r.test.mean(0), t.sizes, t.test.mean(0), target
            )
            interval = exchange_rate.bootstrap_interval(
                r.sizes,
                r.test,
                t.sizes,
                t.test,
                target,
                draws=boot["draws"],
                level=boot["level"],
            )
            out.append(
                {
                    "dataset": config["dataset"],
                    "target_size": target_size,
                    "target": target,
                    "question": comp.question,
                    "reference": "+".join(ref),
                    "treatment": "+".join(comp.treatment),
                    "E": est.value,
                    "kind": est.kind,
                    "low": interval.low,
                    "high": interval.high,
                    "share_censored": interval.share_censored,
                    "share_undefined": interval.share_undefined,
                }
            )
    return pd.DataFrame(out)


def describe(row: Mapping[Hashable, Any]) -> str:
    """Formats one estimate for people, bounds included.

    Args:
        row: A row of the exchange-rate table.

    Returns:
        Text such as "3.2x [1.9, 5.0]" or "more than 10x".
    """
    if row["kind"] == "lower":
        return f"more than {row['E']:.1f}x"
    if row["kind"] == "upper":
        return f"less than {row['E']:.1f}x"
    if row["kind"] == "undefined":
        return "undefined"
    return f"{row['E']:.2f}x [{row['low']:.2f}, {row['high']:.2f}]"


def main() -> None:
    """Writes results/exchange_rates_<dataset>.csv and prints it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = config_lib.load_config(args.config)
    rows = load_finals(config)
    result = table(config, rows)
    path = Path(config["results_dir"]) / f"exchange_rates_{config['dataset']}.csv"
    result.to_csv(path, index=False)
    for row in result.to_dict("records"):
        print(f"{row['target_size']:>6} {row['question']:<34} {describe(row)}")


if __name__ == "__main__":
    main()
