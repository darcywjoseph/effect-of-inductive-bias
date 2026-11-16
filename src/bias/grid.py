"""Runs the study's grid: LeJEPA pretraining, the tuning sweep, final runs.

Each stage is a list of jobs. A job already in the log is skipped, so a
stage can be stopped and resumed, and --shard i/n splits a stage across
processes that share a GPU, each writing its own log.

Tuning: every configuration, at each anchor size, one seed, every
(lr, weight decay) pair. Finals: every configuration, size and seed, with
the pair that scored best on validation at the size's nearest anchor
(nearest by grid position). The test score plays no part in the choice.

Usage:
    python -m bias.grid configs/images.yaml --stage pretrain --shard 0/1
    python -m bias.grid configs/images.yaml --stage tune --shard 0/3
    python -m bias.grid configs/images.yaml --stage final --shard 0/3
"""

import argparse
import dataclasses
import glob
from pathlib import Path
from typing import Any

import pandas as pd

from bias import config as config_lib
from bias import pretrain_lejepa, runlog, train


@dataclasses.dataclass(frozen=True)
class Job:
    """One supervised run.

    Attributes:
        model: "cnn", "vit" or "mlp".
        treatment: "none", "aug", "lejepa" or "imagenet".
        size: Grid entry.
        seed: Run seed.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
    """

    model: str
    treatment: str
    size: str
    seed: int
    lr: float
    weight_decay: float


def nearest_anchor(size: str, grid: list[str], anchors: list[str]) -> str:
    """Returns the anchor nearest to a size by position in the grid.

    The grids are roughly log-spaced, so position stands in for log size,
    and it also places "all", which has no fixed number. No grid position
    is equally far from two anchors.

    Args:
        size: Grid entry, such as "5%".
        grid: The config's grid, smallest first.
        anchors: The anchor entries, all in the grid.

    Returns:
        The nearest anchor.
    """
    position = grid.index(size)
    return min(anchors, key=lambda a: abs(grid.index(a) - position))


def tune_jobs(config: dict[str, Any]) -> list[Job]:
    """Returns the sweep jobs.

    Args:
        config: Parsed config.

    Returns:
        One job per configuration, anchor and (lr, weight decay) pair.
    """
    sweep = config["sweep"]
    return [
        Job(model, treatment, anchor, sweep["seed"], lr, wd)
        for model, treatment in config["configurations"]
        for anchor in config["anchors"]
        for lr in sweep["lr"]
        for wd in sweep["weight_decay"]
    ]


Settings = dict[tuple[str, str, str], tuple[float, float]]


def best_settings(tune: pd.DataFrame) -> Settings:
    """Picks the (lr, weight decay) with the best validation score.

    Args:
        tune: Tuning rows from the run log.

    Returns:
        A map from (model, treatment, anchor) to (lr, weight decay). Ties
        go to the smaller learning rate, then the smaller weight decay.
    """
    ranked = tune.sort_values(
        ["best_val", "lr", "weight_decay"], ascending=[False, True, True]
    )
    best = ranked.groupby(["model", "treatment", "size"], sort=False).head(1)
    return {
        (str(r["model"]), str(r["treatment"]), str(r["size"])): (
            float(r["lr"]),
            float(r["weight_decay"]),
        )
        for r in best.to_dict("records")
    }


def final_jobs(config: dict[str, Any], tune: pd.DataFrame) -> list[Job]:
    """Returns the final jobs, with settings from the nearest anchor.

    Args:
        config: Parsed config.
        tune: Tuning rows from the run log.

    Returns:
        One job per configuration, size and seed.

    Raises:
        ValueError: If an anchor has no finished tuning runs.
    """
    settings = best_settings(tune)
    jobs: list[Job] = []
    for model, treatment in config["configurations"]:
        for size in config["grid"]:
            anchor = nearest_anchor(size, config["grid"], config["anchors"])
            key = (model, treatment, anchor)
            if key not in settings:
                raise ValueError(f"no tuning result for {key}")
            lr, wd = settings[key]
            jobs.extend(Job(model, treatment, size, s, lr, wd) for s in config["seeds"])
    return jobs


def _read_logs(log_glob: str, stage: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in glob.glob(log_glob)]
    rows = pd.concat(frames) if frames else pd.DataFrame(columns=runlog.FIELDS)
    return rows[rows.stage == stage]


def _done(rows: pd.DataFrame) -> set[Job]:
    return {
        Job(
            str(r["model"]),
            str(r["treatment"]),
            str(r["size"]),
            int(r["seed"]),
            float(r["lr"]),
            float(r["weight_decay"]),
        )
        for r in rows.to_dict("records")
    }


def _shard(items: list[Any], shard: str) -> list[Any]:
    index, count = (int(part) for part in shard.split("/"))
    return items[index::count]


def main() -> None:
    """Runs one shard of one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=["pretrain", "tune", "final"], required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke tests")
    args = parser.parse_args()
    config = config_lib.load_config(args.config)
    device = train.pick_device()
    data = train.load_tensors(config, device)
    logs = Path(config["results_dir"])
    log_glob = str(logs / f"{config['dataset']}_*.csv")

    if args.stage == "pretrain":
        pairs = sorted({m for m, t in config["configurations"] if t == "lejepa"})
        work = [(m, s) for s in config["seeds"] for m in pairs]
        for model, seed in _shard(work, args.shard):
            if config_lib.lejepa_checkpoint(config, model, seed).exists():
                continue
            summary = pretrain_lejepa.pretrain(
                config, data, model, seed, device, args.max_steps
            )
            print({k: v for k, v in summary.items() if k != "history"}, flush=True)
        return

    if args.stage == "tune":
        jobs = tune_jobs(config)
    else:
        jobs = final_jobs(config, _read_logs(log_glob, "tune"))
    done = _done(_read_logs(log_glob, args.stage))
    # Longest first, so shards finish at about the same time.
    jobs.sort(key=lambda j: -config["grid"].index(j.size))
    log = logs / f"{config['dataset']}_{args.stage}_{args.shard.replace('/', 'of')}.csv"
    for job in _shard(jobs, args.shard):
        if job in done:
            continue
        record = train.train_run(
            config,
            data,
            job.model,
            job.treatment,
            job.size,
            job.seed,
            job.lr,
            job.weight_decay,
            args.stage,
            device,
            args.max_steps,
        )
        print(record, flush=True)
        runlog.append_run(log, record)


if __name__ == "__main__":
    main()
