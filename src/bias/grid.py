"""Runs the grid: LeJEPA pretraining, the tuning sweep, final runs.

Each stage is a list of jobs. A job already in the log is skipped, so a
stage can be stopped and resumed, and --shard i/n splits a stage across
processes that share a GPU, each writing its own log.

Tuning: every configuration, at each anchor size, one seed, every
(lr, weight decay) pair. Where the best learning rate is the smallest or
largest tried, the sweep steps one rung further along the config's
learning-rate ladder, and repeats round by round until every best is
bracketed or the ladder ends. Finals: every configuration, size and seed,
with the pair that scored best on validation at the size's nearest anchor
(nearest by grid position).

Budget check: a few setups rerun with a multiple of the step budget, seed
0, at their final settings.

Usage:
    python -m bias.grid configs/images.yaml --stage pretrain --shard 0/1
    python -m bias.grid configs/images.yaml --stage tune --shard 0/3
    python -m bias.grid configs/images.yaml --stage tune --pending
    python -m bias.grid configs/images.yaml --stage final --shard 0/3
    python -m bias.grid configs/images.yaml --stage budget --shard 0/3
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from bias import config as config_lib
from bias import pretrain_lejepa, runlog, train
from bias.data import tensors
from bias.train import Job


def nearest_anchor(size: str, grid: list[str], anchors: list[str]) -> str:
    """Returns the anchor nearest to a size by position in the grid.

    Args:
        size: Grid entry, such as "5%".
        grid: The config's grid, smallest first.
        anchors: The anchor entries, all in the grid.

    Returns:
        The nearest anchor.
    """
    position = grid.index(size)
    return min(anchors, key=lambda a: abs(grid.index(a) - position))


def tune_jobs(config: dict[str, Any], tune: pd.DataFrame) -> list[Job]:
    """Returns the sweep jobs: the base grid plus the next ladder rungs.

    Args:
        config: Parsed config.
        tune: Tuning rows logged so far.

    Returns:
        One job per configuration, anchor and (lr, weight decay) pair of
        the base sweep, plus extension_jobs.
    """
    sweep = config["sweep"]
    base = [
        Job(model, treatment, anchor, sweep["seed"], lr, wd)
        for model, treatment in config["configurations"]
        for anchor in config["anchors"]
        for lr in sweep["lr"]
        for wd in sweep["weight_decay"]
    ]
    return base + extension_jobs(config, tune)


def extension_jobs(config: dict[str, Any], tune: pd.DataFrame) -> list[Job]:
    """Returns sweep jobs that step past a learning rate at the edge.

    A (configuration, anchor) whose best learning rate is the smallest
    (largest) tried gets the next smaller (larger) rung of the ladder, at
    every weight decay. Groups still running the base sweep are skipped.

    Args:
        config: Parsed config, with sweep.lr_ladder.
        tune: Tuning rows logged so far.

    Returns:
        The extra jobs, possibly none.
    """
    sweep = config["sweep"]
    ladder = sorted(sweep.get("lr_ladder", sweep["lr"]))
    wanted = {(m, t) for m, t in config["configurations"]}
    keep = [(m, t) in wanted for m, t in zip(tune.model, tune.treatment, strict=True)]
    rows = tune[pd.Series(keep, index=tune.index, dtype=bool)]
    settings = best_settings(rows)
    jobs: list[Job] = []
    for (model, treatment, anchor), (best_lr, _) in sorted(settings.items()):
        mine = rows[
            (rows.model == model)
            & (rows.treatment == treatment)
            & (rows["size"] == anchor)
        ]
        tried = sorted({float(lr) for lr in mine.lr})
        if not set(sweep["lr"]) <= set(tried):
            continue
        rung = ladder.index(best_lr)
        step = -1 if best_lr == tried[0] else 1 if best_lr == tried[-1] else 0
        if step and 0 <= rung + step < len(ladder):
            jobs.extend(
                Job(model, treatment, anchor, sweep["seed"], ladder[rung + step], wd)
                for wd in sweep["weight_decay"]
            )
    return jobs


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


def tuned(
    config: dict[str, Any], settings: Settings, model: str, treatment: str, size: str
) -> tuple[float, float]:
    """Returns a setup's tuned settings at the anchor nearest to a size.

    Args:
        config: Parsed config.
        settings: Output of best_settings.
        model: Model name.
        treatment: Treatment name.
        size: Grid entry.

    Returns:
        (lr, weight decay).

    Raises:
        ValueError: If the anchor has no finished tuning runs.
    """
    key = (model, treatment, nearest_anchor(size, config["grid"], config["anchors"]))
    if key not in settings:
        raise ValueError(f"no tuning result for {key}")
    return settings[key]


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
    return [
        Job(
            model,
            treatment,
            size,
            seed,
            *tuned(config, settings, model, treatment, size),
        )
        for model, treatment in config["configurations"]
        for size in config["grid"]
        for seed in config["seeds"]
    ]


def budget_jobs(config: dict[str, Any], tune: pd.DataFrame) -> list[Job]:
    """Returns the budget-check jobs: seed 0 at the final settings.

    Args:
        config: Parsed config, with a budget_check section.
        tune: Tuning rows from the run log.

    Returns:
        One job per listed configuration and size; none without the section.
    """
    check = config.get("budget_check")
    if not check:
        return []
    wanted = {
        (m, t, size) for m, t in check["configurations"] for size in check["sizes"]
    }
    return [
        j
        for j in final_jobs(config, tune)
        if j.seed == 0 and (j.model, j.treatment, j.size) in wanted
    ]


def read_runs(config: dict[str, Any], stage: str) -> pd.DataFrame:
    """Reads one stage's runs from every log of the dataset.

    Args:
        config: Parsed config.
        stage: Stage label the runs were logged under.

    Returns:
        One row per run, in log order; empty if there are none.
    """
    logs = Path(config["results_dir"]).glob(f"{config['dataset']}_*.csv")
    frames = [pd.read_csv(path) for path in sorted(logs)]
    rows = pd.concat(frames) if frames else pd.DataFrame(columns=runlog.FIELDS)
    return rows[rows.stage == stage].reset_index(drop=True)


def logged_jobs(rows: pd.DataFrame) -> list[Job]:
    """Returns the job behind each logged run.

    Args:
        rows: Rows of a run log.

    Returns:
        One job per row, in row order.
    """
    return [
        Job(
            str(r["model"]),
            str(r["treatment"]),
            str(r["size"]),
            int(r["seed"]),
            float(r["lr"]),
            float(r["weight_decay"]),
        )
        for r in rows.to_dict("records")
    ]


def unlogged(config: dict[str, Any], stage: str, jobs: list[Job]) -> list[Job]:
    """Returns the jobs a stage has not logged yet, without repeats.

    Args:
        config: Parsed config.
        stage: Stage label.
        jobs: The stage's jobs.

    Returns:
        The jobs still to run, in the order given.
    """
    done = set(logged_jobs(read_runs(config, stage)))
    return list(dict.fromkeys(j for j in jobs if j not in done))


def pending_jobs(config: dict[str, Any], stage: str) -> list[Job]:
    """Returns a stage's jobs not yet in the logs, longest first.

    Args:
        config: Parsed config.
        stage: "tune", "final" or "budget".

    Returns:
        The jobs to run, sorted so shards finish at about the same time.
    """
    make = {"tune": tune_jobs, "final": final_jobs, "budget": budget_jobs}[stage]
    left = unlogged(config, stage, make(config, read_runs(config, "tune")))
    left.sort(key=lambda j: -config["grid"].index(j.size))
    return left


def run_jobs(
    config: dict[str, Any],
    data: tensors.TensorData,
    jobs: list[Job],
    stage: str,
    log: Path,
    device: torch.device,
    max_steps: int | None = None,
    step_factor: int = 1,
) -> None:
    """Trains each job and appends it to the log.

    Args:
        config: Parsed config.
        data: Dataset tensors on the device.
        jobs: The runs to make.
        stage: Stage label for the log.
        log: CSV to append to.
        device: Training device.
        max_steps: Optional override of the step budget, for smoke tests.
        step_factor: Multiple of the step budget to train for.
    """
    for job in jobs:
        record = train.train_run(
            config, data, job, stage, device, max_steps, step_factor
        )
        print(record, flush=True)
        runlog.append_run(log, record)


def _shard(items: list[Any], shard: str) -> list[Any]:
    index, count = (int(part) for part in shard.split("/"))
    return items[index::count]


def main() -> None:
    """Runs one shard of one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--stage",
        choices=["pretrain", "tune", "final", "budget"],
        required=True,
    )
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke tests")
    parser.add_argument(
        "--pending", action="store_true", help="print the jobs left and exit"
    )
    args = parser.parse_args()
    config = config_lib.load_config(args.config)
    if args.pending:
        print(len(pending_jobs(config, args.stage)))
        return
    device = train.pick_device()
    data = tensors.load_tensors(config, device)

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

    name = f"{config['dataset']}_{args.stage}_{args.shard.replace('/', 'of')}.csv"
    run_jobs(
        config,
        data,
        _shard(pending_jobs(config, args.stage), args.shard),
        args.stage,
        Path(config["results_dir"]) / name,
        device,
        args.max_steps,
        config["budget_check"]["factor"] if args.stage == "budget" else 1,
    )


if __name__ == "__main__":
    main()
