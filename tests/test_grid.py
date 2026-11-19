from pathlib import Path

import pandas as pd

from bias import config as config_lib
from bias import grid

IMAGES = config_lib.load_config(Path("configs/images.yaml"))
HSI = config_lib.load_config(Path("configs/hsi_salinas.yaml"))


def test_nearest_anchor_images() -> None:
    mapping = {
        s: grid.nearest_anchor(s, IMAGES["grid"], IMAGES["anchors"])
        for s in IMAGES["grid"]
    }
    assert mapping == {
        "1%": "1%",
        "2%": "1%",
        "5%": "10%",
        "10%": "10%",
        "20%": "10%",
        "50%": "100%",
        "100%": "100%",
    }


def test_nearest_anchor_hsi() -> None:
    mapping = {
        s: grid.nearest_anchor(s, HSI["grid"], HSI["anchors"]) for s in HSI["grid"]
    }
    assert mapping["10pc"] == "5pc"
    assert mapping["100pc"] == "50pc"
    assert mapping["200pc"] == "all"


EMPTY = pd.DataFrame(
    columns=["model", "treatment", "size", "lr", "weight_decay", "best_val"]
)


def _sweep_rows(jobs: list[grid.Job], score: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": j.model,
                "treatment": j.treatment,
                "size": j.size,
                "lr": j.lr,
                "weight_decay": j.weight_decay,
                "best_val": score[j.lr] + j.weight_decay,
            }
            for j in jobs
        ]
    )


def test_job_counts() -> None:
    tune = grid.tune_jobs(IMAGES, EMPTY)
    assert len(tune) == 12 * 3 * 6
    rows = pd.DataFrame(
        [
            {
                "model": j.model,
                "treatment": j.treatment,
                "size": j.size,
                "lr": j.lr,
                "weight_decay": j.weight_decay,
                "best_val": j.lr * j.weight_decay,
            }
            for j in tune
        ]
    )
    finals = grid.final_jobs(IMAGES, rows)
    assert len(finals) == 12 * 7 * 3
    # Highest validation wins: lr 1e-3, wd 0.1 in this fake sweep.
    assert {(j.lr, j.weight_decay) for j in finals} == {(1e-3, 0.1)}


def test_best_settings_ties_go_to_smaller_lr() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "cnn",
                "treatment": "none",
                "size": "1%",
                "lr": 1e-3,
                "weight_decay": 0.05,
                "best_val": 0.5,
            },
            {
                "model": "cnn",
                "treatment": "none",
                "size": "1%",
                "lr": 3e-4,
                "weight_decay": 0.1,
                "best_val": 0.5,
            },
        ]
    )
    assert grid.best_settings(rows) == {("cnn", "none", "1%"): (3e-4, 0.1)}


def test_ladder_steps_past_an_edge_until_bracketed() -> None:
    config = {**IMAGES, "configurations": [["cnn", "aug"]], "anchors": ["1%"]}
    # Validation keeps rising with lr up to 3e-3, then falls.
    score = {1e-4: 0.1, 3e-4: 0.2, 1e-3: 0.3, 3e-3: 0.4, 1e-2: 0.35}
    base = grid.tune_jobs(config, EMPTY)
    assert len(base) == 6
    first = grid.extension_jobs(config, _sweep_rows(base, score))
    assert {j.lr for j in first} == {3e-3}
    second = grid.extension_jobs(config, _sweep_rows(base + first, score))
    assert {j.lr for j in second} == {1e-2}
    third = grid.extension_jobs(config, _sweep_rows(base + first + second, score))
    assert third == []


def test_ladder_steps_down_and_stops_at_its_end() -> None:
    config = {**IMAGES, "configurations": [["vit", "imagenet"]], "anchors": ["1%"]}
    score = {1e-5: 0.5, 3e-5: 0.4, 1e-4: 0.3, 3e-4: 0.2, 1e-3: 0.1}
    jobs = grid.tune_jobs(config, EMPTY)
    for _ in range(4):
        jobs += grid.extension_jobs(config, _sweep_rows(jobs, score))
    assert sorted({j.lr for j in jobs})[0] == 1e-5
    assert grid.extension_jobs(config, _sweep_rows(jobs, score)) == []
