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


def test_job_counts() -> None:
    tune = grid.tune_jobs(IMAGES)
    assert len(tune) == 10 * 3 * 6
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
    assert len(finals) == 10 * 7 * 3
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
