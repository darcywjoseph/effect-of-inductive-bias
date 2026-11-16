import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from bias import runlog

RECORD = runlog.RunRecord(
    dataset="cifar100",
    model="cnn",
    treatment="none",
    size="1%",
    n_labelled=450,
    seed=0,
    lr=1e-3,
    weight_decay=0.05,
    steps=2000,
    best_val=0.1,
    test=0.1,
    metric="top1",
    n_params=11_000_000,
    gpu="mps",
    wall_seconds=60.0,
    stage="final",
    precision="bf16",
)


def test_append_writes_header_once(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    runlog.append_run(path, RECORD)
    runlog.append_run(path, dataclasses.replace(RECORD, seed=1))
    frame = pd.read_csv(path)
    assert list(frame.columns) == runlog.FIELDS
    assert frame["seed"].tolist() == [0, 1]


def test_mismatched_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        runlog.append_run(path, RECORD)
