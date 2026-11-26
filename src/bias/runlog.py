"""One CSV row per training run."""

import csv
import dataclasses
import json
import os
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class RunRecord:
    """What the log keeps about one training run.

    Attributes:
        dataset: Dataset name, such as "cifar100".
        model: "cnn", "vit", "mlp" or "vit_<variant>" (see models.vit).
        treatment: "none", "aug", "lejepa", "imagenet" or "imagenet1k".
        size: Grid entry, such as "10%".
        n_labelled: Labelled examples actually used.
        seed: Seed for the subset and the weight initialisation.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        steps: Optimisation steps run.
        best_val: Best validation score, used to pick the checkpoint.
        test: Test score of the chosen checkpoint.
        metric: "top1".
        n_params: Trainable parameter count.
        gpu: Device name.
        wall_seconds: Wall-clock training time.
        stage: What the run was for, such as "tune", "final" or "scramble_pixels".
        precision: "fp32" or "bf16" (autocast).
    """

    dataset: str
    model: str
    treatment: str
    size: str
    n_labelled: int
    seed: int
    lr: float
    weight_decay: float
    steps: int
    best_val: float
    test: float
    metric: str
    n_params: int
    gpu: str
    wall_seconds: float
    stage: str
    precision: str


FIELDS = [field.name for field in dataclasses.fields(RunRecord)]


def append_run(path: Path, record: RunRecord) -> None:
    """Appends one run to the CSV, writing a header if the file is new.

    Args:
        path: CSV file to append to. Parent directories are created.
        record: The run to log.

    Raises:
        ValueError: If an existing file has a different header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != FIELDS:
            raise ValueError(f"{path} has header {header}, expected {FIELDS}")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(dataclasses.asdict(record))


def append_history(path: Path, entry: dict[str, Any]) -> None:
    """Appends one run's validation history as a JSON line.

    The line goes out in one write to a file opened for appending, so
    shards that share the file do not interleave their lines.

    Args:
        path: JSON-lines file. Parent directories are created.
        entry: The run's key fields and its (step, score) pairs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(entry) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
