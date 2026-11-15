"""One CSV row per training run."""

import csv
import dataclasses
from pathlib import Path


@dataclasses.dataclass
class RunRecord:
    """Everything the study logs about one training run.

    Attributes:
        dataset: Dataset name, such as "cifar100".
        model: "cnn", "vit" or "mlp".
        treatment: "none", "aug", "lejepa" or "imagenet".
        size: Grid entry, such as "10%" or "20pc".
        n_labelled: Labelled examples actually used.
        seed: Seed for the subset and the weight initialisation.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        steps: Optimisation steps run.
        best_val: Best validation score, used to pick the checkpoint.
        test: Test score of the chosen checkpoint.
        metric: "top1" or "balanced_acc".
        n_params: Trainable parameter count.
        gpu: Device name.
        wall_seconds: Wall-clock training time.
        stage: "pilot", "tune" for sweep runs, or "final" for reported runs.
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
