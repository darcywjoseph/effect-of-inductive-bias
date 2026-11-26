"""Supervised training and fine-tuning of one configuration.

Usage:
    python -m bias.train configs/images.yaml --model cnn \
        --treatment none --size 10% --seed 0 --lr 1e-3 --wd 0.05
"""

import argparse
import copy
import dataclasses
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from bias import augment, runlog
from bias import config as config_lib
from bias.data import subsets, tensors
from bias.models import registry

# Treatments that start from outside-data weights instead of random ones.
PRETRAINED = frozenset({"imagenet", "imagenet1k"})


@dataclasses.dataclass(frozen=True)
class Job:
    """One supervised run.

    Attributes:
        model: "cnn", "vit" or "mlp".
        treatment: "none", "aug", "lejepa", "imagenet" or "imagenet1k".
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


def pick_device() -> torch.device:
    """Returns CUDA if present, then MPS, then CPU.

    Returns:
        The device to train on.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # By default MPS may allocate past physical RAM and swap until the
        # Mac hangs. Capped at Metal's recommended working set, an oversized
        # run fails with an out-of-memory error instead.
        torch.mps.set_per_process_memory_fraction(1.0)
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    """Returns a readable name for the device, for the run log.

    Args:
        device: The training device.

    Returns:
        The GPU or chip name.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{device.type}:{platform.processor() or platform.machine()}"


def step_budget(n_labelled: int, config: dict[str, Any]) -> int:
    """Returns the optimisation steps for a subset size.

    Args:
        n_labelled: Labelled examples in the subset.
        config: Parsed config with batch_size and steps.

    Returns:
        Steps worth epochs_equivalent epochs, clipped to [min, max].
    """
    steps = config["steps"]
    raw = math.ceil(steps["epochs_equivalent"] * n_labelled / config["batch_size"])
    return int(min(max(raw, steps["min"]), steps["max"]))


@torch.no_grad()
def accuracy(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch: int = 1000,
    bf16: bool = False,
) -> float:
    """Returns the model's top-1 accuracy on a labelled set.

    Args:
        model: Classifier.
        x: Inputs.
        y: Labels.
        batch: Evaluation batch size.
        bf16: Run the forward pass under bf16 autocast, as in training.

    Returns:
        The accuracy, in [0, 1].
    """
    model.eval()
    predictions = []
    for start in range(0, len(x), batch):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=bf16):
            predictions.append(model(x[start : start + batch]).argmax(1))
    model.train()
    return float((torch.cat(predictions) == y).float().mean())


def lr_at(step: int, total: int, peak: float, warmup_frac: float) -> float:
    """Returns the learning rate for linear warmup then cosine decay.

    Args:
        step: Current step, from 0.
        total: Total steps.
        peak: Peak learning rate.
        warmup_frac: Share of steps spent warming up.

    Returns:
        The learning rate.
    """
    warmup = max(1, int(warmup_frac * total))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    micro_batch: int | None = None,
    bf16: bool = False,
) -> torch.Tensor:
    """Adds the gradient of the batch's mean cross-entropy to the model.

    Args:
        model: Classifier, with gradients zeroed by the caller.
        x: Input batch.
        y: Labels.
        micro_batch: If set, the batch goes through the model in pieces of
          this many examples, to save memory. Without batch norm the
          gradient is the same as for the whole batch.
        bf16: Run the forward pass under bf16 autocast.

    Returns:
        The loss of the last piece, detached, for divergence checks.
    """
    piece = micro_batch or len(x)
    for start in range(0, len(x), piece):
        xp, yp = x[start : start + piece], y[start : start + piece]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = torch.nn.functional.cross_entropy(model(xp), yp)
        (loss * len(xp) / len(x)).backward()  # type: ignore[no-untyped-call]
    return loss.detach()


def _load_backbone(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    head = {"fc.weight", "fc.bias", "head.weight", "head.bias"}
    if result.unexpected_keys or not set(result.missing_keys) <= head:
        raise ValueError(f"checkpoint does not fit the model: {result}")


def train_run(
    config: dict[str, Any],
    data: tensors.TensorData,
    job: Job,
    stage: str,
    device: torch.device,
    max_steps: int | None = None,
    step_factor: int = 1,
    save_to: Path | None = None,
    micro_batch: int | None = None,
) -> runlog.RunRecord:
    """Trains one model on one subset and scores the best checkpoint.

    The checkpoint with the best validation score is kept and only it is
    scored on the test set.

    Args:
        config: Parsed config.
        data: Dataset tensors on the device.
        job: What to train. Its seed sets the subset, the init and the
          batch order. The "imagenet", "imagenet1k" and "lejepa" (this
          seed's backbone) treatments fine-tune the whole network with
          augmentation.
        stage: Label for the run log, such as "tune" or "final".
        device: Training device.
        max_steps: Optional override of the step budget, for smoke tests.
        step_factor: Multiple of the step budget to train for.
        save_to: Where to save the best checkpoint's weights, if anywhere.
        micro_batch: If set, each batch goes through the model in pieces of
          this many images and their gradients are summed before the step,
          to save memory. The update is the same as for the whole batch.
          Evaluation then runs in batches of 8 x micro_batch.

    Returns:
        The run's log record.

    Raises:
        ValueError: If the treatment is unknown.
        FloatingPointError: If the loss diverges.
    """
    treatment, seed, lr = job.treatment, job.seed, job.lr
    if treatment not in PRETRAINED | {"none", "aug", "lejepa"}:
        raise ValueError(f"unknown treatment: {treatment=}")
    subset = subsets.draw_subset(data.pool_y.cpu().numpy(), job.size, seed)
    index = torch.from_numpy(subset.indices).to(device)
    x, y = data.pool_x[index], data.pool_y[index]
    total = max_steps or step_factor * step_budget(len(index), config)
    batch = min(config["batch_size"], len(index))
    eval_every = max(1, total // config["steps"]["evals"])

    torch.manual_seed(seed)
    model = registry.build(
        job.model, weights=treatment if treatment in PRETRAINED else None
    )
    if treatment == "lejepa":
        _load_backbone(model, config_lib.lejepa_checkpoint(config, job.model, seed))
    model = model.to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=job.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(index), generator=generator)
    cursor = 0
    best_val, best_state = -1.0, copy.deepcopy(model.state_dict())
    history: list[tuple[int, float]] = []
    augmented = None
    if treatment != "none":
        augmented = augment.augmented_batches(
            data.pool_raw[torch.from_numpy(subset.indices)],
            data.pool_y.cpu()[torch.from_numpy(subset.indices)],
            data.augment_example,
            batch,
            total,
            seed,
            int(os.environ.get("BIAS_WORKERS", config.get("workers", 8))),
        )
    bf16 = config.get("precision", "fp32") == "bf16"
    # Evaluation has no gradients to store, so it can take bigger pieces.
    eval_batch = 1000 if micro_batch is None else 8 * micro_batch

    started = time.perf_counter()
    for step in range(total):
        if augmented is not None:
            raw, labels = next(augmented)
            xb = data.prepare(raw)
            yb = labels.to(device, non_blocking=True)
        else:
            if cursor + batch > len(order):
                order = torch.randperm(len(index), generator=generator)
                cursor = 0
            take = order[cursor : cursor + batch].to(device)
            cursor += batch
            xb, yb = x[take], y[take]
        for group in optimiser.param_groups:
            group["lr"] = lr_at(step, total, lr, config["warmup_frac"])
        optimiser.zero_grad(set_to_none=True)
        loss = backward(model, xb, yb, micro_batch, bf16)
        optimiser.step()
        if (step + 1) % eval_every == 0 or step + 1 == total:
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss diverged at step {step}: {loss.item()}")
            val = accuracy(model, data.val_x, data.val_y, eval_batch, bf16)
            history.append((step + 1, val))
            if val > best_val:
                best_val, best_state = val, copy.deepcopy(model.state_dict())
    if device.type == "mps":
        torch.mps.synchronize()
    wall = time.perf_counter() - started

    model.load_state_dict(best_state)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_to)
    runlog.append_history(
        Path(config["results_dir"]) / "history" / f"{config['dataset']}_{stage}.jsonl",
        {**dataclasses.asdict(job), "steps": total, "val": history},
    )
    return runlog.RunRecord(
        **dataclasses.asdict(job),
        dataset=config["dataset"],
        n_labelled=len(index),
        steps=total,
        best_val=best_val,
        test=accuracy(model, data.test_x, data.test_y, eval_batch, bf16),
        metric=config["metric"],
        n_params=registry.count_params(model),
        gpu=device_name(device),
        wall_seconds=wall,
        stage=stage,
        precision="bf16" if bf16 else "fp32",
    )


def main() -> None:
    """Trains one configuration from the command line and logs it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--treatment", default="none")
    parser.add_argument("--size", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--wd", type=float, required=True)
    parser.add_argument("--stage", default="final")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    config = config_lib.load_config(args.config)
    device = pick_device()
    data = tensors.load_tensors(config, device)
    job = Job(args.model, args.treatment, args.size, args.seed, args.lr, args.wd)
    record = train_run(config, data, job, args.stage, device, args.max_steps)
    print(record)
    default = Path(config["results_dir"]) / f"{config['dataset']}_manual.csv"
    runlog.append_run(args.log or default, record)


if __name__ == "__main__":
    main()
