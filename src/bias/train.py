"""Supervised training and fine-tuning of one configuration.

Usage:
    python -m bias.train configs/images.yaml --model cnn \
        --treatment none --size 10% --seed 0 --lr 1e-3 --wd 0.05
"""

import argparse
import copy
import dataclasses
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bias import augment, runlog
from bias import config as config_lib
from bias.data import cifar, subsets
from bias.models import registry

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


def pick_device() -> torch.device:
    """Returns CUDA if present, then MPS, then CPU.

    Returns:
        The device to train on.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
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


@dataclasses.dataclass(frozen=True)
class TensorData:
    """Model-ready tensors for one dataset, already on the device.

    Attributes:
        pool_x: Labelled pool inputs.
        pool_y: Labelled pool labels.
        val_x: Validation inputs.
        val_y: Validation labels.
        test_x: Test inputs.
        test_y: Test labels.
        num_classes: Number of classes.
        pool_raw: Labelled pool before normalisation, on the CPU, as the
          augmentation expects it (uint8 NCHW for images).
        mean: Per-channel mean on the device, broadcastable to a batch.
        std: Per-channel std on the device, broadcastable to a batch.
    """

    pool_x: torch.Tensor
    pool_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int
    pool_raw: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def normalise(self, raw: torch.Tensor) -> torch.Tensor:
        """Moves a raw batch to the device and normalises it.

        Args:
            raw: A batch shaped like pool_raw.

        Returns:
            Model-ready inputs.
        """
        x = raw.to(self.mean.device, non_blocking=True).float().div(255.0)
        return (x - self.mean) / self.std


def load_tensors(config: dict[str, Any], device: torch.device) -> TensorData:
    """Loads a dataset and normalises it onto the device.

    Args:
        config: Parsed config.
        device: Where the tensors live.

    Returns:
        The dataset as tensors.

    Raises:
        ValueError: If the dataset has no loader yet.
    """
    if config["dataset"] != "cifar100":
        raise ValueError(f"no tensor loader for {config['dataset']=}")
    data = config["data"]
    splits = cifar.load(
        Path(config["data_root"]), data["val_per_class"], data["split_seed"]
    )
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)

    def images(x: np.ndarray[Any, Any]) -> torch.Tensor:
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255.0)
        return ((t - mean) / std).to(device)

    def labels(y: np.ndarray[Any, Any]) -> torch.Tensor:
        return torch.from_numpy(y).to(device)

    return TensorData(
        pool_x=images(splits.pool_x),
        pool_y=labels(splits.pool_y),
        val_x=images(splits.val_x),
        val_y=labels(splits.val_y),
        test_x=images(splits.test_x),
        test_y=labels(splits.test_y),
        num_classes=100,
        pool_raw=torch.from_numpy(splits.pool_x).permute(0, 3, 1, 2).contiguous(),
        mean=mean.to(device),
        std=std.to(device),
    )


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
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 1000
) -> float:
    """Returns top-1 accuracy of the model on a labelled set.

    Args:
        model: Classifier.
        x: Inputs.
        y: Labels.
        batch: Evaluation batch size.

    Returns:
        Fraction of correct predictions.
    """
    model.eval()
    correct = 0
    for start in range(0, len(x), batch):
        logits = model(x[start : start + batch])
        correct += int((logits.argmax(1) == y[start : start + batch]).sum())
    model.train()
    return correct / len(x)


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


def _load_backbone(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    head = {"fc.weight", "fc.bias", "head.weight", "head.bias"}
    if result.unexpected_keys or not set(result.missing_keys) <= head:
        raise ValueError(f"checkpoint does not fit the model: {result}")


def train_run(
    config: dict[str, Any],
    data: TensorData,
    model_name: str,
    treatment: str,
    size: str,
    seed: int,
    lr: float,
    weight_decay: float,
    stage: str,
    device: torch.device,
    max_steps: int | None = None,
) -> runlog.RunRecord:
    """Trains one model on one subset and scores the best checkpoint.

    The checkpoint with the best validation score is kept and only it is
    scored on the test set. The test set plays no part in any choice.

    Args:
        config: Parsed config.
        data: Dataset tensors on the device.
        model_name: "cnn", "vit" or "mlp".
        treatment: "none", "aug", "imagenet" (ViT from ImageNet weights)
          or "lejepa" (from this seed's LeJEPA backbone). The last two
          fine-tune the whole network with augmentation.
        size: Grid entry, such as "10%".
        seed: Seed for the subset, the init and the batch order.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        stage: "pilot", "tune" or "final".
        device: Training device.
        max_steps: Optional override of the step budget, for benchmarks.

    Returns:
        The run's log record.

    Raises:
        ValueError: If the treatment is not implemented.
        FloatingPointError: If the loss diverges.
    """
    if treatment not in ("none", "aug", "imagenet", "lejepa"):
        raise ValueError(f"treatment not implemented yet: {treatment=}")
    subset = subsets.draw_subset(data.pool_y.cpu().numpy(), size, seed)
    index = torch.from_numpy(subset.indices).to(device)
    x, y = data.pool_x[index], data.pool_y[index]
    total = max_steps or step_budget(len(index), config)
    batch = min(config["batch_size"], len(index))
    eval_every = max(1, total // config["steps"]["evals"])

    torch.manual_seed(seed)
    model = registry.build(
        model_name, config["dataset"], pretrained=treatment == "imagenet"
    )
    if treatment == "lejepa":
        _load_backbone(model, config_lib.lejepa_checkpoint(config, model_name, seed))
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(index), generator=generator)
    cursor = 0
    best_val, best_state = -1.0, copy.deepcopy(model.state_dict())
    augmented = None
    if treatment in ("aug", "imagenet", "lejepa"):
        augmented = augment.augmented_batches(
            data.pool_raw[torch.from_numpy(subset.indices)],
            data.pool_y.cpu()[torch.from_numpy(subset.indices)],
            augment.cifar_transform(),
            batch,
            total,
            seed,
            config.get("workers", 8),
        )
    bf16 = config.get("precision", "fp32") == "bf16"

    started = time.perf_counter()
    for step in range(total):
        if augmented is not None:
            raw, labels = next(augmented)
            xb, yb = data.normalise(raw), labels.to(device, non_blocking=True)
        else:
            if cursor + batch > len(order):
                order = torch.randperm(len(index), generator=generator)
                cursor = 0
            take = order[cursor : cursor + batch].to(device)
            cursor += batch
            xb, yb = x[take], y[take]
        for group in optimiser.param_groups:
            group["lr"] = lr_at(step, total, lr, config["optimiser"]["warmup_frac"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        if (step + 1) % eval_every == 0 or step + 1 == total:
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss diverged at step {step}: {loss.item()}")
            val = accuracy(model, data.val_x, data.val_y)
            if val > best_val:
                best_val, best_state = val, copy.deepcopy(model.state_dict())
    if device.type == "mps":
        torch.mps.synchronize()
    wall = time.perf_counter() - started

    model.load_state_dict(best_state)
    return runlog.RunRecord(
        dataset=config["dataset"],
        model=model_name,
        treatment=treatment,
        size=size,
        n_labelled=len(index),
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
        steps=total,
        best_val=best_val,
        test=accuracy(model, data.test_x, data.test_y),
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
    data = load_tensors(config, device)
    record = train_run(
        config,
        data,
        args.model,
        args.treatment,
        args.size,
        args.seed,
        args.lr,
        args.wd,
        args.stage,
        device,
        args.max_steps,
    )
    print(record)
    runlog.append_run(args.log or Path(config["results_csv"]), record)


if __name__ == "__main__":
    main()
