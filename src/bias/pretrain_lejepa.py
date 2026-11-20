"""LeJEPA self-supervised pretraining on the full pool, labels removed.

LeJEPA (Balestriero and LeCun, arXiv:2511.08544) pulls every view of an
image towards the mean embedding of its views and keeps the embeddings
close to an isotropic Gaussian with SIGReg. There is no teacher, EMA or
stop-gradient. This follows the authors' minimal reference
(github.com/galilai-group/lejepa, MINIMAL.md): SIGReg on the projector
output, 256 random slices redrawn each step, a 17-knot Epps-Pulley test on
[-3, 3], and the mean over all views as the prediction target.

Views come from augment.cifar_ssl_view.

Usage:
    python -m bias.pretrain_lejepa configs/images.yaml --model vit --seed 0
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data
from torchvision.ops import MLP

from bias import augment, train
from bias import config as config_lib
from bias.data import tensors
from bias.models import registry


class _Views(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(
        self, x: torch.Tensor, views: int, transform: augment.Transform
    ) -> None:
        self.x, self.views, self.transform = x, views, transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.stack([self.transform(self.x[i]) for _ in range(self.views)])


class SIGReg(torch.nn.Module):
    """Sketched isotropic Gaussian regulariser with the Epps-Pulley test.

    Projects embeddings onto random unit directions and measures, per
    direction, the weighted distance between the empirical characteristic
    function and that of N(0, 1).
    """

    def __init__(self, slices: int = 256, knots: int = 17) -> None:
        """Builds the integration grid.

        Args:
            slices: Random directions drawn each call.
            knots: Trapezoid knots on [0, 3]; the integrand is symmetric.
        """
        super().__init__()
        self.slices = slices
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Returns the test statistic averaged over views and directions.

        Args:
            proj: Embeddings of shape (views, batch, dim), float32.

        Returns:
            A scalar; 0 means the slices look exactly Gaussian.
        """
        directions = torch.randn(proj.size(-1), self.slices, device=proj.device)
        directions = directions / directions.norm(p=2, dim=0)
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic: torch.Tensor = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def lejepa_loss(
    proj: torch.Tensor, sigreg: SIGReg, lam: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the LeJEPA loss and its two parts.

    Args:
        proj: Projector outputs of shape (views, batch, dim).
        sigreg: The regulariser.
        lam: Weight on SIGReg; the prediction term gets 1 - lam.

    Returns:
        (total, prediction, sigreg) losses.
    """
    prediction = (proj.mean(0) - proj).square().mean()
    regulariser = sigreg(proj)
    return (1 - lam) * prediction + lam * regulariser, prediction, regulariser


@torch.no_grad()
def _features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    # Small chunks: other shards share the GPU.
    out = torch.cat([model(x[i : i + 200]) for i in range(0, len(x), 200)])
    model.train()
    return F.normalize(out.float(), dim=1)


def knn_accuracy(
    model: torch.nn.Module, data: tensors.TensorData, k: int = 20
) -> float:
    """Returns k-nearest-neighbour validation accuracy of frozen features.

    Args:
        model: Backbone with its head removed.
        data: Dataset tensors.
        k: Neighbours that vote.

    Returns:
        Fraction of validation examples classified correctly.
    """
    bank = _features(model, data.pool_x)
    query = _features(model, data.val_x)
    correct = 0
    for start in range(0, len(query), 1000):
        sims = query[start : start + 1000] @ bank.T
        top = sims.topk(k, dim=1).indices
        votes = F.one_hot(data.pool_y[top], data.num_classes).sum(1)
        correct += int((votes.argmax(1) == data.val_y[start : start + 1000]).sum())
    return correct / len(query)


def pretrain(
    config: dict[str, Any],
    data: tensors.TensorData,
    model_name: str,
    seed: int,
    device: torch.device,
    max_steps: int | None = None,
    snapshots: Sequence[int] = (),
) -> dict[str, Any]:
    """Pretrains one backbone with LeJEPA and saves it.

    Args:
        config: Parsed config with a lejepa section.
        data: Dataset tensors; only the unlabelled pool is trained on.
        model_name: "cnn", "vit" or "mlp".
        seed: Seed for the init, the order and the views.
        device: Training device.
        max_steps: Optional override of the step budget, for smoke tests.
        snapshots: Steps before which the backbone is also saved, as
          <checkpoint stem>_step<N>.pt.

    Returns:
        A summary of the run, also written next to the checkpoint.

    Raises:
        FloatingPointError: If the loss diverges.
    """
    cfg = config["lejepa"]
    batch = cfg["batch"]
    total = max_steps or math.ceil(cfg["epochs"] * len(data.pool_raw) / batch)
    weight_decay = cfg["weight_decay"][model_name]
    torch.manual_seed(seed)
    backbone = registry.build(model_name)
    backbone.reset_classifier(0)  # type: ignore[operator]
    dim = int(backbone.num_features)  # type: ignore[arg-type]
    projector = MLP(dim, [2048, 2048, cfg["proj_dim"]], norm_layer=torch.nn.BatchNorm1d)
    backbone, projector = backbone.to(device), projector.to(device)
    sigreg = SIGReg(cfg["slices"]).to(device)
    params = list(backbone.parameters()) + list(projector.parameters())
    optimiser = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=weight_decay)
    workers = config.get("workers", 8)
    loader = torch.utils.data.DataLoader(
        _Views(data.pool_raw, cfg["views"], data.ssl_view),
        batch_size=batch,
        sampler=augment._EpochSampler(len(data.pool_raw), total * batch, seed),
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
        prefetch_factor=4 if workers > 0 else None,
        drop_last=True,
    )
    bf16 = config.get("precision", "fp32") == "bf16"
    history = []
    started = time.perf_counter()
    path = config_lib.lejepa_checkpoint(config, model_name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    for step, raw in enumerate(loader):
        if step in snapshots:
            torch.save(
                backbone.state_dict(), path.with_name(f"{path.stem}_step{step}.pt")
            )
        views = raw.shape[1]
        x = data.prepare(raw.flatten(0, 1))
        for group in optimiser.param_groups:
            group["lr"] = train.lr_at(step, total, cfg["lr"], cfg["warmup_frac"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            proj = projector(backbone(x))
        proj = proj.float().view(-1, views, proj.size(-1)).transpose(0, 1)
        loss, prediction, regulariser = lejepa_loss(proj, sigreg, cfg["lam"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        if step % max(1, total // 50) == 0 or step + 1 == total:
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss diverged at step {step}")
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "prediction": prediction.item(),
                    "sigreg": regulariser.item(),
                }
            )
            print(history[-1], flush=True)
    wall = time.perf_counter() - started

    torch.save(backbone.state_dict(), path)
    summary = {
        "dataset": config["dataset"],
        "model": model_name,
        "seed": seed,
        "steps": total,
        "batch": batch,
        "views": cfg["views"],
        "lam": cfg["lam"],
        "lr": cfg["lr"],
        "weight_decay": weight_decay,
        "knn_val": knn_accuracy(backbone, data),
        "gpu": train.device_name(device),
        "wall_seconds": wall,
        "history": history,
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    """Pretrains one backbone from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--snapshots", type=int, nargs="*", default=[])
    args = parser.parse_args()
    config = config_lib.load_config(args.config)
    device = train.pick_device()
    data = tensors.load_tensors(config, device)
    summary = pretrain(
        config, data, args.model, args.seed, device, args.max_steps, args.snapshots
    )
    print({k: v for k, v in summary.items() if k != "history"})


if __name__ == "__main__":
    main()
