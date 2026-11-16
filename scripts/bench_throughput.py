"""Measures training steps per second for each image model, fp32 and bf16."""

import time

import torch

from bias import train
from bias.models import registry


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _rate(name: str, bf16: bool, dev: torch.device, steps: int = 100) -> float:
    x = torch.randn(128, 3, 32, 32, device=dev)
    y = torch.randint(0, 100, (128,), device=dev)
    model = registry.build(name, "cifar100").to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    start = 0.0
    for i in range(steps + 10):
        if i == 10:
            _sync(dev)
            start = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    _sync(dev)
    return steps / (time.perf_counter() - start)


def main() -> None:
    """Prints steps/s per model and precision at batch 128."""
    dev = train.pick_device()
    for name in ("cnn", "vit", "mlp"):
        for bf16 in (False, True):
            rate = _rate(name, bf16, dev)
            print(f"{name} {'bf16' if bf16 else 'fp32'} {rate:.1f} steps/s", flush=True)


if __name__ == "__main__":
    main()
