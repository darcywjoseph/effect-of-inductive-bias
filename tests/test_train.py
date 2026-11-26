import json
from pathlib import Path

import pytest
import torch

from bias import augment, train
from bias import config as config_lib
from bias.data import tensors
from bias.models import registry

IMAGES = config_lib.load_config(Path("configs/images.yaml"))


@pytest.mark.parametrize("micro", [4, 5])
def test_micro_batches_give_the_whole_batch_gradient(micro: int) -> None:
    torch.manual_seed(0)
    model = registry.build("vit_patch8")
    x, y = torch.randn(16, 3, 32, 32), torch.arange(16) % 100
    grads = []
    for size in (None, micro):
        model.zero_grad(set_to_none=True)
        train.backward(model, x, y, size)
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    for whole, pieces in zip(*grads, strict=True):
        torch.testing.assert_close(pieces, whole, rtol=1e-4, atol=1e-6)


def _tiny_data() -> tensors.TensorData:
    generator = torch.Generator().manual_seed(0)
    raw = torch.randint(
        0, 256, (260, 3, 32, 32), dtype=torch.uint8, generator=generator
    )
    labels = torch.arange(260) % 100

    def prepare(batch: torch.Tensor) -> torch.Tensor:
        return batch.float().div(255.0)

    return tensors.TensorData(
        pool_x=prepare(raw[:200]),
        pool_y=labels[:200],
        val_x=prepare(raw[200:230]),
        val_y=labels[200:230],
        test_x=prepare(raw[230:]),
        test_y=labels[230:],
        num_classes=100,
        pool_raw=raw[:200],
        prepare=prepare,
        augment_example=augment.cifar_transform(),
        ssl_view=augment.cifar_ssl_view(),
    )


@pytest.mark.parametrize("treatment", ["none", "aug"])
def test_train_run_logs_a_reproducible_run(treatment: str, tmp_path: Path) -> None:
    config = {**IMAGES, "results_dir": str(tmp_path), "workers": 0}
    job = train.Job("mlp", treatment, "50%", 0, 1e-3, 0.05)
    device = torch.device("cpu")
    records = [
        train.train_run(config, _tiny_data(), job, "smoke", device, max_steps=3)
        for _ in range(2)
    ]
    first = records[0]
    assert (first.model, first.treatment, first.size, first.stage) == (
        "mlp",
        treatment,
        "50%",
        "smoke",
    )
    assert (first.n_labelled, first.steps) == (100, 3)
    assert 0.0 <= first.best_val <= 1.0
    assert 0.0 <= first.test <= 1.0
    assert (first.best_val, first.test) == (records[1].best_val, records[1].test)
    lines = (tmp_path / "history" / "cifar100_smoke.jsonl").read_text().splitlines()
    history = json.loads(lines[0])
    assert [step for step, _ in history["val"]] == [1, 2, 3]
    assert max(score for _, score in history["val"]) == first.best_val


def test_train_run_rejects_an_unknown_treatment(tmp_path: Path) -> None:
    config = {**IMAGES, "results_dir": str(tmp_path)}
    job = train.Job("mlp", "mixup", "50%", 0, 1e-3, 0.05)
    with pytest.raises(ValueError, match="unknown treatment"):
        train.train_run(config, _tiny_data(), job, "smoke", torch.device("cpu"))
