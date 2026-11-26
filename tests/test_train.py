import pytest
import torch

from bias import train
from bias.models import registry


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
