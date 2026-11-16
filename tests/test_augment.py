import torch

from bias import augment


def test_sampler_walks_whole_epochs() -> None:
    order = list(augment._EpochSampler(n=10, num_samples=25, seed=0))
    assert len(order) == 25
    assert sorted(order[:10]) == list(range(10))
    assert sorted(order[10:20]) == list(range(10))


def test_batches_keep_shape_dtype_and_labels() -> None:
    x = torch.randint(0, 256, (20, 3, 32, 32), dtype=torch.uint8)
    y = torch.arange(20)
    batches = list(
        augment.augmented_batches(
            x, y, augment.cifar_transform(), batch=8, steps=5, seed=0, workers=0
        )
    )
    assert len(batches) == 5
    for xb, yb in batches:
        assert xb.shape == (8, 3, 32, 32)
        assert xb.dtype == torch.uint8
        assert set(yb.tolist()) <= set(range(20))


def test_same_seed_gives_same_batches() -> None:
    x = torch.randint(0, 256, (16, 3, 32, 32), dtype=torch.uint8)
    y = torch.arange(16)

    def first() -> torch.Tensor:
        torch.manual_seed(0)
        batches = augment.augmented_batches(
            x, y, augment.cifar_transform(), batch=4, steps=1, seed=3, workers=0
        )
        return next(iter(batches))[0]

    assert torch.equal(first(), first())
