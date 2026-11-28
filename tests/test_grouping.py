import pytest
import torch

from bias.models import grouping, registry


@pytest.mark.parametrize("kind", grouping.KINDS)
def test_grouping_models_classify_a_batch(kind: str) -> None:
    model = registry.build(f"group_{kind}").eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)


@pytest.mark.parametrize("kind", grouping.KINDS)
def test_every_pixel_shares_itself_out_completely(kind: str) -> None:
    model = grouping.GroupViT(100, kind).eval()
    with torch.no_grad():
        weights = model.assignments(model.pixel_features(torch.randn(2, 3, 32, 32)))
    assert weights.shape == (2, 1024, 64)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 1024))


def test_grid_groups_are_the_4x4_blocks() -> None:
    blocks = grouping.grid_assignment().argmax(1).view(32, 32)
    assert (blocks[:4, :4] == 0).all()
    assert (blocks[:4, 4:8] == 1).all()
    assert (blocks[4:8, :4] == 8).all()
    assert (grouping.grid_assignment().sum(0) == 16).all()


def test_learned_grouping_is_blind_to_a_fixed_pixel_permutation() -> None:
    torch.manual_seed(0)
    model = grouping.GroupViT(100, "learned").eval()
    permuted = grouping.GroupViT(100, "learned").eval()
    permuted.load_state_dict(model.state_dict())
    source = torch.randperm(1024)
    with torch.no_grad():
        permuted.pixel_pos[0] = model.pixel_pos[0][source]
        x = torch.randn(2, 3, 32, 32)
        shuffled = x.flatten(2)[:, :, source].reshape(x.shape)
        torch.testing.assert_close(permuted(shuffled), model(x))


def test_unknown_grouping_is_rejected() -> None:
    with pytest.raises(ValueError):
        registry.build("group_superpixel")
