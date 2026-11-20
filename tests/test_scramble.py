import numpy as np
import pytest
import torch

from bias import inside, scramble
from bias.models import registry

SIDE, PATCH = scramble.SIDE, scramble.PATCH


def _blocks(values: np.ndarray, block: int) -> np.ndarray:
    """Splits a SIDE x SIDE array into rows of one block's values each."""
    per_side = SIDE // block
    split = values.reshape(per_side, block, per_side, block).transpose(0, 2, 1, 3)
    return split.reshape(per_side**2, block**2)


@pytest.mark.parametrize("kind", ["pixels", "patches", "local8", "jigsaw2"])
def test_permutation_moves_every_pixel_once(kind: str) -> None:
    source = scramble.permutation(kind)
    assert sorted(source.tolist()) == list(range(SIDE * SIDE))


def test_unknown_scrambles_are_rejected() -> None:
    for kind in ("rows", "jigsaw", "local0"):
        with pytest.raises(ValueError, match="unknown scramble"):
            scramble.permutation(kind)
    with pytest.raises(ValueError, match="must divide"):
        scramble.permutation("local5")


def test_local_scramble_keeps_every_pixel_in_its_block() -> None:
    home = _blocks(np.arange(SIDE * SIDE), 8)
    source = _blocks(scramble.permutation("local8"), 8)
    assert np.array_equal(np.sort(source, axis=1), home)
    assert not np.array_equal(source, home)


def test_jigsaw_moves_whole_blocks_with_their_insides_intact() -> None:
    home = _blocks(np.arange(SIDE * SIDE), 2)
    source = _blocks(scramble.permutation("jigsaw2"), 2)
    assert sorted(map(tuple, source)) == sorted(map(tuple, home))
    assert not np.array_equal(source, home)


def test_patch_scramble_reorders_every_patch_the_same_way() -> None:
    source = _blocks(scramble.permutation("patches"), PATCH)
    rows, cols = source // SIDE, source % SIDE
    assert (rows // PATCH == rows[:, :1] // PATCH).all()
    assert (cols // PATCH == cols[:, :1] // PATCH).all()
    assert (rows % PATCH == rows[0] % PATCH).all()
    assert (cols % PATCH == cols[0] % PATCH).all()


def test_mlp_with_reordered_weights_cannot_see_the_pixel_scramble() -> None:
    torch.manual_seed(0)
    model = registry.build("mlp").eval()
    source = torch.from_numpy(scramble.permutation("pixels"))
    x = torch.randn(4, 3, SIDE, SIDE)
    want = model(x)
    embed = model.get_parameter("embed.weight")
    weight = embed.data.view(-1, 3, SIDE * SIDE)
    embed.data = weight[:, :, source].reshape(weight.shape[0], -1)
    assert torch.allclose(model(scramble.apply(x, source)), want, atol=1e-5)


def test_vit_with_reordered_weights_cannot_see_the_patch_scramble() -> None:
    torch.manual_seed(0)
    model = registry.build("vit").eval()
    pos = model.get_parameter("pos_embed")
    proj = model.get_parameter("patch_embed.proj.weight")
    with torch.no_grad():
        pos.normal_(std=0.02)
    source = scramble.permutation("patches")
    x = torch.randn(4, 3, SIDE, SIDE)
    want = model(x)
    # The reordering inside a patch, read off the first one.
    first = _blocks(source, PATCH)[0]
    inner = ((first // SIDE) % PATCH) * PATCH + (first % SIDE) % PATCH
    with torch.no_grad():
        flat = proj.view(proj.shape[0], 3, PATCH * PATCH)
        proj.copy_(flat[:, :, torch.from_numpy(inner)].view_as(proj))
        pos[:, 1:] = pos[:, 1:][:, torch.from_numpy(inside.true_position())].clone()
    got = model(scramble.apply(x, torch.from_numpy(source)))
    assert torch.allclose(got, want, atol=1e-4)
