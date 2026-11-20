from typing import Any

import pytest
import torch

from bias.models import registry, vit


def test_neighbour_mask_allows_the_3x3_neighbourhood_and_the_class_token() -> None:
    mask = vit.neighbour_mask(8)
    assert mask[0].all()
    assert mask[:, 0].all()
    corner, centre = mask[1, 1:].view(8, 8), mask[1 + 3 * 8 + 3, 1:].view(8, 8)
    assert corner.sum() == 4
    assert corner[:2, :2].all()
    assert centre.sum() == 9
    assert centre[2:5, 2:5].all()


def test_fixed_positions_differ_for_every_patch_and_are_not_trained() -> None:
    model = registry.build("vit_fixedpos")
    pos = model.get_parameter("pos_embed")
    assert not pos.requires_grad
    assert len(torch.unique(pos[0, 1:], dim=0)) == 64


@pytest.mark.parametrize("name", ["vit_local", "vit_convstem", "vit_fixedpos+local"])
def test_variants_classify_a_batch(name: str) -> None:
    model = registry.build(name).eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)


def test_one_local_block_spreads_a_change_by_one_patch() -> None:
    torch.manual_seed(0)
    model = registry.build("vit_local").eval()
    inner: Any = model.get_submodule("vit")

    def tokens(images: torch.Tensor) -> torch.Tensor:
        embedded = inner._pos_embed(inner.patch_embed(images))
        out: torch.Tensor = inner.blocks[0](embedded, attn_mask=model.mask)
        return out[:, 1:].view(8, 8, -1)

    x = torch.randn(1, 3, 32, 32)
    moved = x.clone()
    moved[:, :, 28:, 28:] += 1.0
    with torch.no_grad():
        before, after = tokens(x), tokens(moved)
    changed = (before != after).any(-1)
    assert changed[6:, 6:].all()
    assert not changed[:6].any()
    assert not changed[:, :6].any()


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ViT variant"):
        registry.build("vit_global")
