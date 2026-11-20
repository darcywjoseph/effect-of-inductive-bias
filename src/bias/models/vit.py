"""ViT-Tiny for 32x32 images, in 4x4 patches, and variants of it.

Each variant changes one thing, and several can be combined:

- "fixedpos": the position embeddings are fixed 2-D sinusoids.
- "local": every patch attends only to itself and its eight neighbours
  (and the class token), in every block.
- "convstem": the linear patch embedding becomes two strided 3x3
  convolutions.
"""

import math

import timm
import torch

IMG_SIZE, PATCH_SIZE = 32, 4
VARIANTS = ("fixedpos", "local", "convstem")
# Pretrained starts: ImageNet-21k weights, and DeiT-Tiny's from ImageNet-1k.
WEIGHTS = {
    "imagenet": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "imagenet1k": "deit_tiny_patch16_224.fb_in1k",
}


def vit_tiny(num_classes: int, weights: str | None = None) -> torch.nn.Module:
    """Builds a ViT-Tiny (192 wide, 12 blocks, 3 heads).

    With weights set, timm resamples the 16-pixel patch embedding to
    PATCH_SIZE and interpolates the position embeddings to the new grid.
    The head is new.

    Args:
        num_classes: Output classes.
        weights: None for random weights, or a key of WEIGHTS.

    Returns:
        The model.
    """
    model: torch.nn.Module = timm.create_model(
        WEIGHTS[weights] if weights else "vit_tiny_patch16_224",
        pretrained=weights is not None,
        num_classes=num_classes,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
    )
    return model


def sincos_2d(side: int, dim: int) -> torch.Tensor:
    """Returns fixed 2-D sinusoidal position embeddings.

    Args:
        side: Patches per side of the square grid.
        dim: Embedding width, divisible by 4.

    Returns:
        A (side * side, dim) tensor: half the width encodes the row and
        half the column.
    """
    freqs = torch.exp(-math.log(10000.0) * torch.arange(dim // 4) / (dim // 4))
    angles = torch.arange(side)[:, None] * freqs[None, :]
    one_axis = torch.cat([angles.sin(), angles.cos()], dim=1)
    rows = one_axis[:, None, :].expand(side, side, -1)
    cols = one_axis[None, :, :].expand(side, side, -1)
    return torch.cat([rows, cols], dim=2).reshape(side * side, dim)


def neighbour_mask(side: int) -> torch.Tensor:
    """Returns which tokens may attend to which under local attention.

    Args:
        side: Patches per side of the square grid.

    Returns:
        A boolean (1 + side * side) square matrix, True where attention is
        allowed. Token 0 is the class token, which sees and is seen by all.
    """
    rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    rows, cols = rows.reshape(-1), cols.reshape(-1)
    near = ((rows[:, None] - rows[None, :]).abs() <= 1) & (
        (cols[:, None] - cols[None, :]).abs() <= 1
    )
    mask = torch.ones(1 + side * side, 1 + side * side, dtype=torch.bool)
    mask[1:, 1:] = near
    return mask


class _LocalViT(torch.nn.Module):
    """A ViT whose patches attend only to their 3x3 neighbourhood."""

    def __init__(self, vit: torch.nn.Module, side: int) -> None:
        super().__init__()
        self.vit = vit
        self.mask: torch.Tensor
        self.register_buffer("mask", neighbour_mask(side), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit.forward_features(x, attn_mask=self.mask)  # type: ignore[operator]
        out: torch.Tensor = self.vit.forward_head(features)  # type: ignore[operator]
        return out


def vit_tiny_variant(variant: str, num_classes: int) -> torch.nn.Module:
    """Builds a ViT-Tiny variant from random weights.

    Args:
        variant: One of VARIANTS, or several joined by "+".
        num_classes: Output classes.

    Returns:
        The model.

    Raises:
        ValueError: If the variant is unknown.
    """
    parts = set(variant.split("+"))
    if not parts <= set(VARIANTS):
        raise ValueError(f"unknown ViT variant: {variant=}")
    model = vit_tiny(num_classes)
    side = IMG_SIZE // PATCH_SIZE
    pos = model.get_parameter("pos_embed")
    width = pos.shape[-1]
    if "fixedpos" in parts:
        with torch.no_grad():
            pos.zero_()
            pos[0, 1:] = sincos_2d(side, width)
        pos.requires_grad_(False)
    if "convstem" in parts:
        model.get_submodule("patch_embed").proj = torch.nn.Sequential(
            torch.nn.Conv2d(3, width // 4, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(width // 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(width // 4, width // 2, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(width // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(width // 2, width, 1),
        )
    return _LocalViT(model, side) if "local" in parts else model
