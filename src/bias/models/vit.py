"""Low-bias model A: ViT-Tiny."""

import timm
import torch

# Pretrained starts. ImageNet-21k is the brief's treatment; ImageNet-1k
# (DeiT-Tiny, the same 192/12/3 network) matches the data the CNN's
# ImageNet weights saw, for the like-for-like comparison.
WEIGHTS = {
    "imagenet": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "imagenet1k": "deit_tiny_patch16_224.fb_in1k",
}


def vit_tiny(
    num_classes: int,
    img_size: int | tuple[int, int],
    patch_size: int,
    in_chans: int,
    weights: str | None = None,
) -> torch.nn.Module:
    """Builds a ViT-Tiny (192 wide, 12 blocks, 3 heads).

    With weights set, timm resamples the 16-pixel patch embedding to
    patch_size and interpolates the position embeddings to the new grid,
    so the network is the same shape as the one trained from scratch. The
    head is new.

    Args:
        num_classes: Output classes.
        img_size: Input height and width.
        patch_size: Side of each square patch.
        in_chans: Input channels.
        weights: None for random weights, or a key of WEIGHTS.

    Returns:
        The model.
    """
    model: torch.nn.Module = timm.create_model(
        WEIGHTS[weights] if weights else "vit_tiny_patch16_224",
        pretrained=weights is not None,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    return model
