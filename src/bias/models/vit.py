"""ViT-Tiny for 32x32 images, in 4x4 patches."""

import timm
import torch

IMG_SIZE, PATCH_SIZE = 32, 4
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
