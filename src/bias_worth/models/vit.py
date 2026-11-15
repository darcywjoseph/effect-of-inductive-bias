"""Low-bias model A: ViT-Tiny."""

import timm
import torch


def vit_tiny(
    num_classes: int, img_size: int | tuple[int, int], patch_size: int, in_chans: int
) -> torch.nn.Module:
    """Builds a ViT-Tiny (192 wide, 12 blocks, 3 heads) from scratch.

    Args:
        num_classes: Output classes.
        img_size: Input height and width.
        patch_size: Side of each square patch.
        in_chans: Input channels.

    Returns:
        The model, randomly initialised.
    """
    model: torch.nn.Module = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    return model
