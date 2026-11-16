"""Low-bias model A: ViT-Tiny."""

import timm
import torch


def vit_tiny(
    num_classes: int,
    img_size: int | tuple[int, int],
    patch_size: int,
    in_chans: int,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Builds a ViT-Tiny (192 wide, 12 blocks, 3 heads).

    With pretrained=True the weights are timm's default ViT-Tiny checkpoint
    (augreg_in21k_ft_in1k: ImageNet-21k, then fine-tuned on ImageNet-1k).
    timm resamples the 16-pixel patch embedding to patch_size and
    interpolates the position embeddings to the new grid, so the network
    is the same shape as the one trained from scratch. The head is new.

    Args:
        num_classes: Output classes.
        img_size: Input height and width.
        patch_size: Side of each square patch.
        in_chans: Input channels.
        pretrained: Start from ImageNet weights instead of random ones.

    Returns:
        The model.
    """
    model: torch.nn.Module = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    return model
