"""ResNet-18 for 32x32 images."""

import timm
import torch

# ImageNet-1k weights from the "ResNet strikes back" A1 recipe.
IMAGENET_1K = "resnet18.a1_in1k"


def resnet18_cifar(num_classes: int, pretrained: bool = False) -> torch.nn.Module:
    """Builds the CIFAR variant of ResNet-18.

    The ImageNet stem (7x7 stride-2 conv, max-pool) would cut a 32x32
    input to 8x8 before stage one. From scratch, the stem conv becomes a
    3x3 stride-1 conv. From ImageNet weights, the pretrained 7x7 conv is
    kept at stride 1. Both drop the max-pool.

    Args:
        num_classes: Output classes.
        pretrained: Start from ImageNet-1k weights; the head is new.

    Returns:
        The model.
    """
    model: torch.nn.Module = timm.create_model(
        IMAGENET_1K if pretrained else "resnet18",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    if pretrained:
        model.conv1.stride = (1, 1)  # type: ignore[method-assign,assignment]
    else:
        model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model
