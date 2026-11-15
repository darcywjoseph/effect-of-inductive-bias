"""Domain-specific CNNs."""

import timm
import torch


def resnet18_cifar(num_classes: int, in_chans: int = 3) -> torch.nn.Module:
    """Builds the CIFAR variant of ResNet-18.

    The ImageNet stem (7x7 stride-2 conv, max-pool) is replaced by a 3x3
    stride-1 conv, so a 32x32 input keeps its resolution into stage one.

    Args:
        num_classes: Output classes.
        in_chans: Input channels.

    Returns:
        The model, randomly initialised.
    """
    model: torch.nn.Module = timm.create_model(
        "resnet18", pretrained=False, num_classes=num_classes, in_chans=in_chans
    )
    model.conv1 = torch.nn.Conv2d(in_chans, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model
