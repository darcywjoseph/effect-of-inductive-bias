"""Builds a model by name."""

import torch

from bias.models import cnn, mlp, vit


def build(name: str, weights: str | None = None) -> torch.nn.Module:
    """Builds a model for CIFAR-100.

    Args:
        name: "cnn", "vit" or "mlp".
        weights: None for random weights, "imagenet" (ImageNet-21k, ViT
          only) or "imagenet1k" (ImageNet-1k, ViT or CNN).

    Returns:
        The model.

    Raises:
        ValueError: If the name is unknown, or the model has no such
          pretrained weights.
    """
    allowed = {"vit": {"imagenet", "imagenet1k"}, "cnn": {"imagenet1k"}}
    if weights is not None and weights not in allowed.get(name, set()):
        raise ValueError(f"no pretrained start for {name=}: {weights=}")
    if name == "cnn":
        return cnn.resnet18_cifar(100, pretrained=weights is not None)
    if name == "vit":
        return vit.vit_tiny(100, weights=weights)
    if name == "mlp":
        return mlp.MLP((3, 32, 32), 100)
    raise ValueError(f"unknown model: {name=}")


def count_params(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters.

    Args:
        model: Any module.

    Returns:
        Trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
