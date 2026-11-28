"""Builds a model by name."""

import torch

from bias.models import cnn, grouping, mlp, vit


def build(
    name: str, weights: str | None = None, num_classes: int = 100
) -> torch.nn.Module:
    """Builds a model for 32x32 images.

    Args:
        name: "cnn", "vit", "mlp", "vit_<variant>" (see models.vit), or
          "group_<kind>" (see models.grouping).
        weights: None for random weights, "imagenet" (ImageNet-21k, ViT
          only) or "imagenet1k" (ImageNet-1k, ViT or CNN).
        num_classes: Output classes: 100 for CIFAR-100, 1,000 for
          ImageNet-32.

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
        return cnn.resnet18_cifar(num_classes, pretrained=weights is not None)
    if name == "vit":
        return vit.vit_tiny(num_classes, weights=weights)
    if name == "mlp":
        return mlp.MLP((3, 32, 32), num_classes)
    if name.startswith("group_"):
        return grouping.GroupViT(num_classes, name.removeprefix("group_"))
    if name.startswith("vit_"):
        return vit.vit_tiny_variant(name.removeprefix("vit_"), num_classes)
    raise ValueError(f"unknown model: {name=}")


def count_params(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters.

    Args:
        model: Any module.

    Returns:
        Trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
