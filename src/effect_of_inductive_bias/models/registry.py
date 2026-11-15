"""Builds a model by name for a dataset."""

import torch

from effect_of_inductive_bias.models import cnn, mlp, vit


def build(name: str, dataset: str) -> torch.nn.Module:
    """Builds one of the study's models for a dataset.

    Args:
        name: "cnn", "vit" or "mlp".
        dataset: Dataset name from the config.

    Returns:
        The model, randomly initialised.

    Raises:
        ValueError: If the pair is not defined yet.
    """
    if dataset == "cifar100":
        if name == "cnn":
            return cnn.resnet18_cifar(100)
        if name == "vit":
            return vit.vit_tiny(100, img_size=32, patch_size=4, in_chans=3)
        if name == "mlp":
            return mlp.MLP((3, 32, 32), 100)
    raise ValueError(f"no model defined for {name=} on {dataset=}")


def count_params(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters.

    Args:
        model: Any module.

    Returns:
        Trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
