"""Builds a model by name for a dataset."""

import torch

from bias.models import cnn, hsi_cnn, mlp, vit

# Hyperspectral scenes: (bands, classes, CNN 2-D width). Patches are 5x5.
# The widths keep each scene's three models within about 2x in size.
HSI = {"salinas": (204, 16, 128), "pavia_university": (103, 9, 256)}
HSI_SIDE = 5


def build(name: str, dataset: str, weights: str | None = None) -> torch.nn.Module:
    """Builds one of the study's models for a dataset.

    Args:
        name: "cnn", "vit" or "mlp".
        dataset: Dataset name from the config.
        weights: None for random weights, "imagenet" (ImageNet-21k, ViT
          only) or "imagenet1k" (ImageNet-1k, ViT or image/audio CNN).

    Returns:
        The model.

    Raises:
        ValueError: If the pair is not defined yet, or the model has no
          such pretrained weights.
    """
    allowed = {"vit": {"imagenet", "imagenet1k"}, "cnn": {"imagenet1k"}}
    if weights is not None and weights not in allowed.get(name, set()):
        raise ValueError(f"no pretrained start for {name=}: {weights=}")
    if weights is not None and name == "cnn" and dataset in HSI:
        raise ValueError("the hyperspectral CNN has no ImageNet weights")
    if dataset == "cifar100":
        if name == "cnn":
            return cnn.resnet18_cifar(100, pretrained=weights is not None)
        if name == "vit":
            return vit.vit_tiny(
                100, img_size=32, patch_size=4, in_chans=3, weights=weights
            )
        if name == "mlp":
            return mlp.MLP((3, 32, 32), 100)
    if dataset == "speech_commands_v2":
        # Inputs are log-mels of 1 x 40 mels x 100 frames.
        if name == "cnn":
            return cnn.resnet18_cifar(35, in_chans=1, pretrained=weights is not None)
        if name == "vit":
            return vit.vit_tiny(
                35, img_size=(40, 100), patch_size=4, in_chans=1, weights=weights
            )
        if name == "mlp":
            return mlp.MLP((1, 40, 100), 35)
    if dataset in HSI:
        bands, classes, width = HSI[dataset]
        if name == "cnn":
            return hsi_cnn.SpectralSpatialCNN(bands, HSI_SIDE, classes, width)
        if name == "vit":
            # One token per pixel spectrum: 1x1 patches, all bands.
            return vit.vit_tiny(
                classes,
                img_size=HSI_SIDE,
                patch_size=1,
                in_chans=bands,
                weights=weights,
            )
        if name == "mlp":
            return mlp.MLP((bands, HSI_SIDE, HSI_SIDE), classes)
    raise ValueError(f"no model defined for {name=} on {dataset=}")


def count_params(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters.

    Args:
        model: Any module.

    Returns:
        Trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
