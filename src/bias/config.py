"""Loads a YAML config."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Reads a config file and checks its required keys.

    Args:
        path: YAML config path.

    Returns:
        The parsed config.

    Raises:
        ValueError: If a required key is missing.
    """
    with path.open() as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    required = {"dataset", "data_root", "grid", "seeds", "metric", "targets"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return config


def lejepa_checkpoint(config: dict[str, Any], model_name: str, seed: int) -> Path:
    """Returns where a LeJEPA-pretrained backbone is stored.

    Args:
        config: Parsed config with a lejepa section.
        model_name: "cnn", "vit" or "mlp".
        seed: Pretraining seed.

    Returns:
        The .pt path.
    """
    root = Path(config["lejepa"]["checkpoint_dir"])
    return root / f"{config['dataset']}_{model_name}_s{seed}.pt"
