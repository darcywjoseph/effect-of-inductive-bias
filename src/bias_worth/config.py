"""Loads a modality's YAML config."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Reads a config file and checks the keys every config needs.

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
