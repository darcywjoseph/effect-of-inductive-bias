"""Trains the ViT variants of bias.models.vit along part of the grid.

Each run uses augmentation and the plain ViT's tuned settings.

Usage:
    python -m bias.variants configs/images.yaml
"""

import argparse
from pathlib import Path

from bias import config as config_lib
from bias import grid, train
from bias.data import tensors

SEED = 0
# The sizes each variant is trained at.
RUNS = {
    "fixedpos": ("5%", "10%", "20%"),
    "local": ("5%", "10%", "20%"),
    "convstem": ("5%", "10%", "20%"),
    "local+convstem": ("10%", "20%"),
    "fixedpos+local+convstem": ("10%",),
}


def main() -> None:
    """Trains every variant at its sizes, skipping the runs already logged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = config_lib.load_config(parser.parse_args().config)
    settings = grid.best_settings(grid.read_runs(config, "tune"))
    jobs = [
        train.Job(
            f"vit_{variant}",
            "aug",
            size,
            SEED,
            *grid.tuned(config, settings, "vit", "aug", size),
        )
        for variant, sizes in RUNS.items()
        for size in sizes
    ]
    device = train.pick_device()
    grid.run_jobs(
        config,
        tensors.load_tensors(config, device),
        grid.unlogged(config, "variants", jobs),
        "variants",
        Path(config["results_dir"]) / f"{config['dataset']}_variants.csv",
        device,
    )


if __name__ == "__main__":
    main()
