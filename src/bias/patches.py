"""Trains ViTs with other patch sizes, on real and pixel-shuffled images.

The plain ViT cuts the image into 4x4 patches. These runs use 8 and 2
pixel patches instead, on real images and on images with all pixels
shuffled one fixed way, plus a 1-pixel ViT: every pixel is a token, so
nothing is assumed about which pixels belong together. The 1-pixel runs
need a CUDA GPU; on the Mac they take about 10 hours each. Every run uses
augmentation and the plain ViT's tuned settings.

Runs go cheapest first. The script skips runs already logged, so it can be
stopped and restarted.

Usage:
    python -m bias.patches configs/images.yaml
"""

import argparse
from pathlib import Path

from bias import config as config_lib
from bias import grid, scramble, train
from bias.data import tensors

SEED = 0
# Per model: sizes on real images, sizes on shuffled images, micro-batch.
RUNS: dict[str, tuple[tuple[str, ...], tuple[str, ...], int | None]] = {
    "vit_patch8": (("5%", "10%", "20%"), ("10%",), None),
    "vit_patch2": (("5%", "10%", "20%"), ("10%",), None),
    "vit_patch1": (("5%", "10%", "20%", "100%"), ("10%",), None),
}


def main() -> None:
    """Trains every model at its sizes, skipping the runs already logged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = config_lib.load_config(parser.parse_args().config)
    settings = grid.best_settings(grid.read_runs(config, "tune"))
    device = train.pick_device()
    data = tensors.load_tensors(config, device)
    shuffled = scramble.scrambled(data, "pixels")
    log = Path(config["results_dir"]) / f"{config['dataset']}_patches.csv"
    for model, (real_sizes, shuffled_sizes, micro) in RUNS.items():
        for stage, images, sizes in (
            ("patches", data, real_sizes),
            ("patches_scramble_pixels", shuffled, shuffled_sizes),
        ):
            jobs = [
                train.Job(
                    model,
                    "aug",
                    size,
                    SEED,
                    *grid.tuned(config, settings, "vit", "aug", size),
                )
                for size in sizes
            ]
            grid.run_jobs(
                config,
                images,
                grid.unlogged(config, stage, jobs),
                stage,
                log,
                device,
                micro_batch=micro,
            )


if __name__ == "__main__":
    main()
