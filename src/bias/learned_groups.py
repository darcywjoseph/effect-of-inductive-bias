"""Trains the pixel-grouping ViTs (bias.models.grouping) at chosen sizes.

Every run uses augmentation and the plain ViT's tuned settings, and saves
its best weights so the learned groups can be drawn later. Runs already
logged under the stage are skipped, so the script can be restarted.

Usage:
    python -m bias.learned_groups configs/images.yaml \
        --models group_grid group_learned group_fixedpos --sizes 5% 10% 20%
"""

import argparse
from pathlib import Path

from bias import config as config_lib
from bias import grid, runlog, train
from bias.data import tensors

SEED = 0
STAGE = "grouping"


def main() -> None:
    """Trains every model at every size, smallest size first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sizes", nargs="+", required=True)
    parser.add_argument("--log", default="grouping", help="Log name suffix.")
    args = parser.parse_args()
    config = config_lib.load_config(args.config)
    settings = grid.best_settings(grid.read_runs(config, "tune"))
    device = train.pick_device()
    data = tensors.load_tensors(config, device)
    log = Path(config["results_dir"]) / f"{config['dataset']}_{args.log}.csv"
    checkpoints = Path("checkpoints") / "grouping"
    for size in args.sizes:
        for model in args.models:
            job = train.Job(
                model,
                "aug",
                size,
                SEED,
                *grid.tuned(config, settings, "vit", "aug", size),
            )
            if not grid.unlogged(config, STAGE, [job]):
                continue
            record = train.train_run(
                config,
                data,
                job,
                STAGE,
                device,
                save_to=checkpoints / f"{model}_{size.rstrip('%')}pc_s{SEED}.pt",
            )
            print(record, flush=True)
            runlog.append_run(log, record)


if __name__ == "__main__":
    main()
