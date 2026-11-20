# Effect of Inductive Bias

Investigation into how many extra labelled samples a low-bias model (ViT, MLP) needs to
match architectures with specific inductive bias (CNN).

## Setup

```sh
uv sync
uv run python -m bias.data.download --root data  # CIFAR-100
```

## Commands

```sh
# Per-class split sizes at every grid point
uv run python -m bias.report_splits configs/images.yaml

# Pretraining, tuning and final runs
scripts/run_grid.sh configs/images.yaml

# Scrambled-image runs, ViT variants, and what the ViTs attend to
uv run python -m bias.scramble configs/images.yaml
uv run python -m bias.variants configs/images.yaml
uv run python -m bias.inside configs/images.yaml train
uv run python -m bias.inside configs/images.yaml measure

# Exchange-rate tables and figures
uv run python -m bias.analyse configs/images.yaml
uv run python -m bias.saturation configs/images.yaml
uv run python -m bias.plots configs/images.yaml

# Checks
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy && uv run pytest
```

## Data

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009).
