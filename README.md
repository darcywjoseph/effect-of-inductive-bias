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

# Checks
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy && uv run pytest
```

## Experiment notes

- Subsets are nested per seed: each class has one random order per seed and
  every size takes a prefix of it.
- The exchange-rate rule (isotonic fit, log2 interpolation, first crossing,
  censored bounds, seed bootstrap) lives in `src/bias/exchange_rate.py`
  and the targets are fixed in each config.

## Data

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009).
