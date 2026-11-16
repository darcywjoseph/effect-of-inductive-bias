# Effect of Inductive Bias

Investigation into how many extra labelled samples a low-bias model (ViT, MLP) needs to
match architectures with specific inductive bias (CNN).

## Setup

```sh
uv sync
uv run python -m bias.data.download --root data  # CIFAR-100, Speech Commands v2, Salinas, Pavia University
```

## Commands

```sh
# Per-class split sizes at every grid point, and the hyperspectral block maps
uv run python -m bias.report_splits configs/*.yaml

# Checks
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy && uv run pytest
```

## Experiment notes

- Subsets are nested per seed: each class has one random order per seed and
  every size takes a prefix of it.
- Hyperspectral splits assign whole blocks to train, val or test. A pixel is
  kept only if its whole patch lies inside blocks of its own split, so no
  test patch shares a pixel with a training patch.
- The exchange-rate rule (isotonic fit, log2 interpolation, first crossing,
  censored bounds, seed bootstrap) lives in `src/bias/exchange_rate.py`
  and the targets are fixed in each config.

## Data

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009).
- [Speech Commands v2](https://arxiv.org/abs/1804.03209) (Warden, 2018), CC BY 4.0.
- Salinas (AVIRIS, NASA JPL) and Pavia University (ROSIS; courtesy of Prof.
  Paolo Gamba, University of Pavia), from the UPV/EHU
  [Hyperspectral Remote Sensing Scenes](https://ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
  page, downloaded via a [Hugging Face mirror](https://huggingface.co/datasets/danaroth).
