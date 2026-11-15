# effect-of-inductive-bias

How many extra labelled examples does a low-bias model (ViT, MLP) need to
match a domain-specific one (CNN), and how much of that gap do augmentation,
self-supervision and a pretrained start buy back? Every answer is reported as
"times the data".

## Setup

```sh
uv sync
uv run python -m effect_of_inductive_bias.data.download --root data     # CIFAR-100, Speech Commands v2
```

The hyperspectral `.mat` files (Indian Pines, Pavia University) must be placed
in `data/hsi/` by hand for now; the ehu.eus server refuses scripted downloads.
Expected names: `Indian_pines_corrected.mat`, `Indian_pines_gt.mat`,
`PaviaU.mat`, `PaviaU_gt.mat`.

## Commands

```sh
# Per-class split sizes at every grid point, and the hyperspectral block maps
uv run python -m effect_of_inductive_bias.report_splits configs/*.yaml

# Checks
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy && uv run pytest
```

## Protocol notes

- Subsets are nested per seed: each class has one random order per seed and
  every size takes a prefix of it.
- Hyperspectral splits assign whole blocks to train, val or test. A pixel is
  kept only if its whole patch lies inside blocks of its own split, so no
  test patch shares a pixel with a training patch.
- The exchange-rate rule (isotonic fit, log2 interpolation, first crossing,
  censored bounds, seed bootstrap) lives in `src/effect_of_inductive_bias/exchange_rate.py`
  and the targets are fixed in each config.
