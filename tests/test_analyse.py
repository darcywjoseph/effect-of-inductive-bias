import math

import pandas as pd
import pytest

from bias import analyse

GRID = ["1%", "2%", "5%", "10%", "20%", "50%", "100%"]
N = {
    "1%": 450,
    "2%": 900,
    "5%": 2250,
    "10%": 4500,
    "20%": 9000,
    "50%": 22500,
    "100%": 45000,
}
# Each setup's curve is the CNN+aug curve fed 1/shift of the data.
SHIFT = {
    ("cnn", "none"): 2.0,
    ("cnn", "aug"): 1.0,
    ("cnn", "lejepa"): 1.0,
    ("vit", "none"): 8.0,
    ("vit", "aug"): 4.0,
    ("vit", "lejepa"): 2.0,
    ("vit", "imagenet"): 1.0,
    ("mlp", "none"): 32.0,
    ("mlp", "aug"): 16.0,
    ("mlp", "lejepa"): 8.0,
}
CONFIG = {
    "dataset": "fake",
    "grid": GRID,
    "targets": {
        "reference": {"model": "cnn", "treatment": "aug"},
        "sizes": ["10%", "50%"],
    },
    "bootstrap": {"draws": 200, "level": 0.95},
}


def _score(n: float) -> float:
    return 0.05 * math.log2(n)


def _rows() -> pd.DataFrame:
    rows = []
    for (model, treatment), shift in SHIFT.items():
        for size in GRID:
            for seed in range(3):
                noise = 0.001 * (seed - 1)
                rows.append(
                    {
                        "model": model,
                        "treatment": treatment,
                        "size": size,
                        "seed": seed,
                        "n_labelled": N[size],
                        "test": _score(N[size] / shift) + noise,
                        "best_val": _score(N[size] / shift) + noise,
                    }
                )
    return pd.DataFrame(rows)


def test_exchange_rates_recover_the_shifts() -> None:
    result = analyse.table(CONFIG, _rows())
    at10 = result[result.target_size == "10%"].set_index("question")
    assert at10.loc["CNN bias vs ViT, no aug", "E"] == pytest.approx(4.0, rel=0.02)
    assert at10.loc["Augmentation, ViT", "E"] == pytest.approx(2.0, rel=0.02)
    assert at10.loc["Pretrained start, ViT", "E"] == pytest.approx(4.0, rel=0.02)
    # The best ViT by validation is the ImageNet start, which matches the CNN.
    assert at10.loc["CNN lead left over best ViT", "reference"] == "vit+imagenet"
    assert at10.loc["CNN lead left over best ViT", "E"] == pytest.approx(1.0, rel=0.02)


def test_curves_that_never_reach_the_target_give_lower_bounds() -> None:
    result = analyse.table(CONFIG, _rows())
    at50 = result[result.target_size == "50%"].set_index("question")
    # The MLP without aug needs 32x the data: past the grid at the 50% target.
    row = at50.loc["CNN bias vs MLP, no aug"].to_dict()
    assert row["kind"] == "lower"
    assert analyse.describe(row).startswith("more than")
