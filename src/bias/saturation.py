"""Exchange rates across target scores, and error-rate ratios.

1. E against the target score, over every level both setups reach. Lines
   are seed means, the band a seed bootstrap.
2. Error-rate ratios at fixed data: (1 - score) of the reference over that
   of the treatment, at each grid size.

Usage:
    python -m bias.saturation configs/images.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bias import analyse, exchange_rate
from bias import config as config_lib


def rate_curves(
    config: dict[str, Any], rows: pd.DataFrame, levels: int = 30, draws: int = 1000
) -> pd.DataFrame:
    """Computes E at evenly spaced target scores for every comparison.

    Levels run from 5% of the way above chance to the highest score both
    seed-mean curves reach, so every point is an exact crossing.

    Args:
        config: Parsed config, with chance.
        rows: Final runs.
        levels: Target scores per comparison.
        draws: Bootstrap draws per level.

    Returns:
        One row per comparison and level: E and its 95% interval.
    """
    grid = config["grid"]
    chance = config["chance"]
    out = []
    for question, ref, treat in analyse.pairs(config, rows):
        r = analyse.curve(rows, ref, grid)
        t = analyse.curve(rows, treat, grid)
        top = min(
            exchange_rate.isotonic(r.test.mean(0))[-1],
            exchange_rate.isotonic(t.test.mean(0))[-1],
        )
        low = max(
            exchange_rate.isotonic(r.test.mean(0))[0],
            exchange_rate.isotonic(t.test.mean(0))[0],
            chance + 0.05 * (top - chance),
        )
        if top <= low:
            continue
        for target in np.linspace(low, top, levels):
            est = exchange_rate.exchange_rate(
                r.sizes, r.test.mean(0), t.sizes, t.test.mean(0), target
            )
            band = exchange_rate.bootstrap_interval(
                r.sizes, r.test, t.sizes, t.test, target, draws=draws
            )
            out.append(
                {
                    "question": question,
                    "reference": "+".join(ref),
                    "treatment": "+".join(treat),
                    "target": float(target),
                    "E": est.value if est.kind == "exact" else np.nan,
                    "low": band.low,
                    "high": band.high,
                }
            )
    return pd.DataFrame(out)


def error_ratios(
    config: dict[str, Any], rows: pd.DataFrame, draws: int = 2000
) -> pd.DataFrame:
    """Computes error-rate ratios at each grid size for every comparison.

    Args:
        config: Parsed config.
        rows: Final runs.
        draws: Seed-bootstrap draws for the interval.

    Returns:
        One row per comparison and size: the reference's error rate over
        the treatment's, from seed means, with a 95% interval.
    """
    grid = config["grid"]
    rng = np.random.default_rng(0)
    out = []
    for question, ref, treat in analyse.pairs(config, rows):
        r = analyse.curve(rows, ref, grid)
        t = analyse.curve(rows, treat, grid)
        ref_err, treat_err = 1.0 - r.test, 1.0 - t.test
        boot = np.empty((draws, len(grid)))
        for d in range(draws):
            a = ref_err[rng.integers(0, len(ref_err), len(ref_err))].mean(0)
            b = treat_err[rng.integers(0, len(treat_err), len(treat_err))].mean(0)
            boot[d] = a / b
        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
        point = ref_err.mean(0) / treat_err.mean(0)
        for i, size in enumerate(grid):
            out.append(
                {
                    "question": question,
                    "reference": "+".join(ref),
                    "treatment": "+".join(treat),
                    "size": size,
                    "n_labelled": float(r.sizes[i]),
                    "ref_error": float(ref_err.mean(0)[i]),
                    "treat_error": float(treat_err.mean(0)[i]),
                    "error_ratio": float(point[i]),
                    "low": float(low[i]),
                    "high": float(high[i]),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    """Writes the rate-vs-target and error-ratio tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = config_lib.load_config(parser.parse_args().config)
    rows = analyse.load_finals(config)
    results = Path(config["results_dir"])
    curves = rate_curves(config, rows)
    curves.to_csv(results / f"rate_vs_target_{config['dataset']}.csv", index=False)
    errors = error_ratios(config, rows)
    errors.to_csv(results / f"error_ratios_{config['dataset']}.csv", index=False)
    flat = curves.groupby("question").E.agg(["min", "max"])
    flat["max_over_min"] = flat["max"] / flat["min"]
    print(flat.round(2).to_string())


if __name__ == "__main__":
    main()
