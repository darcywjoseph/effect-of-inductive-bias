"""Learning-curve figures from the final runs.

One panel per model, one line per treatment, so a treatment keeps its
colour and marker in every panel. Lines are seed means, the band spans
the seeds, and the two fixed targets are drawn as grey reference lines.

Usage:
    python -m bias.plots configs/images.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from bias import analyse  # noqa: E402
from bias import config as config_lib  # noqa: E402

# Categorical slots 1-5 of the reference palette, in fixed order, each with
# its own marker so identity never rests on colour alone.
TREATMENT_STYLE = {
    "none": ("#2a78d6", "o", "No augmentation"),
    "aug": ("#eb6834", "s", "Augmentation"),
    "lejepa": ("#1baf7a", "^", "LeJEPA + aug"),
    "imagenet": ("#eda100", "D", "ImageNet-21k start + aug"),
    "imagenet1k": ("#e87ba4", "v", "ImageNet-1k start + aug"),
}
MODEL_TITLES = {"cnn": "CNN", "vit": "ViT-Tiny", "mlp": "MLP"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRIC_LABELS = {
    "top1": "Test accuracy (top-1)",
    "balanced_acc": "Test balanced accuracy",
}


def learning_curves(config: dict[str, Any], out: Path) -> Path:
    """Draws the learning curves of one dataset.

    Args:
        config: Parsed config.
        out: Directory for the figure.

    Returns:
        The PNG path.
    """
    rows = analyse.load_finals(config)
    grid = config["grid"]
    targets = analyse.targets(config, rows)
    models = [m for m in MODEL_TITLES if m in set(rows.model)]
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True, sharex=True
    )
    for ax, model in zip(axes, models, strict=True):
        treatments = [
            t
            for t in TREATMENT_STYLE
            if ((rows.model == model) & (rows.treatment == t)).any()
        ]
        for treatment in treatments:
            colour, marker, label = TREATMENT_STYLE[treatment]
            c = analyse.curve(rows, (model, treatment), grid)
            mean = c.test.mean(0)
            ax.fill_between(
                c.sizes, c.test.min(0), c.test.max(0), color=colour, alpha=0.15, lw=0
            )
            ax.plot(c.sizes, mean, color=colour, lw=2, marker=marker, ms=5, label=label)
        for size, score in targets.items():
            ax.axhline(score, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
            ax.annotate(
                f"target ({size})",
                (0.0, score),
                xycoords=("axes fraction", "data"),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=7,
                color=MUTED,
            )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )
        ax.set_title(MODEL_TITLES[model], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Labelled examples", color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel(METRIC_LABELS[config["metric"]], color=MUTED, fontsize=9)
    handles = [
        matplotlib.lines.Line2D([], [], color=c, marker=m, lw=2, ms=5, label=label)
        for t, (c, m, label) in TREATMENT_STYLE.items()
        if t in set(rows.treatment)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"curves_{config['dataset']}.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    """Writes figures/curves_<dataset>.png."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    print(learning_curves(config_lib.load_config(args.config), args.out))


if __name__ == "__main__":
    main()
