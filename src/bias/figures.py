"""Figures from the results tables.

Usage:
    python -m bias.figures configs/images.yaml
"""

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402

from bias import analyse, scramble, variants  # noqa: E402
from bias import config as config_lib  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Series take a colour and a marker by position, in this order.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKERS = "os^Dv"
MODELS = {"cnn": "CNN", "vit": "ViT", "mlp": "MLP"}
TREATMENTS = {
    "none": "No augmentation",
    "aug": "Augmentation",
    "lejepa": "LeJEPA + aug",
    "imagenet": "ImageNet-21k start + aug",
    "imagenet1k": "ImageNet-1k start + aug",
}
SCRAMBLES = (
    "real",
    *(f"jigsaw{b}" for b in (16, 8, 4, 2)),
    "patches",
    *(f"local{b}" for b in (2, 4, 8, 16)),
    "pixels",
)
# The ViTs of bias.inside that are drawn as series.
VITS = ("random", "supervised_1%", "supervised_10%", "supervised_100%", "lejepa")
PERCENT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def _axes(
    panels: int, cols: int, width: float, height: float, share: bool = False
) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    """Makes a grid of styled panels.

    Args:
        panels: Number of panels.
        cols: Panels per row.
        width: Width of one panel, in inches.
        height: Height of one panel, in inches.
        share: Whether the panels share both axes.

    Returns:
        The figure and its panels, row by row.
    """
    rows = math.ceil(panels / cols)
    fig, grid = plt.subplots(
        rows,
        cols,
        figsize=(width * cols, height * rows),
        squeeze=False,
        sharex=share,
        sharey=share,
    )
    for ax in grid.flat:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.title.set(color=INK, fontsize=9)
        ax.xaxis.label.set(color=MUTED, fontsize=9)
        ax.yaxis.label.set(color=MUTED, fontsize=9)
    for ax in grid.flat[panels:]:
        ax.set_visible(False)
    return fig, list(grid.flat[:panels])


def _series(
    ax: matplotlib.axes.Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    slot: int,
    label: str,
    ls: str = "-",
) -> None:
    """Draws one series in its slot's colour and marker.

    Args:
        ax: The panel.
        x: Horizontal values.
        y: Vertical values.
        slot: Position of the series in PALETTE and MARKERS.
        label: Legend entry.
        ls: Line style; "" draws the markers alone.
    """
    ax.plot(
        x, y, color=PALETTE[slot], marker=MARKERS[slot], lw=2, ms=5, ls=ls, label=label
    )


def _save(fig: matplotlib.figure.Figure, config: dict[str, Any], name: str) -> Path:
    """Adds a legend for the labelled series and writes the figure.

    Args:
        fig: The figure.
        config: Parsed config, with figures_dir.
        name: File stem; the dataset name is appended.

    Returns:
        The PNG path.
    """
    entries: dict[str, Any] = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    fig.tight_layout()
    if entries:
        fig.legend(
            entries.values(),
            entries.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(entries),
            frameon=False,
            fontsize=9,
        )
    out = Path(config["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{config['dataset']}.png"
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def _table(config: dict[str, Any], name: str) -> Path:
    return Path(config["results_dir"]) / name.format(config["dataset"])


def curves(config: dict[str, Any], rows: pd.DataFrame) -> Path:
    """Draws test accuracy against labelled examples, one panel per model.

    Args:
        config: Parsed config.
        rows: Final runs.

    Returns:
        The PNG path.
    """
    setups = set(zip(rows.model, rows.treatment, strict=True))
    models = [m for m in MODELS if m in set(rows.model)]
    fig, axes = _axes(len(models), len(models), 4.2, 3.6, share=True)
    for ax, model in zip(axes, models, strict=True):
        for slot, (treatment, label) in enumerate(TREATMENTS.items()):
            if (model, treatment) not in setups:
                continue
            c = analyse.curve(rows, (model, treatment), config["grid"])
            ax.fill_between(
                c.sizes, c.test.min(0), c.test.max(0), color=PALETTE[slot], alpha=0.15
            )
            _series(ax, c.sizes, c.test.mean(0), slot, label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(c.sizes, [f"{n:,.0f}" for n in c.sizes], minor=False)
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(PERCENT)
        ax.set(title=MODELS[model], xlabel="Labelled examples")
    axes[0].set_ylabel("Test accuracy")
    return _save(fig, config, "curves")


def rates(config: dict[str, Any]) -> Path:
    """Draws the exchange rate against the target score, per comparison.

    Args:
        config: Parsed config.

    Returns:
        The PNG path.
    """
    table = pd.read_csv(_table(config, "rate_vs_target_{}.csv"))
    questions = list(dict.fromkeys(table.question))
    fig, axes = _axes(len(questions), 4, 3.4, 2.6)
    for ax, question in zip(axes, questions, strict=True):
        c = table[table.question == question]
        # Levels some bootstrap draws never cross have no finite band.
        band = c[np.isfinite(c.high) & (c.low > 0)]
        ax.fill_between(band.target, band.low, band.high, color=PALETTE[0], alpha=0.18)
        ax.plot(c.target, c.E, color=PALETTE[0], lw=2)
        ax.axhline(1.0, color=MUTED, lw=1, zorder=0)
        ax.set_yscale("log", base=2)
        ax.xaxis.set_major_formatter(PERCENT)
        ax.set_title(question, loc="left")
    fig.supxlabel("Target test accuracy", color=MUTED, fontsize=9)
    fig.supylabel("Exchange rate E", color=MUTED, fontsize=9)
    return _save(fig, config, "rate_vs_target")


def _dots(table: pd.DataFrame, xlabel: str) -> matplotlib.figure.Figure:
    """Draws test accuracy with one line per row and one marker per column.

    Args:
        table: Scores, indexed by row label, one column per series.
        xlabel: Label of the accuracy axis.

    Returns:
        The figure.
    """
    fig, (ax,) = _axes(1, 1, 6.4, 0.3 * len(table) + 0.9)
    ys = np.arange(len(table))[::-1]
    for slot, column in enumerate(table.columns):
        _series(ax, table[column], ys, slot, str(column), ls="")
    ax.set_yticks(ys, list(table.index))
    ax.xaxis.set_major_formatter(PERCENT)
    ax.set_xlabel(xlabel)
    return fig


def scrambles(config: dict[str, Any], rows: pd.DataFrame) -> Path:
    """Draws each model's accuracy on every scramble it was trained on.

    Args:
        config: Parsed config.
        rows: Final runs, for the accuracy on real images.

    Returns:
        The PNG path.
    """
    runs = pd.read_csv(_table(config, "{}_scramble.csv"))
    runs["scramble"] = runs.stage.str.removeprefix("scramble_")
    real = rows[(rows.treatment == "aug") & (rows["size"] == scramble.SIZE)]
    both = pd.concat([real.assign(scramble="real"), runs])
    table = both.pivot_table(index="scramble", columns="model", values="test")
    table = table.loc[list(SCRAMBLES), list(MODELS)].rename(columns=MODELS)
    fig = _dots(table, f"Test accuracy, {scramble.SIZE} of the labels")
    return _save(fig, config, "scrambles")


def vit_variants(config: dict[str, Any], rows: pd.DataFrame) -> Path:
    """Draws the accuracy of each ViT variant, the plain ViT and the CNN.

    Args:
        config: Parsed config.
        rows: Final runs, for the plain ViT and the CNN.

    Returns:
        The PNG path.
    """
    runs = pd.read_csv(_table(config, "{}_variants.csv"))
    both = pd.concat([rows[rows.treatment == "aug"], runs])
    table = both.pivot_table(index="model", columns="size", values="test")
    models = ["vit", *(f"vit_{v}" for v in variants.RUNS), "cnn"]
    sizes = [size for size in config["grid"] if size in set(runs["size"])]
    fig = _dots(table.loc[models, sizes], "Test accuracy, by share of the labels")
    return _save(fig, config, "variants")


def attention(config: dict[str, Any]) -> Path:
    """Draws how far each ViT's layers look.

    Args:
        config: Parsed config.

    Returns:
        The PNG path.
    """
    table = pd.read_csv(_table(config, "inside_attention_{}.csv"))
    fig, (ax,) = _axes(1, 1, 6.4, 3.6)
    for slot, name in enumerate(VITS):
        mine = table[table.model == name]
        _series(ax, mine.layer, mine.attention_distance, slot, name)
    ax.set(xlabel="Layer", ylabel="Mean attention distance (patches)")
    return _save(fig, config, "attention")


def attention_offsets(config: dict[str, Any]) -> Path:
    """Draws where each ViT's layers look, relative to the patch looking.

    Args:
        config: Parsed config.

    Returns:
        The PNG path.
    """
    profiles = np.load(_table(config, "inside_profiles_{}.npz"))
    layers = len(profiles[VITS[0]])
    fig, axes = _axes(len(VITS) * layers, layers, 1.1, 1.1)
    for i, ax in enumerate(axes):
        name, layer = VITS[i // layers], i % layers
        ax.imshow(profiles[name][layer], cmap="Oranges", vmin=0)
        ax.set(xticks=[], yticks=[])
        if i < layers:
            ax.set_title(f"Layer {layer + 1}")
        if layer == 0:
            ax.set_ylabel(name, fontsize=7)
    return _save(fig, config, "attention_offsets")


def positions(config: dict[str, Any]) -> Path:
    """Draws each ViT's position embeddings in their two main directions.

    Patches that are neighbours in the image are joined by a line.

    Args:
        config: Parsed config.

    Returns:
        The PNG path.
    """
    vectors = np.load(_table(config, "inside_embeddings_{}.npz"))
    fig, axes = _axes(len(vectors.files), 4, 2.4, 2.4)
    for ax, name in zip(axes, vectors.files, strict=True):
        side = math.isqrt(len(vectors[name]))
        u, scale, _ = np.linalg.svd(vectors[name] - vectors[name].mean(0))
        x, y = (u[:, :2] * scale[:2]).reshape(side, side, 2).transpose(2, 0, 1)
        ax.plot(x, y, color=MUTED, lw=0.8)
        ax.plot(x.T, y.T, color=MUTED, lw=0.8)
        ax.scatter(x, y, c=np.indices(x.shape)[0], cmap="Oranges", vmin=-3, s=14)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    return _save(fig, config, "positions")


def main() -> None:
    """Writes every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = config_lib.load_config(parser.parse_args().config)
    rows = analyse.load_finals(config)
    for path in (
        curves(config, rows),
        rates(config),
        scrambles(config, rows),
        vit_variants(config, rows),
        attention(config),
        attention_offsets(config),
        positions(config),
    ):
        print(path)


if __name__ == "__main__":
    main()
