"""Measures where ViTs look and what their position embeddings hold.

Trains ViTs on patch-scrambled CIFAR-100 (see bias.scramble) and saves
them, then measures those, a random-weights ViT and the LeJEPA-pretrained
ViT with its mid-training snapshots:

- Position embeddings, put back in the true layout, and the mean cosine
  similarity between those of adjacent patches.
- Attention distance: how far, in patches of the true layout, each block
  looks on average.
- Attention by offset: where each block looks relative to the patch that
  is looking.

Usage:
    python -m bias.inside configs/images.yaml train
    python -m bias.inside configs/images.yaml measure
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from bias import config as config_lib
from bias import grid, runlog, scramble, train
from bias.data import tensors
from bias.models import registry

SIZES = ("1%", "10%", "100%")
PER_SIDE = scramble.SIDE // scramble.PATCH
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


def checkpoint(config: dict[str, Any], size: str) -> Path:
    """Returns where the scrambled-image ViT for a data size is saved.

    Args:
        config: Parsed config.
        size: Grid entry, such as "10%".

    Returns:
        The checkpoint path.
    """
    name = f"{config['dataset']}_vit_{size.rstrip('%')}.pt"
    return Path("checkpoints") / "inside" / name


def true_position() -> IntArray:
    """Returns the true grid position of each token the scrambled ViT sees.

    Returns:
        For token i of the patch-scrambled image, the row-major index of
        the patch it shows in the real image.
    """
    source = scramble.permutation("patches").reshape(scramble.SIDE, scramble.SIDE)
    corner = source[:: scramble.PATCH, :: scramble.PATCH]
    row = corner // scramble.SIDE // scramble.PATCH
    col = (corner % scramble.SIDE) // scramble.PATCH
    return np.asarray((row * PER_SIDE + col).reshape(-1), dtype=np.int64)


def offsets(where: IntArray) -> tuple[IntArray, IntArray]:
    """Returns the rows and columns between every pair of tokens.

    Args:
        where: Grid position of each token, row-major.

    Returns:
        Two (tokens, tokens) arrays: entry [i, j] is how many rows, and
        how many columns, token j sits from token i.
    """
    rows, cols = np.divmod(where, PER_SIDE)
    return rows[None, :] - rows[:, None], cols[None, :] - cols[:, None]


def neighbour_similarity(pos: torch.Tensor) -> float:
    """Returns how alike the position embeddings of adjacent patches are.

    Args:
        pos: Position embeddings in the true layout, shape (patches, width).

    Returns:
        The mean cosine similarity over pairs of patches that touch,
        diagonals included, after the mean embedding is taken off.
    """
    unit = torch.nn.functional.normalize(pos - pos.mean(0), dim=1)
    dy, dx = offsets(np.arange(len(pos)))
    touching = np.maximum(abs(dy), abs(dx)) == 1
    return float((unit @ unit.T)[torch.from_numpy(touching)].mean())


@torch.no_grad()
def _attention(model: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Returns each block's patch-to-patch attention, averaged over heads.

    Args:
        model: A timm ViT.
        x: Images to run through it.

    Returns:
        One (batch, patches, patches) tensor per block. The class token is
        left out and each row renormalised to sum to one.
    """
    model.eval()
    tokens = model.patch_embed(x)  # type: ignore[operator]
    tokens = model.norm_pre(model.patch_drop(model._pos_embed(tokens)))  # type: ignore[operator]
    out = []
    for block in model.blocks:  # type: ignore[union-attr]
        attn = block.attn
        y = block.norm1(tokens)
        batch, n, width = y.shape
        qkv = attn.qkv(y).reshape(batch, n, 3, attn.num_heads, width // attn.num_heads)
        q, k = qkv.permute(2, 0, 3, 1, 4)[:2]
        weights = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)[:, :, 1:, 1:]
        weights = weights / weights.sum(-1, keepdim=True)
        out.append(weights.mean(1).float())
        tokens = block(tokens)
    return out


def attention_stats(
    model: torch.nn.Module, x: torch.Tensor, where: IntArray
) -> tuple[FloatArray, FloatArray]:
    """Returns how far each block looks, and where.

    Args:
        model: A timm ViT.
        x: Images to run through it.
        where: True grid position of each token the model sees.

    Returns:
        The attention-weighted distance in patches from each patch to the
        ones it attends to, one value per block, averaged over patches and
        images. And a (blocks, 15, 15) array whose entry [l, dy + 7, dx + 7]
        is the mean attention block l pays to the patch dy rows and dx
        columns away, over the patch pairs that far apart.
    """
    dy, dx = offsets(where)
    distance = torch.from_numpy(np.hypot(dy, dx)).to(x)
    span = 2 * PER_SIDE - 1
    cell = torch.from_numpy((dy + PER_SIDE - 1) * span + dx + PER_SIDE - 1).reshape(-1)
    count = torch.bincount(cell, minlength=span * span).clamp(min=1)
    distances, profiles = [], []
    for weights in _attention(model, x):
        distances.append(float((weights * distance).sum(-1).mean()))
        paid = weights.mean(0).cpu().reshape(-1)
        total = torch.bincount(cell, paid, minlength=span * span)
        profiles.append((total / count).reshape(span, span).numpy())
    return np.asarray(distances, np.float64), np.asarray(profiles, np.float64)


def train_all(config: dict[str, Any]) -> None:
    """Trains one ViT per size on patch-scrambled images and saves them.

    Args:
        config: Parsed config.
    """
    settings = grid.best_settings(grid.read_runs(config, "tune"))
    device = train.pick_device()
    data = scramble.scrambled(tensors.load_tensors(config, device), "patches")
    log = Path(config["results_dir"]) / f"{config['dataset']}_inside.csv"
    for size in SIZES:
        path = checkpoint(config, size)
        if path.exists():
            continue
        tuned = grid.tuned(config, settings, "vit", "aug", size)
        job = train.Job("vit", "aug", size, 0, *tuned)
        record = train.train_run(config, data, job, "inside", device, save_to=path)
        print(record, flush=True)
        runlog.append_run(log, record)


def _load(path: Path | None) -> torch.nn.Module:
    torch.manual_seed(0)
    model = registry.build("vit")
    if path is not None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    return model


def measure(config: dict[str, Any]) -> None:
    """Measures every saved ViT and writes the results.

    Writes inside_attention_<dataset>.csv (attention distance per model
    and layer, with the model's neighbour similarity), and one array per
    model in inside_embeddings_<dataset>.npz (position embeddings in the
    true layout) and inside_profiles_<dataset>.npz (attention by offset).

    Args:
        config: Parsed config.
    """
    device = train.pick_device()
    real = tensors.load_tensors(config, device)
    images = {
        True: scramble.scrambled(real, "patches").val_x[:1000],
        False: real.val_x[:1000],
    }
    # name -> (checkpoint or None for random weights, trained on scrambled images)
    entries: dict[str, tuple[Path | None, bool]] = {"random": (None, False)}
    for size in SIZES:
        entries[f"supervised_{size}"] = (checkpoint(config, size), True)
    final = config_lib.lejepa_checkpoint(config, "vit", 0)
    snapshots = final.parent.glob(f"{final.stem}_step*.pt")
    for snapshot in sorted(snapshots, key=lambda p: int(p.stem.rsplit("step", 1)[1])):
        entries[f"lejepa_{snapshot.stem.rsplit('_', 1)[1]}"] = (snapshot, False)
    entries["lejepa"] = (final, False)
    rows, embeddings, profiles = [], {}, {}
    for name, (path, was_scrambled) in entries.items():
        if path is not None and not path.exists():
            continue
        model = _load(path).to(device)
        where = true_position() if was_scrambled else np.arange(PER_SIDE**2)
        pos = model.get_parameter("pos_embed")[0, 1:].detach().cpu()
        pos = pos[torch.from_numpy(np.argsort(where))]
        embeddings[name] = pos.numpy()
        distances, profiles[name] = attention_stats(model, images[was_scrambled], where)
        rows += [
            {
                "model": name,
                "layer": layer + 1,
                "attention_distance": distance,
                "neighbour_similarity": neighbour_similarity(pos),
            }
            for layer, distance in enumerate(distances)
        ]
    results = Path(config["results_dir"])
    table = pd.DataFrame(rows)
    table.to_csv(results / f"inside_attention_{config['dataset']}.csv", index=False)
    np.savez(results / f"inside_embeddings_{config['dataset']}.npz", **embeddings)  # type: ignore[arg-type]
    np.savez(results / f"inside_profiles_{config['dataset']}.npz", **profiles)  # type: ignore[arg-type]
    print(table.pivot(index="layer", columns="model", values="attention_distance"))


def main() -> None:
    """Trains the scrambled-image ViTs, or measures the saved ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("step", choices=("train", "measure"))
    args = parser.parse_args()
    config = config_lib.load_config(args.config)
    {"train": train_all, "measure": measure}[args.step](config)


if __name__ == "__main__":
    main()
