"""Decodes the ImageNet-32 parquet files into one uint8 .npz cache.

The parquet files come from the Hugging Face dataset
benjamin-paine/imagenet-1k-32x32 (ImageNet-1k downsampled to 32x32). Its
"test" split has no labels, so only "train" and "validation" are kept.

Usage (pyarrow is needed only here, not by the project):
    uv run --with pyarrow python scripts/imagenet32_to_npz.py data/imagenet32
"""

import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def _decode(blobs: list[bytes]) -> np.ndarray:
    return np.stack(
        [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in blobs]
    )


def _split(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for path in sorted(root.glob(f"{name}-*.parquet")):
        table = pq.read_table(path)
        blobs = [row["bytes"] for row in table.column("image").to_pylist()]
        chunks = [blobs[i : i + 20000] for i in range(0, len(blobs), 20000)]
        with ProcessPoolExecutor() as pool:
            images.extend(pool.map(_decode, chunks))
        labels.append(np.asarray(table.column("label").to_pylist(), dtype=np.int64))
    return np.concatenate(images), np.concatenate(labels)


def main() -> None:
    """Writes <root>/imagenet32.npz with train and validation arrays."""
    root = Path(sys.argv[1])
    train_x, train_y = _split(root, "train")
    val_x, val_y = _split(root, "validation")
    assert train_x.shape[1:] == (32, 32, 3) and train_y.max() == 999
    np.savez(
        root / "imagenet32.npz",
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
    )
    print(train_x.shape, val_x.shape)


if __name__ == "__main__":
    main()
