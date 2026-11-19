"""Download CIFAR-100 into the data root."""

import argparse
from pathlib import Path

import torchvision


def main() -> None:
    """Fetches the CIFAR-100 train and test sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.CIFAR100(str(root), train=True, download=True)
    torchvision.datasets.CIFAR100(str(root), train=False, download=True)


if __name__ == "__main__":
    main()
