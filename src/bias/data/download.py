"""Download every dataset used by the study into one data root."""

import argparse
import urllib.request
from pathlib import Path

import torchaudio
import torchvision
import truststore

# ehu.eus serves an incomplete chain; the OS trust store resolves it.
truststore.inject_into_ssl()

HSI_URLS = {
    "Indian_pines_corrected.mat": "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
    "Indian_pines_gt.mat": "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    "PaviaU.mat": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
    "PaviaU_gt.mat": "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
}


def download_hsi(root: Path) -> None:
    """Fetch the Indian Pines and Pavia University .mat files."""
    hsi_dir = root / "hsi"
    hsi_dir.mkdir(parents=True, exist_ok=True)
    for name, url in HSI_URLS.items():
        target = hsi_dir / name
        if not target.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--only", choices=["cifar", "audio", "hsi"], default=None)
    args = parser.parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "hsi"):
        download_hsi(root)
    if args.only in (None, "cifar"):
        torchvision.datasets.CIFAR100(str(root), train=True, download=True)
        torchvision.datasets.CIFAR100(str(root), train=False, download=True)
    if args.only in (None, "audio"):
        torchaudio.datasets.SPEECHCOMMANDS(str(root), download=True)


if __name__ == "__main__":
    main()
