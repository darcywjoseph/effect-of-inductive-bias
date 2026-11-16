"""Download every dataset used by the study into one data root."""

import argparse
import urllib.request
from pathlib import Path

import torchaudio
import torchvision

# Copies of the UPV/EHU "Hyperspectral Remote Sensing Scenes" files (sizes,
# shapes and class counts match). The original host refuses scripts.
_MIRROR = "https://huggingface.co/datasets/danaroth"
HSI_URLS = {
    "Salinas_corrected.mat": f"{_MIRROR}/salinas/resolve/main/Salinas_corrected.mat",
    "Salinas_gt.mat": f"{_MIRROR}/salinas/resolve/main/Salinas_gt.mat",
    "PaviaU.mat": f"{_MIRROR}/pavia/resolve/main/PaviaU.mat",
    "PaviaU_gt.mat": f"{_MIRROR}/pavia/resolve/main/PaviaU_gt.mat",
}


def download_hsi(root: Path) -> None:
    """Fetches the Salinas and Pavia University .mat files.

    Args:
        root: Data root; files land in root/hsi.
    """
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
