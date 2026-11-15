"""Speech Commands v2: official splits, cached waveforms, log-mel features.

Waveforms are padded or cut to one second and cached as int16 arrays, so
time-shift and noise augmentation can run on the waveform before the
log-mel transform.
"""

import dataclasses
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile
import torch
import torchaudio

SAMPLE_RATE = 16_000
SPLIT_SUBSETS = {"pool": "training", "val": "validation", "test": "testing"}


@dataclasses.dataclass(frozen=True)
class AudioSplit:
    """One split of Speech Commands.

    Attributes:
        waves: One-second clips, shape (n, 16000), int16.
        labels: Class index per clip.
    """

    waves: npt.NDArray[np.int16]
    labels: npt.NDArray[np.int64]


def _dataset_dir(root: Path) -> Path:
    return root / "SpeechCommands" / "speech_commands_v0.02"


def class_names(root: Path) -> list[str]:
    """Returns the 35 word classes in sorted order.

    Args:
        root: Data root that torchaudio downloaded into.

    Returns:
        Sorted class names.
    """
    base = _dataset_dir(root)
    return sorted(
        d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")
    )


def split_files(root: Path, split: str) -> list[Path]:
    """Lists the audio files of one official split.

    Args:
        root: Data root that torchaudio downloaded into.
        split: "pool", "val" or "test".

    Returns:
        Absolute paths of the split's clips.
    """
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        str(root), subset=SPLIT_SUBSETS[split], download=False
    )
    return [Path(p) for p in dataset._walker]


def load_split(root: Path, split: str) -> AudioSplit:
    """Loads one split, building the int16 cache on first use.

    Args:
        root: Data root that torchaudio downloaded into.
        split: "pool", "val" or "test".

    Returns:
        The split's waveforms and labels.
    """
    cache = root / "speech_commands_cache" / f"{split}.npz"
    if cache.exists():
        with np.load(cache) as stored:
            return AudioSplit(waves=stored["waves"], labels=stored["labels"])
    names = {name: i for i, name in enumerate(class_names(root))}
    files = split_files(root, split)
    waves = np.zeros((len(files), SAMPLE_RATE), dtype=np.int16)
    labels = np.empty(len(files), dtype=np.int64)
    for i, path in enumerate(files):
        clip, rate = soundfile.read(path, dtype="int16")
        if rate != SAMPLE_RATE:
            raise ValueError(f"unexpected sample rate {rate} in {path}")
        waves[i, : min(len(clip), SAMPLE_RATE)] = clip[:SAMPLE_RATE]
        labels[i] = names[path.parent.name]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, waves=waves, labels=labels)
    return AudioSplit(waves=waves, labels=labels)


def background_noise(root: Path) -> list[npt.NDArray[np.float32]]:
    """Loads the dataset's background-noise recordings for augmentation.

    Args:
        root: Data root that torchaudio downloaded into.

    Returns:
        One float waveform in [-1, 1] per noise file.
    """
    noise_dir = _dataset_dir(root) / "_background_noise_"
    return [
        soundfile.read(path, dtype="float32")[0]
        for path in sorted(noise_dir.glob("*.wav"))
    ]


class LogMel(torch.nn.Module):
    """Turns int16-scaled waveforms into log-mel spectrograms.

    Output shape is (batch, 1, n_mels, frames).
    """

    def __init__(self, n_mels: int = 40, win_ms: int = 30, hop_ms: int = 10) -> None:
        """Builds the transform.

        Args:
            n_mels: Mel bands.
            win_ms: Window length in milliseconds.
            hop_ms: Hop length in milliseconds.
        """
        super().__init__()
        win = SAMPLE_RATE * win_ms // 1000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=win,
            hop_length=SAMPLE_RATE * hop_ms // 1000,
            n_mels=n_mels,
        )

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        """Computes log-mel features.

        Args:
            waves: Float waveforms in [-1, 1], shape (batch, samples).

        Returns:
            Log-mel features, shape (batch, 1, n_mels, frames).
        """
        mel: torch.Tensor = self.mel(waves)
        return torch.log(mel + 1e-6).unsqueeze(1)
