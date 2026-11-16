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


class WaveAugment:
    """Time shift and background-noise mixing for one waveform.

    This is the dataset's own recipe (Warden, 2018, and the TensorFlow
    speech-commands example): shift by up to 100 ms with zero fill, then,
    with some probability, add a random stretch of a background-noise
    recording at a random volume. It is a class so DataLoader workers can
    pickle it.
    """

    def __init__(
        self,
        noise: list[npt.NDArray[np.float32]],
        shift_ms: int = 100,
        noise_prob: float = 0.8,
        noise_max: float = 0.1,
    ) -> None:
        """Stores the noise recordings and settings.

        Args:
            noise: Background-noise waveforms in [-1, 1].
            shift_ms: Largest shift either way, in milliseconds.
            noise_prob: Chance of mixing in noise.
            noise_max: Largest noise volume; the volume is uniform below it.
        """
        self.noise = [torch.from_numpy(n) for n in noise if len(n) >= SAMPLE_RATE]
        self.shift = SAMPLE_RATE * shift_ms // 1000
        self.noise_prob = noise_prob
        self.noise_max = noise_max

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """Augments one int16 clip.

        Args:
            wave: Shape (16000,), int16.

        Returns:
            Float waveform in [-1, 1], shape (16000,).
        """
        x = wave.float() / 32768.0
        shift = int(torch.randint(-self.shift, self.shift + 1, ()))
        x = torch.roll(x, shift)
        if shift > 0:
            x[:shift] = 0
        elif shift < 0:
            x[shift:] = 0
        if self.noise and float(torch.rand(())) < self.noise_prob:
            clip = self.noise[int(torch.randint(len(self.noise), ()))]
            start = int(torch.randint(len(clip) - SAMPLE_RATE + 1, ()))
            volume = float(torch.rand(())) * self.noise_max
            x = x + volume * clip[start : start + SAMPLE_RATE]
        return x.clamp(-1.0, 1.0)


class SpecAugment(torch.nn.Module):
    """Frequency and time masking of a batch of log-mels (Park et al., 2019).

    Every example gets its own masks. Masked cells are set to 0, which is
    the mean after normalisation.
    """

    def __init__(
        self, freq_masks: int, freq_width: int, time_masks: int, time_width: int
    ) -> None:
        """Stores the mask counts and largest widths.

        Args:
            freq_masks: Frequency masks per example.
            freq_width: Largest frequency mask, in mel bins.
            time_masks: Time masks per example.
            time_width: Largest time mask, in frames.
        """
        super().__init__()
        self.freq = (freq_masks, freq_width)
        self.time = (time_masks, time_width)

    def _mask(self, n: int, size: int, count: int, width: int, device: torch.device) -> torch.Tensor:
        widths = torch.randint(0, width + 1, (n, count, 1), device=device)
        starts = (torch.rand(n, count, 1, device=device) * (size - widths + 1)).long()
        cells = torch.arange(size, device=device).view(1, 1, size)
        return ((cells >= starts) & (cells < starts + widths)).any(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Masks a batch.

        Args:
            x: Log-mels of shape (batch, 1, mels, frames).

        Returns:
            The masked batch, same shape.
        """
        n, _, mels, frames = x.shape
        freq = self._mask(n, mels, *self.freq, x.device).view(n, 1, mels, 1)
        time = self._mask(n, frames, *self.time, x.device).view(n, 1, 1, frames)
        return x.masked_fill(freq | time, 0.0)
