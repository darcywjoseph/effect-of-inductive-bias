import numpy as np

from bias.data import hsi


def toy_labels(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.full((80, 80), -1, dtype=np.int64)
    labels[5:75, 5:75] = rng.integers(0, 3, (70, 70))
    return labels


def test_patches_never_cross_splits() -> None:
    labels = toy_labels()
    radius = 2
    result = hsi.block_split(labels, block=10, radius=radius, tries=50)
    rows, cols = np.nonzero(result.split >= 0)
    for r, c in zip(rows, cols, strict=True):
        own = result.split[r, c]
        window = result.block_split[
            max(r - radius, 0) : r + radius + 1, max(c - radius, 0) : c + radius + 1
        ]
        assert np.all(window == own)


def test_every_split_gets_every_class() -> None:
    result = hsi.block_split(toy_labels(), block=10, radius=1, tries=200)
    for k in (hsi.TRAIN, hsi.VAL, hsi.TEST):
        assert set(np.unique(toy_labels()[result.split == k])) == {0, 1, 2}
    assert result.score > 0


def test_split_is_deterministic() -> None:
    a = hsi.block_split(toy_labels(), block=10, radius=1, seed=3, tries=20)
    b = hsi.block_split(toy_labels(), block=10, radius=1, seed=3, tries=20)
    np.testing.assert_array_equal(a.split, b.split)


def test_random_split_covers_all_labelled_pixels() -> None:
    labels = toy_labels()
    split = hsi.random_split(labels)
    assert np.all((split >= 0) == (labels >= 0))


def test_extract_patches_centre_matches_pixel() -> None:
    cube = np.random.default_rng(0).normal(size=(20, 30, 4)).astype(np.float32)
    rows, cols = np.array([0, 10, 19]), np.array([0, 15, 29])
    patches = hsi.extract_patches(cube, rows, cols, radius=3)
    assert patches.shape == (3, 4, 7, 7)
    np.testing.assert_array_equal(patches[:, :, 3, 3], cube[rows, cols])
