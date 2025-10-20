"""Tests for data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from src.data import (
    CLASS_NAMES,
    AugmentationConfig,
    DataModuleConfig,
    EmotionDataModule,
    apply_augmentations,
    compute_class_distribution,
    compute_class_weights,
    compute_dataset_statistics,
    normalize_image,
    stratified_split,
    _scan_split,
)


def _write_image(path: Path, value: int) -> None:
    """Write a synthetic grayscale image with a constant pixel value."""
    arr = np.full((48, 48), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset hierarchy for use across data tests."""
    dataset_root = tmp_path / "dataset"
    for split in ("train", "test"):
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = dataset_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            _write_image(class_dir / "img0.png", idx * 10 + 5)
            if split == "train":
                _write_image(class_dir / "img1.png", idx * 10 + 15)
    return dataset_root


def test_emotion_data_module_end_to_end(tiny_dataset: Path) -> None:
    """Test that the data module sets up splits, statistics, and batches."""
    config = DataModuleConfig(
        data_dir=tiny_dataset,
        batch_size=4,
        val_ratio=0.25,
        seed=123,
        augmentation=AugmentationConfig(
            horizontal_flip_prob=1.0,
            rotation_degrees=10.0,
            scale_range=(0.8, 1.2),
            elastic_blur_sigma=0.3,
        ),
    )
    module = EmotionDataModule(config)
    module.setup(force_recompute_stats=True)

    assert module.stats.mean is not None
    assert module.stats.std is not None
    assert module.class_weights.shape == (len(CLASS_NAMES),)

    train_batch = next(module.train_batches(rng_seed=0))
    assert train_batch[0].shape[1:] == (48, 48, 1)

    val_batch = next(module.val_batches(batch_size=3))
    assert val_batch[0].dtype == jnp.float32

    test_batch = next(module.test_batches(batch_size=5))
    assert test_batch[1].shape[0] <= 5

    dropped = list(module.train_batches(batch_size=5, drop_last=True))
    assert all(batch[0].shape[0] == 5 for batch in dropped)

    counts = module.split_counts()
    assert set(counts["train"].keys()) == set(CLASS_NAMES)


def test_dataset_statistics_cache(tmp_path: Path, tiny_dataset: Path) -> None:
    """Test that dataset statistics are cached and reused."""
    samples = _scan_split(tiny_dataset, split="train")
    cache_path = tmp_path / "stats.json"
    stats = compute_dataset_statistics(
        samples, cache_path=cache_path, force=True
    )
    cached = compute_dataset_statistics(
        samples, cache_path=cache_path, force=False
    )
    assert stats.mean == cached.mean
    assert stats.std == cached.std


def test_stratified_split_edge_cases() -> None:
    """Test that stratified split handles minimal and multiple samples."""
    indices = [0]
    one_sample_train, one_sample_val = stratified_split(
        [type("S", (), {"label": 0})()], val_ratio=0.5, seed=0
    )
    assert one_sample_train == indices
    assert one_sample_val == []

    samples = [type("S", (), {"label": i % 2})() for i in range(10)]
    train_idx, val_idx = stratified_split(samples, val_ratio=0.2, seed=0)
    assert set(train_idx + val_idx) == set(range(10))


def test_class_distribution_and_weights(tiny_dataset: Path) -> None:
    """Test class distribution reporting and normalized weight computation."""
    samples = _scan_split(tiny_dataset, split="train")
    distribution = compute_class_distribution(samples)
    assert distribution.keys() == set(CLASS_NAMES)

    weights = compute_class_weights(samples, num_classes=len(CLASS_NAMES))
    np.testing.assert_allclose(np.sum(np.asarray(weights)), 1.0, atol=1e-6)


def test_augmentation_and_normalization() -> None:
    """Test augmentation pipeline and normalization helpers."""
    config = AugmentationConfig(
        horizontal_flip_prob=1.0,
        rotation_degrees=5.0,
        scale_range=(0.9, 1.1),
        elastic_blur_sigma=0.2,
    )
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, num=48 * 48, dtype=np.float32).reshape(
        (48, 48, 1)
    )
    augmented = apply_augmentations(base, rng, config)
    assert augmented.shape == base.shape

    normalized = normalize_image(augmented, mean=0.5, std=0.25)
    assert np.isclose(
        np.mean(normalized), (np.mean(augmented) - 0.5) / 0.25, atol=1e-3
    )


def test_scan_split_handles_unknown_class(tmp_path: Path) -> None:
    """Test that unknown class directories raise a ValueError."""
    split_dir = tmp_path / "train" / "unknown"
    split_dir.mkdir(parents=True)
    _write_image(split_dir / "img.png", 0)
    with pytest.raises(ValueError):
        _scan_split(tmp_path, split="train")


def test_stratified_split_zero_ratio() -> None:
    """Test that a zero validation ratio returns all train indices."""
    samples = [type("S", (), {"label": 0})() for _ in range(5)]
    train_idx, val_idx = stratified_split(samples, val_ratio=0.0, seed=0)
    assert len(train_idx) == 5
    assert val_idx == []


def test_stats_and_class_weights_require_setup(tmp_path: Path) -> None:
    """Test that accessing stats or weights before setup raises an error."""
    config = DataModuleConfig(data_dir=tmp_path)
    module = EmotionDataModule(config)
    with pytest.raises(RuntimeError):
        _ = module.stats
    with pytest.raises(RuntimeError):
        _ = module.class_weights


def test_setup_skips_stat_recompute_when_provided(
    monkeypatch, tiny_dataset: Path
) -> None:
    """Test that provided statistics skip recomputation."""
    config = DataModuleConfig(data_dir=tiny_dataset, mean=0.25, std=0.5)
    module = EmotionDataModule(config)

    def boom(*args, **kwargs):
        raise AssertionError(
            "compute_dataset_statistics should not be called when mean/std provided"
        )

    monkeypatch.setattr("src.data.compute_dataset_statistics", boom)
    module.setup()
    stats = module.stats
    assert stats.mean == pytest.approx(0.25)
    assert stats.std == pytest.approx(0.5)
    assert stats.num_pixels == 0


def test_iter_batches_empty_returns_no_batches(tmp_path: Path) -> None:
    """Test that empty sample iterators yield no batches."""
    config = DataModuleConfig(data_dir=tmp_path)
    module = EmotionDataModule(config)
    batches = list(
        module._iter_batches(
            [],
            batch_size=4,
            augment=False,
            rng_seed=None,
            shuffle=False,
            drop_last=None,
        )
    )
    assert batches == []


def test_scan_split_missing_directory(tmp_path: Path) -> None:
    """Test that missing directories trigger a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _scan_split(tmp_path, split="train")


def test_scan_split_filters_non_dirs_and_extensions(tmp_path: Path) -> None:
    """Test that non-directory entries and unmatched extensions are ignored."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "readme.txt").write_text("ignore me")
    anger_dir = train_dir / "angry"
    anger_dir.mkdir()
    (anger_dir / "notes.txt").write_text("not an image")
    _write_image(anger_dir / "img.png", 128)

    samples = _scan_split(tmp_path, split="train")
    assert len(samples) == 1
    assert samples[0].path.name == "img.png"


def test_stratified_split_empty_samples_returns_empty() -> None:
    """Test that an empty sample list returns empty splits."""
    train_idx, val_idx = stratified_split([], val_ratio=0.2, seed=0)
    assert train_idx == []
    assert val_idx == []


def test_normalize_image_without_stats_returns_input() -> None:
    """Test that normalization bypasses transformation when stats missing."""
    image = np.linspace(0, 1, num=16, dtype=np.float32).reshape(4, 4, 1)
    assert np.allclose(normalize_image(image, mean=None, std=0.5), image)
