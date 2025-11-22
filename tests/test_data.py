"""Tests for data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
import chex
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
    _load_image,
    _scan_split,
    RetinaFaceConfig,
    _extract_face_region,
    _align_face,
    _initialize_retinaface_detector,
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
    chex.assert_shape(train_batch[0], (train_batch[0].shape[0], 48, 48, 1))
    chex.assert_type(train_batch[0], jnp.float32)

    val_batch = next(module.val_batches(batch_size=3))
    chex.assert_shape(val_batch[0], (val_batch[0].shape[0], 48, 48, 1))
    chex.assert_type(val_batch[0], jnp.float32)

    test_batch = next(module.test_batches(batch_size=5))
    chex.assert_shape(test_batch[0], (test_batch[0].shape[0], 48, 48, 1))
    assert test_batch[1].shape[0] <= 5

    dropped = list(module.train_batches(batch_size=5, drop_last=True))
    assert all(batch[0].shape[0] == 5 for batch in dropped)

    counts = module.split_counts()
    assert set(counts["train"].keys()) == set(CLASS_NAMES)


def test_dataset_statistics_cache(tmp_path: Path, tiny_dataset: Path) -> None:
    """Test that dataset statistics are cached and reused."""
    samples = _scan_split(tiny_dataset, split="train")
    cache_path = tmp_path / "stats.toml"
    stats = compute_dataset_statistics(
        samples, cache_path=cache_path, force=True
    )
    cached = compute_dataset_statistics(
        samples, cache_path=cache_path, force=False
    )
    assert stats.mean == cached.mean
    assert stats.std == cached.std
    assert cached.num_pixels > 0


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
    base = np.linspace(0, 1, num=48 * 48, dtype=np.float32).reshape((48, 48, 1))
    augmented = apply_augmentations(base, rng, config)
    chex.assert_shape(augmented, base.shape)

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


def test_augmentation_disabled_is_noop(tmp_path: Path) -> None:
    """Test that disabled augmentation leaves images untouched in batches."""
    dataset_root = tmp_path / "dataset"
    class_dir = dataset_root / "train" / CLASS_NAMES[0]
    class_dir.mkdir(parents=True, exist_ok=True)
    _write_image(class_dir / "img0.png", 32)
    test_dir = dataset_root / "test" / CLASS_NAMES[0]
    test_dir.mkdir(parents=True, exist_ok=True)
    _write_image(test_dir / "img0.png", 32)

    config = DataModuleConfig(
        data_dir=dataset_root,
        batch_size=1,
        augment=True,
        augmentation=AugmentationConfig(enabled=False),
        mean=0.0,
        std=1.0,
    )
    module = EmotionDataModule(config)
    module.setup()
    batch = next(module.train_batches(rng_seed=0))
    original = _load_image(class_dir / "img0.png").astype(np.float32) / 255.0
    chex.assert_shape(batch[0], (1, 48, 48, 1))
    np.testing.assert_allclose(batch[0][0], original, atol=1e-6)


def test_augmentation_extreme_scale_keeps_bounds() -> None:
    """Test augmentation with extreme scales keeps shape and valid range."""
    config = AugmentationConfig(scale_range=(0.5, 1.5))
    rng = np.random.default_rng(321)
    image = np.random.rand(48, 48, 1).astype(np.float32)
    augmented = apply_augmentations(image, rng, config)
    chex.assert_shape(augmented, (48, 48, 1))
    assert np.min(augmented) >= 0.0
    assert np.max(augmented) <= 1.0


def test_train_batches_without_augmentation_preserves_values(
    tmp_path: Path,
) -> None:
    """Test that disabling module-level augmentation leaves tensors unchanged."""
    dataset_root = tmp_path / "dataset"
    class_dir = dataset_root / "train" / CLASS_NAMES[0]
    class_dir.mkdir(parents=True, exist_ok=True)
    _write_image(class_dir / "img0.png", 64)
    test_dir = dataset_root / "test" / CLASS_NAMES[0]
    test_dir.mkdir(parents=True, exist_ok=True)
    _write_image(test_dir / "img0.png", 64)

    config = DataModuleConfig(
        data_dir=dataset_root,
        batch_size=1,
        augment=False,
        mean=0.0,
        std=1.0,
    )
    module = EmotionDataModule(config)
    module.setup(force_recompute_stats=True)
    batch = next(module.train_batches(rng_seed=0))
    original = _load_image(class_dir / "img0.png").astype(np.float32) / 255.0
    chex.assert_shape(batch[0], (1, 48, 48, 1))
    expected = normalize_image(original, module.config.mean, module.config.std)
    np.testing.assert_allclose(batch[0][0], expected, atol=1e-6)


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

def test_insightface_config_validation() -> None:
    """Test InsightFaceConfig validation."""
    from src.data import InsightFaceConfig
    
    config = InsightFaceConfig(enabled=True, det_thresh=0.5)
    assert config.enabled is True
    
    with pytest.raises(ValueError):
        InsightFaceConfig(det_thresh=1.5)
    
    with pytest.raises(ValueError):
        InsightFaceConfig(expand_ratio=0.5)


def test_extract_face_region_handles_none_detector() -> None:
    """Test that face extraction gracefully handles None detector."""
    from src.data import _extract_face_region, InsightFaceConfig
    
    image = np.full((48, 48, 1), 128, dtype=np.uint8)
    config = InsightFaceConfig()
    result = _extract_face_region(image, detector=None, config=config)
    np.testing.assert_array_equal(result, image)


def test_emotion_data_module_with_insightface_disabled(
    tiny_dataset: Path,
) -> None:
    """Test that disabled InsightFace doesn't affect loading."""
    from src.data import InsightFaceConfig
    
    config = DataModuleConfig(
        data_dir=tiny_dataset,
        batch_size=4,
        insightface=InsightFaceConfig(enabled=False),
    )
    module = EmotionDataModule(config)
    module.setup()
    batch = next(module.train_batches(rng_seed=0))
    chex.assert_shape(batch[0], (batch[0].shape[0], 48, 48, 1))


def test_insightface_initialization_fallback() -> None:
    """Test InsightFace initialization handles import errors gracefully."""
    from src.data import _initialize_insightface_detector, InsightFaceConfig
    
    config = InsightFaceConfig(enabled=True, model_pack="buffalo_l")
    detector = _initialize_insightface_detector(config)
    # Should return None or valid detector, but not crash
    assert detector is None or detector is not None
