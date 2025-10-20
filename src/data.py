"""Data loading, augmentation, and preprocessing utilities for emotion detection."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - fallback for older Pillow
    RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]


CLASS_NAMES: Tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)
CLASS_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}
IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
DEFAULT_STATS_FILENAME = "stats_train.json"


@dataclass(frozen=True)
class DatasetStats:
    """Dataset-level statistics used for normalization.

    Attributes:
        mean: Mean pixel intensity across the dataset in the [0, 1] range.
        std: Standard deviation of pixel intensities.
        num_pixels: Total number of pixels contributing to the statistics.
    """

    mean: float
    std: float
    num_pixels: int


@dataclass
class AugmentationConfig:
    """Configuration options controlling stochastic image augmentations.

    Attributes:
        horizontal_flip_prob: Probability of performing a horizontal flip.
        rotation_degrees: Maximum absolute rotation in degrees.
        scale_range: Inclusive range of random resized crop scales.
        elastic_blur_sigma: Optional Gaussian blur sigma for elastic effect.
        enabled: Whether augmentation is enabled.
    """

    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 15.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    elastic_blur_sigma: Optional[float] = None
    enabled: bool = True

    def clamp(self) -> AugmentationConfig:
        """Return a sanitized copy of the augmentation configuration.

        Returns:
            AugmentationConfig: Clamped augmentation configuration.
        """
        lo, hi = self.scale_range
        lo = max(0.1, float(lo))
        hi = max(lo, float(hi))
        return AugmentationConfig(
            horizontal_flip_prob=float(
                np.clip(self.horizontal_flip_prob, 0.0, 1.0)
            ),
            rotation_degrees=float(max(0.0, self.rotation_degrees)),
            scale_range=(lo, hi),
            elastic_blur_sigma=self.elastic_blur_sigma
            if (self.elastic_blur_sigma or 0.0) > 0
            else None,
            enabled=self.enabled,
        )


@dataclass(frozen=True)
class Sample:
    """Represents a single image example and its numeric label.

    Attributes:
        path: Filesystem path to the image file.
        label: Integer label matching class index.
    """

    path: Path
    label: int


@dataclass
class DataModuleConfig:
    """Runtime configuration governing dataset preparation.

    Attributes:
        data_dir: Root directory containing train/test folders.
        batch_size: Batch size used for loaders.
        val_ratio: Fraction of training data reserved for validation.
        seed: PRNG seed applied to deterministic operations.
        drop_last: Whether to drop incomplete batches.
        mean: Optional precomputed dataset mean.
        std: Optional precomputed dataset standard deviation.
        augment: If False, augmentation is skipped even for training.
        augmentation: Augmentation-specific configuration.
        stats_cache_path: Optional path to cache statistics JSON.
    """

    data_dir: Path
    batch_size: int = 128
    val_ratio: float = 0.1
    seed: int = 0
    drop_last: bool = False
    mean: Optional[float] = None
    std: Optional[float] = None
    augment: bool = True
    augmentation: AugmentationConfig = dataclasses.field(
        default_factory=AugmentationConfig
    )
    stats_cache_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Normalize path-like fields after dataclass initialization."""
        self.data_dir = Path(self.data_dir)
        if self.stats_cache_path is not None:
            self.stats_cache_path = Path(self.stats_cache_path)


class EmotionDataModule:
    """Loads, splits, and batches the emotion recognition dataset."""

    def __init__(self, config: DataModuleConfig) -> None:
        """Initializes the data module with its configuration.

        Args:
            config: Data module configuration.
        """
        self.config = config
        self._train_samples: list[Sample] = []
        self._val_samples: list[Sample] = []
        self._test_samples: list[Sample] = []
        self._stats: Optional[DatasetStats] = None
        self._class_weights: Optional[jnp.ndarray] = None

    @property
    def stats(self) -> DatasetStats:
        """Returns dataset statistics computed during setup.

        Returns:
            DatasetStats: Mean and standard deviation values.

        Raises:
            RuntimeError: If setup has not been executed.
        """
        if self._stats is None:
            raise RuntimeError(
                "EmotionDataModule.setup must be called before accessing stats."
            )
        return self._stats

    @property
    def class_weights(self) -> jnp.ndarray:
        """Returns class-balanced weights derived from the training split.

        Returns:
            jnp.ndarray: Normalized class weights.

        Raises:
            RuntimeError: If setup has not been executed.
        """
        if self._class_weights is None:
            raise RuntimeError(
                "EmotionDataModule.setup must be called before accessing class weights."
            )
        return self._class_weights

    def setup(self, force_recompute_stats: bool = False) -> None:
        """Initializes dataset splits, statistics, and class weights.

        Args:
            force_recompute_stats: If True, recompute statistics even if cached.
        """
        cfg = self.config
        train_samples = _scan_split(cfg.data_dir, split="train")
        test_samples = _scan_split(cfg.data_dir, split="test")

        train_indices, val_indices = stratified_split(
            train_samples, cfg.val_ratio, cfg.seed
        )
        self._train_samples = [train_samples[i] for i in train_indices]
        self._val_samples = [train_samples[i] for i in val_indices]
        self._test_samples = test_samples

        stats_cache = (
            cfg.stats_cache_path or cfg.data_dir / DEFAULT_STATS_FILENAME
        )
        if (
            cfg.mean is not None
            and cfg.std is not None
            and not force_recompute_stats
        ):
            derived_stats = DatasetStats(
                mean=float(cfg.mean), std=float(cfg.std), num_pixels=0
            )
        else:
            derived_stats = compute_dataset_statistics(
                self._train_samples,
                cache_path=stats_cache,
                force=force_recompute_stats,
            )
            cfg.mean = derived_stats.mean
            cfg.std = derived_stats.std

        self._stats = derived_stats
        self._class_weights = compute_class_weights(
            self._train_samples, len(CLASS_NAMES)
        )

    def train_batches(
        self,
        *,
        batch_size: Optional[int] = None,
        rng_seed: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Yield normalized batches for the training split.

        Args:
            batch_size: Optional override for batch size.
            rng_seed: Seed for shuffle and augmentation randomness.
            drop_last: Optional override for dropping incomplete batches.

        Returns:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: Batched images and labels.
        """
        return self._iter_batches(
            self._train_samples,
            batch_size=batch_size,
            augment=self.config.augment,
            rng_seed=rng_seed,
            shuffle=True,
            drop_last=drop_last,
        )

    def val_batches(
        self,
        *,
        batch_size: Optional[int] = None,
        rng_seed: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Yield normalized batches for the validation split.

        Args:
            batch_size: Optional override for batch size.
            rng_seed: Seed controlling deterministic iteration order.
            drop_last: Optional override for dropping incomplete batches.

        Returns:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: Batched images and labels.
        """
        return self._iter_batches(
            self._val_samples,
            batch_size=batch_size,
            augment=False,
            rng_seed=rng_seed,
            shuffle=False,
            drop_last=drop_last,
        )

    def test_batches(
        self,
        *,
        batch_size: Optional[int] = None,
        rng_seed: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Yield normalized batches for the test split.

        Args:
            batch_size: Optional override for batch size.
            rng_seed: Seed controlling deterministic iteration order.
            drop_last: Optional override for dropping incomplete batches.

        Returns:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: Batched images and labels.
        """
        return self._iter_batches(
            self._test_samples,
            batch_size=batch_size,
            augment=False,
            rng_seed=rng_seed,
            shuffle=False,
            drop_last=drop_last,
        )

    def _iter_batches(
        self,
        samples: Sequence[Sample],
        *,
        batch_size: Optional[int],
        augment: bool,
        rng_seed: Optional[int],
        shuffle: bool,
        drop_last: Optional[bool],
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Internal helper that batches and normalizes a sample collection.

        Args:
            samples: Sequence of Sample instances to iterate.
            batch_size: Effective batch size for iteration.
            augment: Whether to apply stochastic augmentation.
            rng_seed: Seed for deterministic shuffling/augmentations.
            shuffle: Whether to shuffle sample indices.
            drop_last: Whether to drop incomplete final batch.

        Returns:
            Iterator[Tuple[jnp.ndarray, jnp.ndarray]]: Batched tensors.
        """
        if not samples:
            return iter(())

        batch_size = batch_size or self.config.batch_size
        drop_last = (
            drop_last if drop_last is not None else self.config.drop_last
        )

        rng = np.random.default_rng(
            rng_seed if rng_seed is not None else self.config.seed
        )
        indices = np.arange(len(samples))
        if shuffle:
            rng.shuffle(indices)

        aug_cfg = self.config.augmentation.clamp()
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            if len(batch_indices) < batch_size and drop_last:
                continue

            images: list[np.ndarray] = []
            labels: list[int] = []
            for idx in batch_indices:
                sample = samples[idx]
                image = _load_image(sample.path)
                image = image.astype(np.float32) / 255.0
                if augment and aug_cfg.enabled:
                    image = apply_augmentations(image, rng, aug_cfg)
                image = normalize_image(
                    image, self.config.mean, self.config.std
                )
                images.append(image)
                labels.append(sample.label)

            batch_images = jnp.asarray(
                np.stack(images, axis=0), dtype=jnp.float32
            )
            batch_labels = jnp.asarray(np.array(labels, dtype=np.int32))
            yield batch_images, batch_labels

    def split_counts(self) -> dict[str, dict[str, int]]:
        """Return class distributions for each split.

        Returns:
            dict[str, dict[str, int]]: Mapping of split names to class counts.
        """
        return {
            "train": compute_class_distribution(self._train_samples),
            "val": compute_class_distribution(self._val_samples),
            "test": compute_class_distribution(self._test_samples),
        }


def _scan_split(data_dir: Path, *, split: str) -> list[Sample]:
    """Scan a dataset split directory and collect samples.

    Args:
        data_dir: Root dataset directory.
        split: Split name (e.g., ``"train"`` or ``"test"``).

    Returns:
        list[Sample]: Collected samples for the split.

    Raises:
        FileNotFoundError: If the split directory does not exist.
        ValueError: If an unexpected class subdirectory is encountered.
    """
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected split directory at {split_dir}")

    samples: list[Sample] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        if class_name not in CLASS_TO_INDEX:
            raise ValueError(f"Unrecognized class directory: {class_name}")
        label = CLASS_TO_INDEX[class_name]
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append(Sample(path=path, label=label))
    return samples


def stratified_split(
    samples: Sequence[Sample],
    val_ratio: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    """Create stratified train and validation index lists.

    Args:
        samples: Sequence of samples within the training split.
        val_ratio: Desired validation ratio (clamped to [0, 0.5]).
        seed: Random seed used when shuffling indices.

    Returns:
        Tuple[list[int], list[int]]: Training indices, validation indices.
    """
    if not samples:
        return [], []
    val_ratio = max(0.0, min(0.5, float(val_ratio)))
    if val_ratio == 0.0:
        indices = list(range(len(samples)))
        return indices, []

    rng = np.random.default_rng(seed)
    per_class: dict[int, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        per_class[sample.label].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label, idxs in per_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        if len(idxs) == 1:
            train_indices.append(idxs[0])
            continue
        proposed = max(1, int(round(len(idxs) * val_ratio)))
        proposed = min(proposed, len(idxs) - 1)
        val_indices.extend(idxs[:proposed])
        train_indices.extend(idxs[proposed:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_class_distribution(samples: Sequence[Sample]) -> dict[str, int]:
    """Count the number of samples per class.

    Args:
        samples: Sequence of samples to count.

    Returns:
        dict[str, int]: Mapping from class name to sample count.
    """
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        counts[CLASS_NAMES[sample.label]] += 1
    return counts


def compute_class_weights(
    samples: Sequence[Sample], num_classes: int
) -> jnp.ndarray:
    """Derive class-balanced weights for loss scaling.

    Args:
        samples: Sequence of training samples.
        num_classes: Total number of classes.

    Returns:
        jnp.ndarray: Normalized class weights.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in samples:
        counts[sample.label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.sum()
    return jnp.asarray(weights, dtype=jnp.float32)


def compute_dataset_statistics(
    samples: Sequence[Sample],
    *,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> DatasetStats:
    """Compute or load normalization statistics for a sample collection.

    Args:
        samples: Sequence of samples contributing to statistics.
        cache_path: Optional JSON cache to read/write.
        force: If True, recompute statistics ignoring any cache.

    Returns:
        DatasetStats: Mean, standard deviation, and pixel count.
    """
    if cache_path is not None and cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return DatasetStats(
            mean=float(data["mean"]),
            std=float(data["std"]),
            num_pixels=int(data.get("num_pixels", 0)),
        )

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0
    for sample in samples:
        image = _load_image(sample.path)
        arr = image.astype(np.float64) / 255.0
        pixel_sum += float(arr.sum())
        pixel_sq_sum += float(np.square(arr).sum())
        total_pixels += arr.size

    mean = pixel_sum / total_pixels
    variance = max(pixel_sq_sum / total_pixels - mean**2, 1e-12)
    std = math.sqrt(variance)
    stats = DatasetStats(
        mean=float(mean), std=float(std), num_pixels=int(total_pixels)
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(stats), fh, indent=2)

    return stats


def apply_augmentations(
    image: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> np.ndarray:
    """Apply stochastic augmentations to a single image sample.

    Args:
        image: Input image array shaped ``(H, W, C)`` with ``C=1``.
        rng: NumPy random generator controlling augmentation.
        config: Augmentation configuration to apply.

    Returns:
        np.ndarray: Augmented image array.
    """
    aug_cfg = config.clamp()
    augmented = image

    if aug_cfg.scale_range is not None:
        augmented = _random_resized_crop(augmented, rng, aug_cfg.scale_range)

    if (
        aug_cfg.horizontal_flip_prob > 0.0
        and rng.random() < aug_cfg.horizontal_flip_prob
    ):
        augmented = augmented[:, ::-1, :]

    if aug_cfg.rotation_degrees > 0.0:
        angle = float(
            rng.uniform(-aug_cfg.rotation_degrees, aug_cfg.rotation_degrees)
        )
        if abs(angle) > 1e-3:
            rotated = _rotate_image(augmented[..., 0], angle)
            augmented = rotated[..., None]

    if aug_cfg.elastic_blur_sigma is not None:
        augmented = gaussian_filter(
            augmented,
            sigma=(
                aug_cfg.elastic_blur_sigma,
                aug_cfg.elastic_blur_sigma,
                0.0,
            ),
            mode="reflect",
        )

    return np.clip(augmented, 0.0, 1.0)


def normalize_image(
    image: np.ndarray, mean: Optional[float], std: Optional[float]
) -> np.ndarray:
    """Normalize image tensor by mean and standard deviation.

    Args:
        image: Image array with values in [0, 1].
        mean: Dataset mean or ``None`` to skip normalization.
        std: Dataset standard deviation or ``None`` to skip normalization.

    Returns:
        np.ndarray: Normalized image array.
    """
    if mean is None or std is None:
        return image
    return (image - float(mean)) / float(std)


def _load_image(path: Path) -> np.ndarray:
    """Load an image from disk as a grayscale NumPy array.

    Args:
        path: Filesystem path to the image.

    Returns:
        np.ndarray: Unsigned byte image with shape ``(H, W, 1)``.
    """
    with Image.open(path) as img:
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.uint8)
    arr = arr[:, :, None] if arr.ndim == 2 else arr
    return arr


def _random_resized_crop(
    image: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float],
) -> np.ndarray:
    """Perform a random resized crop similar to torchvision implementation.

    Args:
        image: Input single-channel image array.
        rng: NumPy random generator.
        scale_range: Inclusive scale factor range to sample from.

    Returns:
        np.ndarray: Cropped (and possibly rescaled) image.
    """
    h, w, c = image.shape
    assert c == 1, "Random resized crop assumes single-channel images."

    scale_min, scale_max = scale_range
    area = float(h * w)

    scale = float(rng.uniform(scale_min, scale_max))
    scale = max(scale, 0.1)
    target_area = np.clip(scale, 0.05, 1.5) * area
    target_side = int(round(math.sqrt(target_area)))
    target_side = max(4, target_side)

    if target_side <= h:
        max_top = h - target_side
        max_left = w - target_side
        top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
        left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
        crop = image[top : top + target_side, left : left + target_side]
    else:
        pad_h = target_side - h
        pad_w = target_side - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
        top = int(rng.integers(0, padded.shape[0] - target_side + 1))
        left = int(rng.integers(0, padded.shape[1] - target_side + 1))
        crop = padded[top : top + target_side, left : left + target_side]

    if crop.shape[0] != h or crop.shape[1] != w:
        pil_img = Image.fromarray(
            (np.clip(crop, 0.0, 1.0) * 255).astype(np.uint8).squeeze(axis=-1)
        )
        pil_img = pil_img.resize((w, h), resample=RESAMPLE_BILINEAR)
        crop = np.asarray(pil_img, dtype=np.float32)[..., None] / 255.0
    return crop


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a single-channel image around its center.

    Args:
        image: 2D image array with values in [0, 1].
        angle: Rotation angle in degrees.

    Returns:
        np.ndarray: Rotated image array.
    """
    pil_img = Image.fromarray(
        (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    )
    rotated = pil_img.rotate(angle, resample=RESAMPLE_BILINEAR, fillcolor=0)
    return np.asarray(rotated, dtype=np.float32) / 255.0
