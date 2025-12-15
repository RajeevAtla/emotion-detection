"""Data loading, augmentation, and preprocessing utilities for emotion detection."""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import jax.numpy as jnp
import numpy as np
import tomli_w
import tomllib
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


CLASS_NAMES: Tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)
CLASS_TO_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}
IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
DEFAULT_STATS_FILENAME = "stats_train.toml"


@dataclass(frozen=True)
class DatasetStats:
    mean: float
    std: float
    num_pixels: int


@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class AugmentationConfig:
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 15.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Optional[Tuple[float, float]] = None
    contrast_range: Optional[Tuple[float, float]] = None
    gaussian_blur_prob: float = 0.0
    gaussian_blur_sigma: float = 1.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    cutmix_prob: float = 0.5
    enabled: bool = True

    def clamp(self) -> AugmentationConfig:
        lo, hi = self.scale_range
        lo = max(0.1, float(lo))
        hi = max(lo, float(hi))
        return AugmentationConfig(
            horizontal_flip_prob=float(np.clip(self.horizontal_flip_prob, 0.0, 1.0)),
            rotation_degrees=float(max(0.0, self.rotation_degrees)),
            scale_range=(lo, hi),
            brightness_range=self.brightness_range,
            contrast_range=self.contrast_range,
            gaussian_blur_prob=float(np.clip(self.gaussian_blur_prob, 0.0, 1.0)),
            gaussian_blur_sigma=max(0.1, self.gaussian_blur_sigma),
            mixup_alpha=max(0.0, self.mixup_alpha),
            cutmix_alpha=max(0.0, self.cutmix_alpha),
            cutmix_prob=float(np.clip(self.cutmix_prob, 0.0, 1.0)),
            enabled=self.enabled,
        )


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    dataset: str = "unknown"


@dataclass
class DataModuleConfig:
    data_dir: Path
    datasets: List[DatasetConfig] = field(default_factory=list)
    batch_size: int = 128
    val_ratio: float = 0.1
    seed: int = 0
    drop_last: bool = False
    mean: Optional[float] = None
    std: Optional[float] = None
    augment: bool = True
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    stats_cache_path: Optional[Path] = None
    image_size: Tuple[int, int] = (48, 48)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.stats_cache_path is not None:
            self.stats_cache_path = Path(self.stats_cache_path)
        if not self.datasets:
            self.datasets = [DatasetConfig(name="fer2013", data_dir="fer2013", weight=1.0)]


def _load_image(path: Path, target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """Load image, convert to grayscale, and resize to target size."""
    with Image.open(path) as img:
        img = img.convert("L")
        # Resize to target size if different
        if img.size != target_size:
            img = img.resize(target_size, resample=RESAMPLE_BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
    arr = arr[:, :, None] if arr.ndim == 2 else arr
    return arr


class EmotionDataModule:
    def __init__(self, config: DataModuleConfig) -> None:
        self.config = config
        self._train_samples: List[Sample] = []
        self._val_samples: List[Sample] = []
        self._test_samples: List[Sample] = []
        self._stats: Optional[DatasetStats] = None
        self._class_weights: Optional[jnp.ndarray] = None
        self._sample_weights: Optional[np.ndarray] = None
        self._dataset_info: Dict[str, Dict[str, int]] = {}

    @property
    def stats(self) -> DatasetStats:
        if self._stats is None:
            raise RuntimeError("EmotionDataModule.setup must be called first.")
        return self._stats

    @property
    def class_weights(self) -> jnp.ndarray:
        if self._class_weights is None:
            raise RuntimeError("EmotionDataModule.setup must be called first.")
        return self._class_weights

    def setup(self, force_recompute_stats: bool = False) -> None:
        cfg = self.config
        all_train_samples: List[Sample] = []
        all_test_samples: List[Sample] = []
        dataset_weights: Dict[str, float] = {}

        print("=" * 60)
        print("Loading datasets...")
        print("=" * 60)

        for ds_cfg in cfg.datasets:
            if not ds_cfg.enabled:
                print(f"  {ds_cfg.name}: DISABLED")
                continue

            ds_path = cfg.data_dir / ds_cfg.data_dir
            if not ds_path.exists():
                print(f"  {ds_cfg.name}: NOT FOUND at {ds_path}")
                continue

            train_samples = _scan_split(ds_path, split="train", dataset_name=ds_cfg.name)
            test_samples = _scan_split(ds_path, split="test", dataset_name=ds_cfg.name)

            self._dataset_info[ds_cfg.name] = {
                "train": len(train_samples),
                "test": len(test_samples),
                "weight": ds_cfg.weight,
            }

            all_train_samples.extend(train_samples)
            all_test_samples.extend(test_samples)
            dataset_weights[ds_cfg.name] = ds_cfg.weight

            print(f"  {ds_cfg.name}: {len(train_samples)} train, {len(test_samples)} test (weight={ds_cfg.weight})")

        if not all_train_samples:
            raise FileNotFoundError("No training samples found in any dataset.")

        train_indices, val_indices = stratified_split(all_train_samples, cfg.val_ratio, cfg.seed)
        self._train_samples = [all_train_samples[i] for i in train_indices]
        self._val_samples = [all_train_samples[i] for i in val_indices]
        self._test_samples = all_test_samples

        self._sample_weights = self._compute_sample_weights(self._train_samples, dataset_weights)

        stats_cache = cfg.stats_cache_path or cfg.data_dir / DEFAULT_STATS_FILENAME
        if cfg.mean is not None and cfg.std is not None and not force_recompute_stats:
            derived_stats = DatasetStats(mean=float(cfg.mean), std=float(cfg.std), num_pixels=0)
        else:
            derived_stats = compute_dataset_statistics(
                self._train_samples, cache_path=stats_cache, force=force_recompute_stats,
                image_size=cfg.image_size
            )
            cfg.mean = derived_stats.mean
            cfg.std = derived_stats.std

        self._stats = derived_stats
        self._class_weights = compute_class_weights(self._train_samples, len(CLASS_NAMES))

        print("-" * 60)
        print(f"Total: {len(self._train_samples)} train, {len(self._val_samples)} val, {len(self._test_samples)} test")
        print(f"Stats: mean={self._stats.mean:.4f}, std={self._stats.std:.4f}")
        print("=" * 60)

    def _compute_sample_weights(self, samples: List[Sample], dataset_weights: Dict[str, float]) -> np.ndarray:
        if not dataset_weights:
            return np.ones(len(samples), dtype=np.float32)

        dataset_counts = defaultdict(int)
        for s in samples:
            dataset_counts[s.dataset] += 1

        total_weight = sum(dataset_weights.values())
        normalized_weights = {k: v / total_weight for k, v in dataset_weights.items()}

        weights = np.zeros(len(samples), dtype=np.float32)
        for i, s in enumerate(samples):
            ds_weight = normalized_weights.get(s.dataset, 1.0)
            ds_count = dataset_counts[s.dataset]
            weights[i] = ds_weight / ds_count if ds_count > 0 else 1.0

        weights = weights / weights.sum() * len(samples)
        return weights

    def train_batches(
        self,
        *,
        batch_size: Optional[int] = None,
        rng_seed: Optional[int] = None,
        drop_last: Optional[bool] = None,
        use_weighted_sampling: bool = True,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        return self._iter_batches(
            self._train_samples,
            batch_size=batch_size,
            augment=self.config.augment,
            rng_seed=rng_seed,
            shuffle=True,
            drop_last=drop_last,
            sample_weights=self._sample_weights if use_weighted_sampling else None,
        )

    def val_batches(
        self,
        *,
        batch_size: Optional[int] = None,
        rng_seed: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
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
        sample_weights: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        if not samples:
            return iter(())

        batch_size = batch_size or self.config.batch_size
        drop_last = drop_last if drop_last is not None else self.config.drop_last

        rng = np.random.default_rng(rng_seed if rng_seed is not None else self.config.seed)

        if sample_weights is not None and shuffle:
            probs = sample_weights / sample_weights.sum()
            indices = rng.choice(len(samples), size=len(samples), replace=True, p=probs)
        else:
            indices = np.arange(len(samples))
            if shuffle:
                rng.shuffle(indices)

        aug_cfg = self.config.augmentation.clamp()

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            if len(batch_indices) < batch_size and drop_last:
                continue

            images: List[np.ndarray] = []
            labels: List[int] = []

            for idx in batch_indices:
                sample = samples[idx]
                image = _load_image(sample.path, target_size=self.config.image_size)
                image = image.astype(np.float32) / 255.0

                if augment and aug_cfg.enabled:
                    image = apply_augmentations(image, rng, aug_cfg)

                image = normalize_image(image, self.config.mean, self.config.std)
                images.append(image)
                labels.append(sample.label)

            batch_images = jnp.asarray(np.stack(images, axis=0), dtype=jnp.float32)
            batch_labels = jnp.asarray(np.array(labels, dtype=np.int32))
            yield batch_images, batch_labels

    def split_counts(self) -> Dict[str, Dict[str, int]]:
        return {
            "train": compute_class_distribution(self._train_samples),
            "val": compute_class_distribution(self._val_samples),
            "test": compute_class_distribution(self._test_samples),
        }


def _scan_split(data_dir: Path, *, split: str, dataset_name: str = "unknown") -> List[Sample]:
    split_dir = data_dir / split
    if not split_dir.exists():
        return []

    samples: List[Sample] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        if class_name not in CLASS_TO_INDEX:
            continue
        label = CLASS_TO_INDEX[class_name]
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append(Sample(path=path, label=label, dataset=dataset_name))
    return samples


def stratified_split(
    samples: Sequence[Sample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not samples:
        return [], []
    val_ratio = max(0.0, min(0.5, float(val_ratio)))
    if val_ratio == 0.0:
        return list(range(len(samples))), []

    rng = np.random.default_rng(seed)
    per_class: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        per_class[sample.label].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
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


def compute_class_distribution(samples: Sequence[Sample]) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        counts[CLASS_NAMES[sample.label]] += 1
    return counts


def compute_class_weights(samples: Sequence[Sample], num_classes: int) -> jnp.ndarray:
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
    image_size: Tuple[int, int] = (48, 48),
) -> DatasetStats:
    if cache_path is not None and cache_path.exists() and not force:
        with cache_path.open("rb") as fh:
            data = tomllib.load(fh)
        return DatasetStats(
            mean=float(data["mean"]),
            std=float(data["std"]),
            num_pixels=int(data.get("num_pixels", 0)),
        )

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0
    for sample in samples:
        image = _load_image(sample.path, target_size=image_size)
        arr = image.astype(np.float64) / 255.0
        pixel_sum += float(arr.sum())
        pixel_sq_sum += float(np.square(arr).sum())
        total_pixels += arr.size

    mean = pixel_sum / total_pixels
    variance = max(pixel_sq_sum / total_pixels - mean**2, 1e-12)
    std = math.sqrt(variance)
    stats = DatasetStats(mean=float(mean), std=float(std), num_pixels=int(total_pixels))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(tomli_w.dumps(dataclasses.asdict(stats)), encoding="utf-8")

    return stats


def apply_augmentations(
    image: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> np.ndarray:
    aug_cfg = config.clamp()
    augmented = image

    if aug_cfg.scale_range is not None:
        augmented = _random_resized_crop(augmented, rng, aug_cfg.scale_range)

    if aug_cfg.horizontal_flip_prob > 0.0 and rng.random() < aug_cfg.horizontal_flip_prob:
        augmented = augmented[:, ::-1, :]

    if aug_cfg.rotation_degrees > 0.0:
        angle = float(rng.uniform(-aug_cfg.rotation_degrees, aug_cfg.rotation_degrees))
        if abs(angle) > 1e-3:
            rotated = _rotate_image(augmented[..., 0], angle)
            augmented = rotated[..., None]

    if aug_cfg.brightness_range is not None:
        lo, hi = aug_cfg.brightness_range
        factor = rng.uniform(lo, hi)
        augmented = augmented * factor

    if aug_cfg.contrast_range is not None:
        lo, hi = aug_cfg.contrast_range
        factor = rng.uniform(lo, hi)
        mean_val = augmented.mean()
        augmented = (augmented - mean_val) * factor + mean_val

    if aug_cfg.gaussian_blur_prob > 0.0 and rng.random() < aug_cfg.gaussian_blur_prob:
        augmented = gaussian_filter(
            augmented,
            sigma=(aug_cfg.gaussian_blur_sigma, aug_cfg.gaussian_blur_sigma, 0.0),
            mode="reflect",
        )

    return np.clip(augmented, 0.0, 1.0)


def normalize_image(
    image: np.ndarray, mean: Optional[float], std: Optional[float]
) -> np.ndarray:
    if mean is None or std is None:
        return image
    return (image - float(mean)) / float(std)


def _random_resized_crop(
    image: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float],
) -> np.ndarray:
    h, w, c = image.shape
    assert c == 1

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
    pil_img = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    rotated = pil_img.rotate(angle, resample=RESAMPLE_BILINEAR, fillcolor=0)
    return np.asarray(rotated, dtype=np.float32) / 255.0
