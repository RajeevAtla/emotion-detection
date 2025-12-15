"""Multi-dataset data loading with weighted sampling and advanced augmentations.

Supports combining FER-2013, CK+, and RAF-DB datasets for emotion detection
with configurable sampling weights and Mixup/CutMix augmentations.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# Standard emotion classes (7 basic emotions)
CLASS_NAMES: Tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)
CLASS_TO_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# RAF-DB label mapping (1-indexed in original)
RAFDB_LABEL_MAP = {
    1: "surprised",
    2: "fearful", 
    3: "disgusted",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class Sample:
    """Single image sample with metadata."""
    path: Path
    label: int
    dataset: str = "fer2013"


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""
    name: str
    data_dir: Path
    weight: float = 1.0
    enabled: bool = True


@dataclass
class AugmentationConfig:
    """Configuration for image augmentations."""
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 15.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gaussian_blur_prob: float = 0.1
    gaussian_blur_sigma: float = 1.0
    enabled: bool = True
    
    # Advanced augmentations
    mixup_alpha: float = 0.2  # 0 to disable
    cutmix_alpha: float = 1.0  # 0 to disable
    cutmix_prob: float = 0.5
    randaugment_n: int = 2
    randaugment_m: int = 9


@dataclass
class MultiDatasetConfig:
    """Configuration for multi-dataset data loading."""
    datasets: list[DatasetConfig] = field(default_factory=list)
    batch_size: int = 64
    val_ratio: float = 0.1
    seed: int = 42
    target_size: Tuple[int, int] = (96, 96)  # Upscale from 48x48
    mean: Optional[float] = None
    std: Optional[float] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    num_workers: int = 8
    prefetch_batches: int = 3
    cache_images: bool = True
    drop_last: bool = True
    
    def __post_init__(self) -> None:
        for ds in self.datasets:
            ds.data_dir = Path(ds.data_dir)


class ImageCache:
    """Thread-safe image cache with LRU eviction."""
    
    def __init__(self, max_size_mb: int = 8192):
        self._cache: dict[Path, np.ndarray] = {}
        self._lock = threading.Lock()
        self._max_bytes = max_size_mb * 1024 * 1024
        self._current_bytes = 0
    
    def get(self, path: Path) -> Optional[np.ndarray]:
        with self._lock:
            arr = self._cache.get(path)
            return arr.copy() if arr is not None else None
    
    def put(self, path: Path, image: np.ndarray) -> None:
        image_bytes = image.nbytes
        with self._lock:
            if path in self._cache:
                return
            if self._current_bytes + image_bytes <= self._max_bytes:
                self._cache[path] = image
                self._current_bytes += image_bytes
    
    def preload(self, paths: Sequence[Path], num_workers: int = 8) -> None:
        """Preload images using thread pool."""
        paths_to_load = [p for p in paths if p not in self._cache]
        if not paths_to_load:
            return
        
        def load_one(path: Path) -> Tuple[Path, Optional[np.ndarray]]:
            try:
                return (path, _load_image(path))
            except Exception:
                return (path, None)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for path, image in executor.map(load_one, paths_to_load):
                if image is not None:
                    self.put(path, image)


def _load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    with Image.open(path) as img:
        img = img.convert("L")  # Grayscale
        arr = np.asarray(img, dtype=np.uint8)
    return arr if arr.ndim == 2 else arr[:, :, 0]


def _resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size using PIL."""
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    return np.asarray(pil_img)


def _scan_dataset(
    data_dir: Path,
    split: str,
    dataset_name: str,
) -> list[Sample]:
    """Scan a dataset directory for samples."""
    split_dir = data_dir / split
    if not split_dir.exists():
        return []
    
    samples = []
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


# =============================================================================
# Advanced Augmentations
# =============================================================================

def apply_mixup(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Mixup augmentation to a batch.
    
    Mixup: x = λ*x_i + (1-λ)*x_j
           y = λ*y_i + (1-λ)*y_j
    """
    if alpha <= 0:
        return images, labels
    
    batch_size = images.shape[0]
    lam = rng.beta(alpha, alpha)
    
    # Shuffle indices
    indices = rng.permutation(batch_size)
    
    # Mix images
    mixed_images = lam * images + (1 - lam) * images[indices]
    
    # Create soft labels
    num_classes = NUM_CLASSES
    labels_onehot = np.eye(num_classes)[labels]
    labels_shuffled = np.eye(num_classes)[labels[indices]]
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled
    
    return mixed_images, mixed_labels


def apply_cutmix(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CutMix augmentation to a batch.
    
    CutMix: Cut a patch from one image and paste onto another,
           with labels mixed proportionally to area.
    """
    if alpha <= 0:
        return images, labels
    
    batch_size, h, w = images.shape[:3]
    lam = rng.beta(alpha, alpha)
    
    # Compute box size
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    # Random box center
    cy = rng.integers(h)
    cx = rng.integers(w)
    
    # Box bounds
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    
    # Shuffle indices
    indices = rng.permutation(batch_size)
    
    # Cut and paste
    mixed_images = images.copy()
    mixed_images[:, y1:y2, x1:x2] = images[indices, y1:y2, x1:x2]
    
    # Adjust lambda based on actual area
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
    
    # Create soft labels
    num_classes = NUM_CLASSES
    labels_onehot = np.eye(num_classes)[labels]
    labels_shuffled = np.eye(num_classes)[labels[indices]]
    mixed_labels = lam_adjusted * labels_onehot + (1 - lam_adjusted) * labels_shuffled
    
    return mixed_images, mixed_labels


def apply_augmentations(
    image: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> np.ndarray:
    """Apply per-image augmentations."""
    if not config.enabled:
        return image
    
    augmented = image.astype(np.float32)
    
    # Horizontal flip
    if config.horizontal_flip_prob > 0 and rng.random() < config.horizontal_flip_prob:
        augmented = augmented[:, ::-1]
    
    # Random rotation
    if config.rotation_degrees > 0:
        angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
        if abs(angle) > 0.5:
            pil_img = Image.fromarray((augmented * 255).astype(np.uint8))
            pil_img = pil_img.rotate(angle, Image.Resampling.BILINEAR, fillcolor=0)
            augmented = np.asarray(pil_img, dtype=np.float32) / 255.0
    
    # Random scale/crop
    if config.scale_range != (1.0, 1.0):
        scale = rng.uniform(*config.scale_range)
        h, w = augmented.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        pil_img = Image.fromarray((augmented * 255).astype(np.uint8))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        scaled = np.asarray(pil_img, dtype=np.float32) / 255.0
        
        # Center crop or pad back to original size
        if scale > 1.0:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            augmented = scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            augmented = np.zeros((h, w), dtype=np.float32)
            augmented[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
    
    # Brightness adjustment
    if config.brightness_range != (1.0, 1.0):
        brightness = rng.uniform(*config.brightness_range)
        augmented = augmented * brightness
    
    # Contrast adjustment
    if config.contrast_range != (1.0, 1.0):
        contrast = rng.uniform(*config.contrast_range)
        mean = augmented.mean()
        augmented = (augmented - mean) * contrast + mean
    
    # Gaussian blur
    if config.gaussian_blur_prob > 0 and rng.random() < config.gaussian_blur_prob:
        augmented = gaussian_filter(augmented, sigma=config.gaussian_blur_sigma)
    
    return np.clip(augmented, 0.0, 1.0)


# =============================================================================
# Multi-Dataset Data Module
# =============================================================================

class MultiDatasetModule:
    """Data module supporting multiple datasets with weighted sampling."""
    
    def __init__(self, config: MultiDatasetConfig) -> None:
        self.config = config
        self._samples_by_dataset: dict[str, list[Sample]] = {}
        self._train_samples: list[Sample] = []
        self._val_samples: list[Sample] = []
        self._test_samples: list[Sample] = []
        self._dataset_weights: dict[str, float] = {}
        self._cache: Optional[ImageCache] = ImageCache() if config.cache_images else None
        self._class_weights: Optional[jnp.ndarray] = None
        self._stats: Optional[Tuple[float, float]] = None
    
    @property
    def class_weights(self) -> jnp.ndarray:
        if self._class_weights is None:
            raise RuntimeError("Call setup() before accessing class_weights")
        return self._class_weights
    
    @property
    def stats(self) -> Tuple[float, float]:
        if self._stats is None:
            raise RuntimeError("Call setup() before accessing stats")
        return self._stats
    
    def setup(self) -> None:
        """Load and prepare all datasets."""
        all_train_samples: list[Sample] = []
        all_test_samples: list[Sample] = []
        
        # Load each enabled dataset
        for ds_config in self.config.datasets:
            if not ds_config.enabled:
                continue
            
            train = _scan_dataset(ds_config.data_dir, "train", ds_config.name)
            test = _scan_dataset(ds_config.data_dir, "test", ds_config.name)
            
            if train:
                self._samples_by_dataset[ds_config.name] = train
                self._dataset_weights[ds_config.name] = ds_config.weight
                all_train_samples.extend(train)
            
            all_test_samples.extend(test)
            print(f"Loaded {ds_config.name}: {len(train)} train, {len(test)} test")
        
        # Stratified split for validation
        self._train_samples, self._val_samples = self._stratified_split(
            all_train_samples,
            self.config.val_ratio,
            self.config.seed,
        )
        self._test_samples = all_test_samples
        
        # Compute statistics
        self._compute_stats()
        self._compute_class_weights()
        
        # Preload images
        if self._cache is not None:
            all_paths = [s.path for s in self._train_samples + self._val_samples + self._test_samples]
            print(f"Preloading {len(all_paths)} images into cache...")
            self._cache.preload(all_paths, num_workers=self.config.num_workers)
        
        print(f"Total: {len(self._train_samples)} train, {len(self._val_samples)} val, {len(self._test_samples)} test")
    
    def _stratified_split(
        self,
        samples: list[Sample],
        val_ratio: float,
        seed: int,
    ) -> Tuple[list[Sample], list[Sample]]:
        """Perform stratified train/val split."""
        if not samples or val_ratio <= 0:
            return samples, []
        
        rng = np.random.default_rng(seed)
        
        # Group by label and dataset
        groups: dict[Tuple[int, str], list[Sample]] = defaultdict(list)
        for s in samples:
            groups[(s.label, s.dataset)].append(s)
        
        train_samples, val_samples = [], []
        for (label, dataset), group_samples in groups.items():
            rng.shuffle(group_samples)
            n_val = max(1, int(len(group_samples) * val_ratio))
            if len(group_samples) == 1:
                train_samples.extend(group_samples)
            else:
                val_samples.extend(group_samples[:n_val])
                train_samples.extend(group_samples[n_val:])
        
        return train_samples, val_samples
    
    def _compute_stats(self) -> None:
        """Compute mean and std from training data."""
        if self.config.mean is not None and self.config.std is not None:
            self._stats = (self.config.mean, self.config.std)
            return
        
        pixel_sum = 0.0
        pixel_sq_sum = 0.0
        total_pixels = 0
        
        for sample in self._train_samples[:5000]:  # Subsample for speed
            if self._cache:
                img = self._cache.get(sample.path)
                if img is None:
                    img = _load_image(sample.path)
            else:
                img = _load_image(sample.path)
            
            arr = img.astype(np.float64) / 255.0
            pixel_sum += arr.sum()
            pixel_sq_sum += (arr ** 2).sum()
            total_pixels += arr.size
        
        mean = pixel_sum / total_pixels
        std = np.sqrt(max(pixel_sq_sum / total_pixels - mean ** 2, 1e-8))
        self._stats = (float(mean), float(std))
    
    def _compute_class_weights(self) -> None:
        """Compute class-balanced weights."""
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for sample in self._train_samples:
            counts[sample.label] += 1
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (NUM_CLASSES * counts)
        weights = weights / weights.sum()
        self._class_weights = jnp.asarray(weights, dtype=jnp.float32)
    
    def _load_and_preprocess(
        self,
        sample: Sample,
        augment: bool,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        """Load, resize, and preprocess a single image."""
        # Load from cache or disk
        if self._cache:
            img = self._cache.get(sample.path)
            if img is None:
                img = _load_image(sample.path)
        else:
            img = _load_image(sample.path)
        
        # Resize to target size
        if img.shape[:2] != self.config.target_size:
            img = _resize_image(img, self.config.target_size)
        
        # Convert to float
        img = img.astype(np.float32) / 255.0
        
        # Apply augmentations
        if augment and rng is not None:
            img = apply_augmentations(img, rng, self.config.augmentation)
        
        # Normalize
        mean, std = self._stats
        img = (img - mean) / std
        
        # Add channel dimension
        return img[:, :, None]
    
    def _weighted_sample_indices(
        self,
        rng: np.random.Generator,
        num_samples: int,
    ) -> list[int]:
        """Sample indices with dataset weighting."""
        # Group indices by dataset
        dataset_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sample in enumerate(self._train_samples):
            dataset_indices[sample.dataset].append(idx)
        
        # Normalize weights
        total_weight = sum(
            self._dataset_weights.get(ds, 1.0)
            for ds in dataset_indices.keys()
        )
        
        # Sample from each dataset proportionally
        indices = []
        for dataset, ds_indices in dataset_indices.items():
            weight = self._dataset_weights.get(dataset, 1.0) / total_weight
            n_from_dataset = int(num_samples * weight)
            if n_from_dataset > 0:
                sampled = rng.choice(ds_indices, size=n_from_dataset, replace=True)
                indices.extend(sampled.tolist())
        
        rng.shuffle(indices)
        return indices[:num_samples]
    
    def train_batches(
        self,
        *,
        rng_seed: Optional[int] = None,
        use_mixup: bool = True,
        use_cutmix: bool = True,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Yield training batches with weighted sampling and augmentations."""
        rng = np.random.default_rng(rng_seed or self.config.seed)
        batch_size = self.config.batch_size
        
        # Weighted sampling
        num_batches = len(self._train_samples) // batch_size
        indices = self._weighted_sample_indices(rng, num_batches * batch_size)
        
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            if len(batch_indices) < batch_size and self.config.drop_last:
                continue
            
            # Load batch
            images = []
            labels = []
            for idx in batch_indices:
                sample = self._train_samples[idx]
                img = self._load_and_preprocess(sample, augment=True, rng=rng)
                images.append(img)
                labels.append(sample.label)
            
            images_arr = np.stack(images, axis=0)
            labels_arr = np.array(labels, dtype=np.int32)
            
            # Apply Mixup or CutMix (randomly choose one)
            aug_config = self.config.augmentation
            apply_mix = rng.random()
            
            if use_mixup and use_cutmix:
                if apply_mix < aug_config.cutmix_prob and aug_config.cutmix_alpha > 0:
                    images_arr, labels_arr = apply_cutmix(
                        images_arr, labels_arr, aug_config.cutmix_alpha, rng
                    )
                elif aug_config.mixup_alpha > 0:
                    images_arr, labels_arr = apply_mixup(
                        images_arr, labels_arr, aug_config.mixup_alpha, rng
                    )
            elif use_mixup and aug_config.mixup_alpha > 0:
                images_arr, labels_arr = apply_mixup(
                    images_arr, labels_arr, aug_config.mixup_alpha, rng
                )
            elif use_cutmix and aug_config.cutmix_alpha > 0:
                images_arr, labels_arr = apply_cutmix(
                    images_arr, labels_arr, aug_config.cutmix_alpha, rng
                )
            
            yield jnp.asarray(images_arr), jnp.asarray(labels_arr)
    
    def val_batches(
        self,
        *,
        batch_size: Optional[int] = None,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Yield validation batches (no augmentation)."""
        batch_size = batch_size or self.config.batch_size
        
        for batch_start in range(0, len(self._val_samples), batch_size):
            batch_samples = self._val_samples[batch_start:batch_start + batch_size]
            
            images = []
            labels = []
            for sample in batch_samples:
                img = self._load_and_preprocess(sample, augment=False, rng=None)
                images.append(img)
                labels.append(sample.label)
            
            yield (
                jnp.asarray(np.stack(images, axis=0)),
                jnp.asarray(np.array(labels, dtype=np.int32)),
            )
    
    def test_batches(
        self,
        *,
        batch_size: Optional[int] = None,
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Yield test batches (no augmentation)."""
        batch_size = batch_size or self.config.batch_size
        
        for batch_start in range(0, len(self._test_samples), batch_size):
            batch_samples = self._test_samples[batch_start:batch_start + batch_size]
            
            images = []
            labels = []
            for sample in batch_samples:
                img = self._load_and_preprocess(sample, augment=False, rng=None)
                images.append(img)
                labels.append(sample.label)
            
            yield (
                jnp.asarray(np.stack(images, axis=0)),
                jnp.asarray(np.array(labels, dtype=np.int32)),
            )
    
    def split_counts(self) -> dict[str, dict[str, int]]:
        """Return class distribution for each split."""
        def count_labels(samples: list[Sample]) -> dict[str, int]:
            counts = {name: 0 for name in CLASS_NAMES}
            for s in samples:
                counts[CLASS_NAMES[s.label]] += 1
            return counts
        
        return {
            "train": count_labels(self._train_samples),
            "val": count_labels(self._val_samples),
            "test": count_labels(self._test_samples),
        }
