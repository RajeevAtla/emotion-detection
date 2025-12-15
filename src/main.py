"""Command-line interface for training and evaluating emotion detection models."""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Optional, Tuple, TypeVar, Union, cast, List

import tomli_w
import tomllib
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.data import AugmentationConfig, DataModuleConfig, DatasetConfig
from src.train_multi_gpu import TrainingConfig, train_and_evaluate

ConfigValue = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "ConfigValue"],
    Sequence["ConfigValue"],
]
SummaryMetrics = Mapping[
    str, Union[int, float, None, str, Mapping[str, Sequence[float]]]
]
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emotion detection training entrypoint.")
    parser.add_argument("--config", type=Path, help="Path to TOML config file.")
    parser.add_argument("--output-dir", type=Path, help="Override for output directory.")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from.")
    parser.add_argument("--seed", type=int, help="Override for random seed.")
    parser.add_argument("--num-epochs", type=int, help="Override for number of epochs.")
    parser.add_argument("--experiment-name", type=str, help="Name appended to output dir.")
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, ConfigValue]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    suffix = path.suffix.lower()
    if suffix not in {".toml", ".tml"}:
        raise ValueError(f"Unsupported config format for {path}. Expected .toml")
    with path.open("rb") as fh:
        try:
            loaded = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Failed to parse TOML config at {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config file {path} must define a mapping at the top level.")
    training_section = loaded.get("training")
    if not isinstance(training_section, Mapping):
        raise ValueError("Config file must provide a top-level [training] table.")
    return cast(dict[str, ConfigValue], dict(training_section))


class RuntimeDatasetModel(BaseModel):
    name: str
    data_dir: str
    weight: float = Field(1.0, ge=0.0)
    enabled: bool = True


class RuntimeAugmentationModel(BaseModel):
    horizontal_flip_prob: float = Field(0.5, ge=0.0, le=1.0)
    rotation_degrees: float = Field(15.0, ge=0.0)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Optional[Tuple[float, float]] = None
    contrast_range: Optional[Tuple[float, float]] = None
    gaussian_blur_prob: float = Field(0.0, ge=0.0, le=1.0)
    gaussian_blur_sigma: float = Field(1.0, ge=0.0)
    mixup_alpha: float = Field(0.0, ge=0.0)
    cutmix_alpha: float = Field(0.0, ge=0.0)
    cutmix_prob: float = Field(0.5, ge=0.0, le=1.0)
    enabled: bool = True

    @field_validator("scale_range")
    @classmethod
    def validate_scale_range(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        if len(value) != 2:
            raise ValueError("scale_range must contain two values (min, max).")
        lo, hi = value
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("scale_range values must be positive.")
        if hi < lo:
            raise ValueError("scale_range max must be >= min.")
        return value


class RuntimeDataModel(BaseModel):
    data_dir: Path
    datasets: List[RuntimeDatasetModel] = []
    batch_size: Optional[int] = Field(None, gt=0)
    val_ratio: float = Field(0.1, ge=0.0, lt=1.0)
    seed: int = 0
    drop_last: bool = False
    mean: Optional[float] = None
    std: Optional[float] = None
    augment: bool = True
    augmentation: Optional[RuntimeAugmentationModel] = None
    stats_cache_path: Optional[Path] = None


class RuntimeTrainingModel(BaseModel):
    data: RuntimeDataModel
    output_dir: Optional[Path] = None
    model_depth: int = Field(34, ge=1)
    width_multiplier: int = Field(1, ge=1)
    dropout_rate: float = Field(0.0, ge=0.0, lt=1.0)
    num_epochs: int = Field(50, gt=0)
    batch_size: int = Field(128, gt=0)
    learning_rate: float = Field(3e-4, gt=0.0)
    min_learning_rate: float = Field(1e-5, ge=0.0)
    warmup_epochs: int = Field(5, ge=0)
    weight_decay: float = Field(1e-4, ge=0.0)
    gradient_accumulation_steps: int = Field(1, ge=1)
    label_smoothing: float = Field(0.0, ge=0.0, lt=1.0)
    seed: int = 0
    log_every: int = Field(100, ge=1)
    checkpoint_every: int = Field(5, ge=1)
    max_checkpoints: int = Field(3, ge=1)
    use_mixed_precision: bool = False
    patience: Optional[int] = Field(None, ge=1)
    freeze_stem: bool = False
    freeze_classifier: bool = False
    frozen_stages: Tuple[int, ...] = ()
    pretrained_checkpoint: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None
    experiment_name: Optional[str] = None

    @field_validator("frozen_stages")
    @classmethod
    def validate_stages(cls, value: Tuple[int, ...]) -> Tuple[int, ...]:
        if any(stage < 1 for stage in value):
            raise ValueError("frozen_stages must contain positive integers.")
        return value


def resolve_configs(args: argparse.Namespace) -> TrainingConfig:
    raw_config = load_config(args.config)

    payload: dict[str, object] = dict(raw_config)
    data_payload_raw = payload.get("data", {})
    if isinstance(data_payload_raw, Mapping):
        data_payload: dict[str, object] = dict(data_payload_raw)
    else:
        data_payload = {}
    payload["data"] = data_payload

    if args.output_dir is not None:
        payload["output_dir"] = args.output_dir
    if args.resume is not None:
        payload["resume_checkpoint"] = args.resume
    if args.num_epochs is not None:
        payload["num_epochs"] = args.num_epochs
    if args.seed is not None:
        payload["seed"] = args.seed
        data_payload["seed"] = args.seed
    if args.experiment_name is not None:
        payload["experiment_name"] = args.experiment_name

    data_payload.setdefault("data_dir", "data")

    config_model = RuntimeTrainingModel.model_validate(payload)

    output_root = args.output_dir or config_model.output_dir or Path("runs")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_suffix = args.experiment_name or config_model.experiment_name
    run_name = timestamp if experiment_suffix is None else f"{timestamp}-{experiment_suffix}"
    output_dir = (Path(output_root) / run_name).resolve()

    training_seed = args.seed if args.seed is not None else config_model.seed or config_model.data.seed

    dataset_configs = []
    for ds in config_model.data.datasets:
        dataset_configs.append(DatasetConfig(
            name=ds.name,
            data_dir=ds.data_dir,
            weight=ds.weight,
            enabled=ds.enabled,
        ))

    if not dataset_configs:
        dataset_configs = [DatasetConfig(name="fer2013", data_dir="fer2013", weight=1.0, enabled=True)]

    aug_config = None
    if config_model.data.augmentation is not None:
        aug = config_model.data.augmentation
        aug_config = AugmentationConfig(
            horizontal_flip_prob=aug.horizontal_flip_prob,
            rotation_degrees=aug.rotation_degrees,
            scale_range=aug.scale_range,
            brightness_range=aug.brightness_range,
            contrast_range=aug.contrast_range,
            gaussian_blur_prob=aug.gaussian_blur_prob,
            gaussian_blur_sigma=aug.gaussian_blur_sigma,
            mixup_alpha=aug.mixup_alpha,
            cutmix_alpha=aug.cutmix_alpha,
            cutmix_prob=aug.cutmix_prob,
            enabled=aug.enabled,
        )
    else:
        aug_config = AugmentationConfig()

    data_config = DataModuleConfig(
        data_dir=Path(config_model.data.data_dir),
        datasets=dataset_configs,
        batch_size=config_model.batch_size,
        val_ratio=config_model.data.val_ratio,
        seed=training_seed,
        drop_last=config_model.data.drop_last,
        mean=config_model.data.mean,
        std=config_model.data.std,
        augment=config_model.data.augment,
        augmentation=aug_config,
        stats_cache_path=config_model.data.stats_cache_path,
    )

    training_config = TrainingConfig(
        data=data_config,
        output_dir=output_dir,
        model_depth=config_model.model_depth,
        width_multiplier=config_model.width_multiplier,
        dropout_rate=config_model.dropout_rate,
        num_epochs=config_model.num_epochs,
        batch_size=config_model.batch_size,
        learning_rate=config_model.learning_rate,
        min_learning_rate=config_model.min_learning_rate,
        warmup_epochs=config_model.warmup_epochs,
        weight_decay=config_model.weight_decay,
        gradient_accumulation_steps=config_model.gradient_accumulation_steps,
        label_smoothing=config_model.label_smoothing,
        seed=training_seed,
        log_every=config_model.log_every,
        checkpoint_every=config_model.checkpoint_every,
        max_checkpoints=config_model.max_checkpoints,
        use_mixed_precision=config_model.use_mixed_precision,
        patience=config_model.patience,
        freeze_stem=config_model.freeze_stem,
        freeze_classifier=config_model.freeze_classifier,
        frozen_stages=tuple(config_model.frozen_stages),
        pretrained_checkpoint=config_model.pretrained_checkpoint,
        resume_checkpoint=config_model.resume_checkpoint,
    )

    return training_config


def prepare_environment(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jax.random.PRNGKey(seed)


def to_serializable(obj: object) -> ConfigValue:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer, jnp.generic)):
        return obj.item()
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        data = obj.tolist()
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return cast(ConfigValue, [to_serializable(v) for v in data])
        return to_serializable(data)
    if dataclasses.is_dataclass(obj):
        field_values = {
            field.name: getattr(obj, field.name)
            for field in dataclasses.fields(obj)
        }
        return cast(ConfigValue, {k: to_serializable(v) for k, v in field_values.items()})
    if isinstance(obj, Mapping):
        return cast(ConfigValue, {k: to_serializable(v) for k, v in obj.items()})
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return cast(ConfigValue, [to_serializable(v) for v in obj])
    return cast(ConfigValue, obj)


def prune_nones(obj: ConfigValue) -> ConfigValue:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        pruned: dict[str, ConfigValue] = {}
        for key, value in obj.items():
            new_value = prune_nones(cast(ConfigValue, value))
            if new_value is not None:
                pruned[key] = new_value
        return cast(ConfigValue, pruned)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        filtered: list[ConfigValue] = []
        for value in obj:
            new_value = prune_nones(cast(ConfigValue, value))
            if new_value is not None:
                filtered.append(new_value)
        return cast(ConfigValue, filtered)
    return obj


def persist_artifacts(
    output_dir: Path, config: TrainingConfig, metrics: Mapping[str, ConfigValue]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config_resolved.toml"
    metrics_path = output_dir / "metrics.toml"
    config_payload = prune_nones(cast(Mapping[str, ConfigValue], to_serializable(config)))
    metrics_payload = prune_nones(cast(Mapping[str, ConfigValue], to_serializable(dict(metrics))))
    config_path.write_text(tomli_w.dumps(config_payload), encoding="utf-8")
    metrics_path.write_text(tomli_w.dumps(metrics_payload), encoding="utf-8")


def summarize(metrics: SummaryMetrics) -> str:
    def _metric_value(value):
        if isinstance(value, Mapping) or value is None or isinstance(value, str):
            return float("nan")
        return float(value)

    def _maybe_format(label, value):
        if isinstance(value, Mapping) or value is None or isinstance(value, str):
            return None
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return f"{label}: {numeric:.4f}"

    summary_lines = [
        f"Final train loss:     {_metric_value(metrics.get('train_loss')):.4f}",
        f"Final train accuracy: {_metric_value(metrics.get('train_accuracy')):.4f}",
        f"Final val loss:       {_metric_value(metrics.get('val_loss')):.4f}",
        f"Final val accuracy:   {_metric_value(metrics.get('val_accuracy')):.4f}",
    ]
    val_f1_line = _maybe_format("Final val F1", metrics.get("val_f1"))
    if val_f1_line is not None:
        summary_lines.append(val_f1_line)
    val_macro_f1_line = _maybe_format("Final val macro F1", metrics.get("val_macro_f1"))
    if val_macro_f1_line is not None:
        summary_lines.append(val_macro_f1_line)
    test_acc = metrics.get("test_accuracy")
    if test_acc is not None and not isinstance(test_acc, Mapping):
        summary_lines.append(f"Test accuracy:         {float(test_acc):.4f}")
    test_f1_line = _maybe_format("Test F1", metrics.get("test_f1"))
    if test_f1_line is not None:
        summary_lines.append(test_f1_line)
    test_macro_f1_line = _maybe_format("Test macro F1", metrics.get("test_macro_f1"))
    if test_macro_f1_line is not None:
        summary_lines.append(test_macro_f1_line)
    best_epoch = metrics.get("best_epoch")
    if isinstance(best_epoch, (int, float)) and not math.isnan(float(best_epoch)):
        summary_lines.append(f"Best epoch:            {int(best_epoch)}")
    best_ckpt = metrics.get("best_checkpoint")
    if isinstance(best_ckpt, str) and best_ckpt:
        summary_lines.append(f"Best checkpoint:       {best_ckpt}")
    return "\n".join(summary_lines)


def main() -> None:
    args = parse_args()
    training_config = resolve_configs(args)
    prepare_environment(training_config.seed)
    metrics = train_and_evaluate(training_config)
    persist_artifacts(training_config.output_dir, training_config, metrics)
    print(summarize(metrics))
    print(f"Artifacts stored in {training_config.output_dir}")


if __name__ == "__main__":
    main()
