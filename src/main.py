"""Command-line interface for training and evaluating emotion detection models."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Optional, Tuple, TypeVar, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.data import AugmentationConfig, DataModuleConfig
from src.train import TrainingConfig, train_and_evaluate

JSONValue = Union[
    str,
    int,
    float,
    bool,
    None,
    Mapping[str, "JSONValue"],
    Sequence["JSONValue"],
]
SummaryMetrics = Mapping[
    str, Union[float, None, Mapping[str, Sequence[float]]]
]
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training entrypoint.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Emotion detection training entrypoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file describing data/model/training settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional override for run output directory (defaults to config or ./runs).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override for random seed applied to data and training config.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Override for number of training epochs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Optional name appended to output directory for easier identification.",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, JSONValue]:
    """Load a JSON configuration file if provided.

    Args:
        path: Path to the JSON file or ``None``.

    Returns:
        dict[str, JSONValue]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON cannot be parsed.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            loaded = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON config at {path}: {exc}"
            ) from exc
    return cast(dict[str, JSONValue], loaded)


class RuntimeAugmentationModel(BaseModel):
    """Schema describing augmentation-related configuration."""

    horizontal_flip_prob: float = Field(0.5, ge=0.0, le=1.0)
    rotation_degrees: float = Field(15.0, ge=0.0)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    elastic_blur_sigma: Optional[float] = Field(None, ge=0.0)
    enabled: bool = True

    @field_validator("scale_range")
    @classmethod
    def validate_scale_range(
        cls, value: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Validate that the scale range is positive and ordered.

        Args:
            value: Tuple containing the minimum and maximum scale factors.

        Returns:
            Tuple[float, float]: Sanitized scale range.

        Raises:
            ValueError: If the tuple is not length two, non-positive, or inverted.
        """
        if len(value) != 2:
            raise ValueError("scale_range must contain two values (min, max).")
        lo, hi = value
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("scale_range values must be positive.")
        if hi < lo:
            raise ValueError("scale_range max must be >= min.")
        return value


class RuntimeDataModel(BaseModel):
    """Schema describing dataset configuration."""

    data_dir: Path
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
    """Schema describing top-level training configuration."""

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
        """Ensure that frozen stage indices are positive.

        Args:
            value: Tuple of stage indices requested for freezing.

        Returns:
            Tuple[int, ...]: The validated stage tuple.

        Raises:
            ValueError: If any value is less than one.
        """
        if any(stage < 1 for stage in value):
            raise ValueError("frozen_stages must contain positive integers.")
        return value


def _build_dataclass(cls: type[T], raw: Mapping[str, JSONValue]) -> T:
    """Build a dataclass instance from a raw mapping.

    Args:
        cls: Dataclass type to instantiate.
        raw: Mapping of field names to values.

    Returns:
        T: Instantiated dataclass of the requested type.
    """
    field_names = {field.name for field in dataclasses.fields(cls)}
    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key not in field_names:
            continue
        if key.endswith("dir") or key.endswith("path"):
            kwargs[key] = Path(value) if value is not None else None
        elif key == "augmentation" and isinstance(value, Mapping):
            kwargs[key] = AugmentationConfig(**value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def resolve_configs(args: argparse.Namespace) -> TrainingConfig:
    """Resolve CLI arguments and configuration into ``TrainingConfig``."""
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
    run_name = (
        timestamp
        if experiment_suffix is None
        else f"{timestamp}-{experiment_suffix}"
    )
    output_dir = (Path(output_root) / run_name).resolve()

    training_seed = (
        args.seed
        if args.seed is not None
        else config_model.seed or config_model.data.seed
    )

    data_dict = cast(dict[str, JSONValue], config_model.data.model_dump())
    data_dict["seed"] = training_seed
    data_dict["batch_size"] = config_model.batch_size
    if data_dict.get("augmentation") is not None:
        data_dict["augmentation"] = data_dict["augmentation"]
    data_config = _build_dataclass(DataModuleConfig, data_dict)

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

    data_config.batch_size = training_config.batch_size
    return training_config


def prepare_environment(seed: int) -> None:
    """Seed Python, NumPy, and JAX random number generators.

    Args:
        seed: Seed value used for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    jax.random.PRNGKey(seed)


def to_serializable(obj: object) -> JSONValue:
    """Convert complex objects into JSON-serializable structures."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer, jnp.generic)):
        return obj.item()
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        data = obj.tolist()
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return cast(JSONValue, [to_serializable(v) for v in data])
        return to_serializable(data)
    if dataclasses.is_dataclass(obj):
        field_values = {
            field.name: getattr(obj, field.name)
            for field in dataclasses.fields(obj)
        }
        return cast(
            JSONValue, {k: to_serializable(v) for k, v in field_values.items()}
        )
    if isinstance(obj, Mapping):
        return cast(JSONValue, {k: to_serializable(v) for k, v in obj.items()})
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return cast(JSONValue, [to_serializable(v) for v in obj])
    return cast(JSONValue, obj)


def persist_artifacts(
    output_dir: Path, config: TrainingConfig, metrics: Mapping[str, JSONValue]
) -> None:
    """Write resolved configuration and metrics to disk.

    Args:
        output_dir: Directory where artifacts are stored.
        config: Resolved training configuration to persist.
        metrics: Dictionary of run metrics to write.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(to_serializable(config), fh, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(to_serializable(metrics), fh, indent=2)


def summarize(metrics: SummaryMetrics) -> str:
    """Create a human-readable summary of key metrics."""

    def _metric_value(
        value: Union[float, None, Mapping[str, Sequence[float]]],
    ) -> float:
        if isinstance(value, Mapping) or value is None:
            return float("nan")
        return float(value)

    summary_lines = [
        f"Final train loss:     {_metric_value(metrics.get('train_loss')):.4f}",
        f"Final train accuracy: {_metric_value(metrics.get('train_accuracy')):.4f}",
        f"Final val loss:       {_metric_value(metrics.get('val_loss')):.4f}",
        f"Final val accuracy:   {_metric_value(metrics.get('val_accuracy')):.4f}",
    ]
    test_acc = metrics.get("test_accuracy")
    if test_acc is not None and not isinstance(test_acc, Mapping):
        summary_lines.append(f"Test accuracy:         {float(test_acc):.4f}")
    return "\n".join(summary_lines)


def main() -> None:
    """CLI entrypoint for training and evaluation."""
    args = parse_args()
    training_config = resolve_configs(args)
    prepare_environment(training_config.seed)
    metrics = train_and_evaluate(training_config)
    persist_artifacts(training_config.output_dir, training_config, metrics)
    print(summarize(metrics))
    print(f"Artifacts stored in {training_config.output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
