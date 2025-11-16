"""Tests for the command-line entrypoint and configuration helpers."""

from __future__ import annotations

import random
import textwrap
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import tomli_w
import tomllib
from pydantic import ValidationError
from PIL import Image

from src import main
from src.data import AugmentationConfig, DataModuleConfig
from src.train import TrainingConfig


def test_load_config_success_and_errors(tmp_path: Path) -> None:
    """Test config loading success, missing files, and parsing errors."""
    toml_path = tmp_path / "config.toml"
    toml_path.write_text("[training]\nkey = \"value\"\n")
    assert main.load_config(toml_path) == {"key": "value"}

    with pytest.raises(FileNotFoundError):
        main.load_config(tmp_path / "missing.toml")
    bad_path = tmp_path / "bad.toml"
    bad_path.write_text("not = toml")
    with pytest.raises(ValueError):
        main.load_config(bad_path)
    txt_path = tmp_path / "config.txt"
    txt_path.write_text("key=value")
    with pytest.raises(ValueError):
        main.load_config(txt_path)


def test_load_config_requires_mapping(monkeypatch, tmp_path: Path) -> None:
    """Ensure non-mapping TOML payloads raise parsing errors."""
    config_path = tmp_path / "bad.toml"
    config_path.write_text('key = "value"\n')

    def fake_load(_fh):
        return ["not-a-mapping"]

    monkeypatch.setattr(main.tomllib, "load", fake_load)
    with pytest.raises(ValueError):
        main.load_config(config_path)


def test_load_config_requires_training_table(tmp_path: Path) -> None:
    """Ensure configs without [training] raise errors."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('root_key = "value"\n')
    with pytest.raises(ValueError):
        main.load_config(config_path)


def test_load_config_none_returns_empty_dict() -> None:
    """Test that passing None returns an empty configuration."""
    assert main.load_config(None) == {}


@pytest.mark.parametrize(
    "scale_range",
    [
        (0.9,),
        (-0.1, 1.2),
        (1.2, 0.8),
    ],
)
def test_runtime_augmentation_model_validation_errors(scale_range) -> None:
    """Test that invalid augmentation scale ranges raise validation errors."""
    with pytest.raises(ValidationError):
        main.RuntimeAugmentationModel(scale_range=scale_range)


def test_validate_scale_range_direct_calls() -> None:
    """Test the scale range validator when invoked directly."""
    with pytest.raises(ValueError):
        main.RuntimeAugmentationModel.validate_scale_range((0.8,))
    assert main.RuntimeAugmentationModel.validate_scale_range((0.8, 1.2)) == (
        0.8,
        1.2,
    )


def test_runtime_training_model_rejects_invalid_frozen_stage(
    tmp_path: Path,
) -> None:
    """Test that zero-stage entries are rejected by the training schema."""
    payload = {
        "data": {"data_dir": tmp_path},
        "frozen_stages": (0,),
    }
    with pytest.raises(ValidationError):
        main.RuntimeTrainingModel.model_validate(payload)


def test_runtime_training_model_accepts_valid_frozen_stages(
    tmp_path: Path,
) -> None:
    """Test that positive frozen stages are accepted."""
    payload = {
        "data": {"data_dir": tmp_path},
        "frozen_stages": (1, 2),
    }
    model = main.RuntimeTrainingModel.model_validate(payload)
    assert model.frozen_stages == (1, 2)


def test_runtime_data_model_validation(tmp_path: Path) -> None:
    """Test that RuntimeDataModel coerces and validates numeric fields."""
    payload = {
        "data_dir": str(tmp_path),
        "batch_size": 16,
        "val_ratio": 0.2,
        "seed": 123,
    }
    model = main.RuntimeDataModel.model_validate(payload)
    assert model.batch_size == 16
    assert model.val_ratio == 0.2
    assert model.data_dir == tmp_path


@pytest.mark.parametrize("bad_ratio", [-0.1, 1.2])
def test_runtime_data_model_invalid_val_ratio(bad_ratio) -> None:
    """Test invalid validation ratio values raise errors."""
    payload = {
        "data_dir": Path("."),
        "batch_size": 8,
        "val_ratio": bad_ratio,
    }
    with pytest.raises(ValidationError):
        main.RuntimeDataModel.model_validate(payload)


def test_build_dataclass_converts_paths(tmp_path: Path) -> None:
    """Test conversion to dataclasses including nested path fields."""
    payload = {
        "data_dir": str(tmp_path),
        "stats_cache_path": str(tmp_path / "stats.toml"),
        "augment": False,
        "augmentation": {
            "horizontal_flip_prob": 0.3,
            "rotation_degrees": 5.0,
            "scale_range": [0.9, 1.1],
        },
        "ignored_field": "value",
    }
    config = main._build_dataclass(DataModuleConfig, payload)
    assert isinstance(config.data_dir, Path)
    assert config.augment is False
    assert isinstance(config.augmentation, AugmentationConfig)


def test_prepare_environment_sets_seeds() -> None:
    """Test that environment preparation resets PRNG sequences."""
    main.prepare_environment(1234)
    values = [random.randint(0, 1000) for _ in range(3)]
    np_values = np.random.rand(3)
    main.prepare_environment(1234)
    assert values == [random.randint(0, 1000) for _ in range(3)]
    assert np.allclose(np_values, np.random.rand(3))


def test_to_serializable_handles_complex_types(tmp_path: Path) -> None:
    """Test conversion of config, paths, and arrays into serializable types."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
    )
    serialized = main.to_serializable(
        {"path": tmp_path, "config": config, "array": np.array([1, 2, 3])}
    )
    assert serialized["path"] == str(tmp_path)
    assert serialized["array"] == [1, 2, 3]


def test_persist_artifacts_and_summarize(tmp_path: Path) -> None:
    """Test artifact persistence and summary output creation."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path, batch_size=2),
        output_dir=tmp_path / "artifacts",
    )
    metrics: dict[str, float] = {
        "train_loss": 0.1,
        "train_accuracy": 0.9,
        "val_loss": 0.2,
        "val_accuracy": 0.8,
    }
    main.persist_artifacts(config.output_dir, config, metrics)
    summary = main.summarize(metrics)
    assert "Final train loss" in summary
    assert (config.output_dir / "config_resolved.toml").exists()
    assert (config.output_dir / "metrics.toml").exists()


def test_resolve_configs_and_main_entry(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI resolution and main execution with monkeypatched dependencies."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            [training]
            num_epochs = 2
            batch_size = 4

            [training.data]
            data_dir = "{dataset_dir.as_posix()}"
            """
        ).strip()
        + "\n"
    )
    output_dir = tmp_path / "runs"

    args = SimpleNamespace(
        config=config_path,
        output_dir=output_dir,
        resume=None,
        seed=99,
        num_epochs=1,
        experiment_name="unit",
    )
    training_config = main.resolve_configs(args)
    assert training_config.data.seed == 99
    assert training_config.output_dir.parent == output_dir

    def fake_train_and_evaluate(cfg: TrainingConfig) -> dict[str, float]:
        assert cfg.seed == 99
        return {
            "train_loss": 0.1,
            "train_accuracy": 1.0,
            "val_loss": 0.2,
            "val_accuracy": 0.9,
        }

    argv = [
        "prog",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        "99",
        "--experiment-name",
        "unit",
    ]
    monkeypatch.setattr(main.argparse, "_sys", SimpleNamespace(argv=argv))
    monkeypatch.setattr(main, "train_and_evaluate", fake_train_and_evaluate)

    main.main()
    captured = capsys.readouterr()
    assert "Final train loss" in captured.out


def test_resolve_configs_applies_overrides_and_augmentation(
    tmp_path: Path,
) -> None:
    """Test CLI overrides for resume, epochs, and augmentation payloads."""
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            [training]
            num_epochs = 3

            [training.data]
            data_dir = "{dataset_dir.as_posix()}"

            [training.data.augmentation]
            enabled = true
            horizontal_flip_prob = 0.1
            """
        ).strip()
        + "\n"
    )

    args = SimpleNamespace(
        config=config_path,
        output_dir=tmp_path / "runs",
        resume=tmp_path / "resume",
        seed=11,
        num_epochs=5,
        experiment_name=None,
    )
    config = main.resolve_configs(args)
    assert config.resume_checkpoint == args.resume
    assert config.num_epochs == 5
    assert isinstance(config.data.augmentation, AugmentationConfig)
    assert config.data.seed == 11
    assert config.data.batch_size == config.batch_size


def test_to_serializable_numpy_and_jax_scalars() -> None:
    """Test serialization for numpy and jax scalar types."""
    payload: dict[str, object] = {
        "numpy_float": np.float32(1.25),
        "numpy_int": np.int32(7),
        "jax_scalar": jnp.float32(3.5),
        "array": np.array([1, 2, 3], dtype=np.int32),
        "path": Path("artifact"),
    }
    serialized = main.to_serializable(payload)
    assert serialized["numpy_float"] == pytest.approx(1.25)
    assert serialized["numpy_int"] == 7
    assert serialized["jax_scalar"] == pytest.approx(3.5)
    assert serialized["array"] == [1, 2, 3]
    assert serialized["path"] == "artifact"


def test_to_serializable_handles_jax_generic(monkeypatch) -> None:
    """Test serialization when numpy scalar fallbacks are monkeypatched."""
    monkeypatch.setattr(main.np, "floating", (), raising=False)
    monkeypatch.setattr(main.np, "integer", (), raising=False)
    value = np.float32(2.5)
    assert main.to_serializable(value) == pytest.approx(2.5)


def test_summarize_includes_test_accuracy() -> None:
    """Test that optional test accuracy is included in summaries."""
    metrics = {
        "train_loss": 0.1,
        "train_accuracy": 0.9,
        "val_loss": 0.2,
        "val_accuracy": 0.8,
        "test_accuracy": 0.95,
    }
    summary = main.summarize(metrics)
    assert "Test accuracy" in summary


def test_summarize_includes_f1_and_best_checkpoint() -> None:
    """Test summarize prints F1 metrics and checkpoint information."""
    metrics = {
        "train_loss": 0.1,
        "train_accuracy": 0.9,
        "val_loss": 0.2,
        "val_accuracy": 0.8,
        "val_f1": 0.85,
        "val_macro_f1": 0.83,
        "test_accuracy": 0.9,
        "test_f1": 0.88,
        "test_macro_f1": 0.86,
        "best_epoch": 3,
        "best_checkpoint": "runs/checkpoints/epoch_0003",
    }
    summary = main.summarize(metrics)
    assert "Final val F1" in summary
    assert "Final val macro F1" in summary
    assert "Test F1" in summary
    assert "Best epoch" in summary
    assert "Best checkpoint" in summary


def test_summarize_ignores_nan_metrics() -> None:
    """Test summarize skips F1 entries when metrics are NaN."""
    nan_val = float("nan")
    metrics = {
        "train_loss": 0.1,
        "train_accuracy": 0.9,
        "val_loss": 0.2,
        "val_accuracy": 0.8,
        "val_f1": nan_val,
        "val_macro_f1": nan_val,
        "test_accuracy": 0.9,
        "test_f1": nan_val,
        "test_macro_f1": nan_val,
    }
    summary = main.summarize(metrics)
    assert "Final val F1" not in summary
    assert "Final val macro F1" not in summary
    assert "Test F1" not in summary
    assert "Test macro F1" not in summary


def test_example_config_round_trip(tmp_path: Path) -> None:
    """Ensure the example config file stays aligned with the schema."""
    example_config = Path("configs/example.toml")
    payload = tomllib.loads(example_config.read_text())
    training_payload = payload["training"]
    training_payload["data"]["data_dir"] = tmp_path.as_posix()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        tomli_w.dumps({"training": training_payload}), encoding="utf-8"
    )
    args = SimpleNamespace(
        config=config_path,
        output_dir=None,
        resume=None,
        seed=None,
        num_epochs=None,
        experiment_name=None,
    )
    training_config = main.resolve_configs(args)
    assert training_config.data.data_dir == tmp_path
    assert training_config.num_epochs == training_payload["num_epochs"]


def test_main_entrypoint_executes(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the CLI entrypoint executes end-to-end with faked training."""
    config_path = tmp_path / "cfg.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            [training]
            num_epochs = 1

            [training.data]
            data_dir = "{tmp_path.as_posix()}"
            """
        ).strip()
        + "\n"
    )
    output_dir = tmp_path / "runs"

    def write_sample(split: str, class_name: str, value: int) -> None:
        class_dir = tmp_path / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        arr = np.full((48, 48), value, dtype=np.uint8)
        Image.fromarray(arr).save(class_dir / "img.png")

    write_sample("train", "angry", 64)
    write_sample("test", "angry", 128)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    import src.train as train_module

    calls: dict[str, TrainingConfig] = {}

    def fake_train_and_evaluate(cfg: TrainingConfig) -> dict[str, float]:
        calls["config"] = cfg
        return {
            "train_loss": 0.0,
            "train_accuracy": 1.0,
            "val_loss": 0.0,
            "val_accuracy": 1.0,
        }

    monkeypatch.setattr(
        train_module, "train_and_evaluate", fake_train_and_evaluate
    )
    monkeypatch.setattr(main, "train_and_evaluate", fake_train_and_evaluate)

    main.main()
    captured = capsys.readouterr()
    assert "Final train loss" in captured.out or captured.out == ""
    assert "config" in calls


def test_resolve_configs_handles_non_mapping_data(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[training]\ndata = []\n")
    output_dir = tmp_path / "runs"

    args = SimpleNamespace(
        config=config_path,
        output_dir=output_dir,
        resume=None,
        seed=None,
        num_epochs=None,
        experiment_name=None,
    )

    training_config = main.resolve_configs(args)
    assert training_config.data.data_dir.name == "data"
    assert training_config.output_dir.parent == output_dir


def test_summarize_with_missing_metrics() -> None:
    summary = main.summarize(
        {
            "train_loss": None,
            "train_accuracy": 0.5,
            "val_loss": 0.3,
            "val_accuracy": None,
        }
    )
    assert "Final train loss" in summary
