"""Tests for the command-line entrypoint and configuration helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
from pydantic import ValidationError

from src import main
from src.data import AugmentationConfig, DataModuleConfig
from src.train import TrainingConfig


def test_load_config_success_and_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"key": "value"}))
    assert main.load_config(config_path) == {"key": "value"}
    with pytest.raises(FileNotFoundError):
        main.load_config(tmp_path / "missing.json")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not-json")
    with pytest.raises(ValueError):
        main.load_config(bad_path)


def test_load_config_none_returns_empty_dict() -> None:
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
    with pytest.raises(ValidationError):
        main.RuntimeAugmentationModel(scale_range=scale_range)


def test_runtime_training_model_rejects_invalid_frozen_stage(tmp_path: Path) -> None:
    payload = {
        "data": {"data_dir": tmp_path},
        "frozen_stages": (0,),
    }
    with pytest.raises(ValidationError):
        main.RuntimeTrainingModel.model_validate(payload)


def test_build_dataclass_converts_paths(tmp_path: Path) -> None:
    payload = {
        "data_dir": str(tmp_path),
        "stats_cache_path": str(tmp_path / "stats.json"),
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
    main.prepare_environment(1234)
    values = [random.randint(0, 1000) for _ in range(3)]
    np_values = np.random.rand(3)
    main.prepare_environment(1234)
    assert values == [random.randint(0, 1000) for _ in range(3)]
    assert np.allclose(np_values, np.random.rand(3))


def test_to_serializable_handles_complex_types(tmp_path: Path) -> None:
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
    )
    serialized = main.to_serializable({"path": tmp_path, "config": config, "array": np.array([1, 2, 3])})
    assert serialized["path"] == str(tmp_path)
    assert serialized["array"] == [1, 2, 3]


def test_persist_artifacts_and_summarize(tmp_path: Path) -> None:
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path, batch_size=2),
        output_dir=tmp_path / "artifacts",
    )
    metrics: Dict[str, Any] = {"train_loss": 0.1, "train_accuracy": 0.9, "val_loss": 0.2, "val_accuracy": 0.8}
    main.persist_artifacts(config.output_dir, config, metrics)
    summary = main.summarize(metrics)
    assert "Final train loss" in summary
    assert (config.output_dir / "config_resolved.json").exists()
    assert (config.output_dir / "metrics.json").exists()


def test_resolve_configs_and_main_entry(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    config_payload = {
        "data": {"data_dir": str(dataset_dir)},
        "num_epochs": 2,
        "batch_size": 4,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload))
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

    def fake_train_and_evaluate(cfg: TrainingConfig) -> Dict[str, Any]:
        assert cfg.seed == 99
        return {"train_loss": 0.1, "train_accuracy": 1.0, "val_loss": 0.2, "val_accuracy": 0.9}

    argv = ["prog", "--config", str(config_path), "--output-dir", str(output_dir), "--seed", "99", "--experiment-name", "unit"]
    monkeypatch.setattr(main.argparse, "_sys", SimpleNamespace(argv=argv))
    monkeypatch.setattr(main, "train_and_evaluate", fake_train_and_evaluate)

    main.main()
    captured = capsys.readouterr()
    assert "Final train loss" in captured.out


def test_resolve_configs_applies_overrides_and_augmentation(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    config_payload = {
        "data": {
            "data_dir": str(dataset_dir),
            "augmentation": {"enabled": True, "horizontal_flip_prob": 0.1},
        },
        "num_epochs": 3,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload))

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


def test_to_serializable_numpy_and_jax_scalars(monkeypatch) -> None:
    class FakeJnpScalar:
        def __init__(self, value: float):
            self._value = value

        def item(self) -> float:
            return self._value

    fake_jnp = SimpleNamespace(generic=FakeJnpScalar, ndarray=tuple())
    monkeypatch.setattr(main, "jnp", fake_jnp)

    payload: Dict[str, Any] = {
        "numpy_float": np.float32(1.25),
        "numpy_int": np.int32(7),
        "jax_scalar": FakeJnpScalar(3.5),
        "array": np.array([1, 2, 3], dtype=np.int32),
        "path": Path("artifact"),
    }
    serialized = main.to_serializable(payload)
    assert serialized["numpy_float"] == pytest.approx(1.25)
    assert serialized["numpy_int"] == 7
    assert serialized["jax_scalar"] == pytest.approx(3.5)
    assert serialized["array"] == [1, 2, 3]
    assert serialized["path"] == "artifact"


def test_summarize_includes_test_accuracy() -> None:
    metrics = {
        "train_loss": 0.1,
        "train_accuracy": 0.9,
        "val_loss": 0.2,
        "val_accuracy": 0.8,
        "test_accuracy": 0.95,
    }
    summary = main.summarize(metrics)
    assert "Test accuracy" in summary


def test_main_entrypoint_executes(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"data": {"data_dir": str(tmp_path)}}))
    output_dir = tmp_path / "runs"

    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_path), "--output-dir", str(output_dir)])

    import src.train as train_module

    def fake_train_and_evaluate(cfg: TrainingConfig) -> Dict[str, Any]:
        return {"train_loss": 0.0, "train_accuracy": 1.0, "val_loss": 0.0, "val_accuracy": 1.0}

    monkeypatch.setattr(train_module, "train_and_evaluate", fake_train_and_evaluate)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    result = runpy.run_module("src.main", run_name="__main__")
    assert "train_loss" in result["metrics"]
