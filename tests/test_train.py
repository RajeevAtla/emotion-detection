"""Tests for training utilities and orchestration."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, cast

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.core import freeze
from flax.training.dynamic_scale import DynamicScale

from src import train
from src.data import DataModuleConfig
from src.model import ResNet
from src.train import (
    TrainingConfig,
    TrainState,
    build_eval_step,
    build_train_step,
    compute_confusion_matrix,
    create_learning_rate_schedule,
    create_optimizer,
    create_train_state,
    cross_entropy_loss,
    format_confusion_matrix,
    maybe_restore_checkpoint,
    predict_batches,
    save_checkpoint,
    train_and_evaluate,
)


def test_learning_rate_schedule_shape() -> None:
    """Test learning rate schedule warmup and decay behavior."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("runs"),
        num_epochs=2,
        warmup_epochs=1,
        learning_rate=0.01,
        min_learning_rate=0.001,
    )
    schedule = create_learning_rate_schedule(config, steps_per_epoch=4)
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(2)) == pytest.approx(0.005, rel=1e-5)
    assert float(schedule(4)) == pytest.approx(0.01, rel=1e-5)
    assert float(schedule(10)) == pytest.approx(0.001, rel=1e-2)


def test_create_optimizer_with_mask_and_accum() -> None:
    """Test optimizer creation with gradient accumulation and masks."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("out"),
        gradient_accumulation_steps=2,
    )
    def schedule(_: int) -> float:
        """Return a constant learning rate for optimizer tests."""
        return 0.1
    params = freeze({"w": jnp.array(1.0)})
    mask = freeze({"w": True})
    tx = create_optimizer(config, schedule, mask)
    opt_state = tx.init(params)
    assert hasattr(opt_state, "inner_state")


def test_create_train_state_with_pretrained(monkeypatch, tmp_path: Path) -> None:
    """Test train state initialization when loading pretrained checkpoints."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
        freeze_stem=True,
        freeze_classifier=True,
        frozen_stages=(1,),
        pretrained_checkpoint=tmp_path / "ckpt",
    )
    monkeypatch.setattr(train, "maybe_load_pretrained_params", lambda params, config: params)
    monkeypatch.setattr(
        train,
        "create_optimizer",
        lambda *args, **kwargs: SimpleNamespace(init=lambda params: "opt_state"),
    )
    resnet = train.create_resnet(depth=18, num_classes=2)
    schedule = create_learning_rate_schedule(config, steps_per_epoch=1)
    state = create_train_state(jax.random.PRNGKey(0), resnet, config, schedule)
    assert isinstance(state, TrainState)
    assert state.dynamic_scale is None


def test_create_train_state_without_freeze(tmp_path: Path) -> None:
    """Test train state initialization without freezing any layers."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
    )
    resnet = train.create_resnet(depth=18, num_classes=2)
    schedule = create_learning_rate_schedule(config, steps_per_epoch=1)
    state = create_train_state(jax.random.PRNGKey(1), resnet, config, schedule)
    assert isinstance(state, TrainState)


def _dummy_apply_fn(variables: Dict[str, Any], images: jnp.ndarray, *, train: bool, mutable=None, rngs=None):
    """Produce deterministic logits for dummy training utilities."""
    logits = jnp.stack([jnp.sum(images, axis=(1, 2, 3)), jnp.zeros(images.shape[0])], axis=-1)
    batch_stats = variables.get("batch_stats", freeze({}))
    if mutable:
        return logits, {"batch_stats": batch_stats}
    return logits


def _make_train_state(dynamic: bool = False) -> TrainState:
    """Construct a lightweight train state for unit tests."""
    params = {"w": jnp.array(1.0)}
    tx = optax.sgd(learning_rate=0.1)
    state = TrainState.create(apply_fn=_dummy_apply_fn, params=params, tx=tx, batch_stats=freeze({}))
    if dynamic:
        state = state.replace(dynamic_scale=DynamicScale())
    return state


class DummyModel:
    """Lightweight ResNet stand-in for train/eval step tests."""
    config = train.create_resnet(depth=18, num_classes=2).config

    @staticmethod
    def apply(variables: Dict[str, Any], images: jnp.ndarray, *, train: bool, mutable=None, rngs=None):
        """Apply the dummy model by delegating to the helper apply function."""
        return _dummy_apply_fn(variables, images, train=train, mutable=mutable, rngs=rngs)


def test_cross_entropy_loss_with_smoothing_and_weights() -> None:
    """Test label smoothing and per-class weighting inside the loss."""
    logits = jnp.array([[2.0, 0.5]])
    labels = jnp.array([0])
    weights = jnp.array([0.7, 0.3])
    loss = cross_entropy_loss(logits, labels, label_smoothing=0.1, class_weights=weights)
    assert float(loss) > 0


def test_build_train_step_standard() -> None:
    """Test the standard training step with deterministic gradients."""
    config = TrainingConfig(data=DataModuleConfig(data_dir=Path(".")), output_dir=Path("out"))
    model = cast(ResNet, DummyModel())
    train_step = build_train_step(model, config, class_weights=jnp.ones(2))
    state = _make_train_state()
    images = jnp.ones((2, 2, 2, 1))
    labels = jnp.zeros((2,), dtype=jnp.int32)
    new_state, metrics = train_step(state, (images, labels), jax.random.PRNGKey(0))
    assert metrics["loss"] >= 0
    assert isinstance(new_state, TrainState)


def test_build_train_step_mixed_precision() -> None:
    """Test the training step when mixed precision is enabled."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("out"),
        use_mixed_precision=True,
    )
    model = cast(ResNet, DummyModel())
    train_step = build_train_step(model, config, class_weights=None)
    class DummyDynamicScale:
        loss_scale = jnp.array(1.0)

        def scale(self, loss):
            return loss

        def value_and_grad(self, fn, has_aux):
            def wrapped(params):
                loss, aux = fn(params)
                grads = jax.tree_util.tree_map(jnp.zeros_like, params)
                return (loss, aux), grads

            return wrapped

    state = _make_train_state()
    state = state.replace(dynamic_scale=DummyDynamicScale())
    images = jnp.ones((1, 2, 2, 1))
    labels = jnp.zeros((1,), dtype=jnp.int32)
    jax.config.update("jax_disable_jit", True)
    try:
        train_step(state, (images, labels), jax.random.PRNGKey(1))
    finally:
        jax.config.update("jax_disable_jit", False)


def test_build_eval_step_returns_predictions() -> None:
    """Test evaluation step outputs accuracy metrics and predictions."""
    config = TrainingConfig(data=DataModuleConfig(data_dir=Path(".")), output_dir=Path("out"))
    eval_step = build_eval_step(cast(ResNet, DummyModel()), config)
    state = _make_train_state()
    metrics, preds = eval_step(state, (jnp.ones((1, 2, 2, 1)), jnp.zeros((1,), dtype=jnp.int32)))
    assert "accuracy" in metrics
    assert preds.shape == (1,)


def test_confusion_matrix_helpers() -> None:
    """Test confusion matrix computation and formatting helpers."""
    preds = jnp.array([0, 1, 1])
    labels = jnp.array([0, 0, 1])
    cm = compute_confusion_matrix(preds, labels, num_classes=2)
    assert cm.tolist() == [[1, 1], [0, 1]]
    text = format_confusion_matrix(cm, ["neg", "pos"])
    assert "neg" in text and "pos" in text


def test_checkpoint_save_and_restore(tmp_path: Path) -> None:
    """Test checkpoint rotation and restoration behavior."""
    config = TrainingConfig(data=DataModuleConfig(data_dir=tmp_path), output_dir=tmp_path, max_checkpoints=2)
    state = _make_train_state()
    save_checkpoint(state, config, epoch=1)
    save_checkpoint(state, config, epoch=2)
    save_checkpoint(state, config, epoch=3)
    assert not (config.output_dir / "checkpoints" / "epoch_0001").exists()
    restored = maybe_restore_checkpoint(replace(config, resume_checkpoint=config.output_dir / "checkpoints" / "epoch_0003"))
    assert restored is not None


def test_predict_batches_handles_empty_and_non_empty() -> None:
    """Test prediction helper behavior with empty and populated inputs."""
    config = TrainingConfig(data=DataModuleConfig(data_dir=Path(".")), output_dir=Path("out"))
    state = _make_train_state()
    preds, labels = predict_batches(state, cast(ResNet, DummyModel()), batches=[], config=config)
    assert preds.size == 0

    batches = [
        (jnp.ones((1, 2, 2, 1)), jnp.zeros((1,), dtype=jnp.int32)),
        (jnp.ones((1, 2, 2, 1)) * 2, jnp.ones((1,), dtype=jnp.int32)),
    ]
    preds, labels = predict_batches(state, cast(ResNet, DummyModel()), batches=batches, config=config)
    assert preds.shape == labels.shape


def test_train_and_evaluate_with_stubs(monkeypatch, tmp_path: Path) -> None:
    """Test train-and-evaluate loop using stubbed collaborators."""
    class StubDataModule:
        """Stub implementation of the data module contract."""

        def __init__(self, config):
            """Initialize the stub with config and synthetic class weights."""
            self.config = config
            self.class_weights = jnp.ones(2)

        def setup(self):
            """No-op setup to satisfy interface."""
            pass

        def split_counts(self):
            """Return a deterministic set of train/validation counts."""
            return {"train": {"neg": 2, "pos": 2}, "val": {"neg": 1, "pos": 1}}

        def train_batches(self, **kwargs):
            """Yield two small synthetic batches for training."""
            def generator():
                yield jnp.ones((1, 2, 2, 1)), jnp.zeros((1,), dtype=jnp.int32)
                yield jnp.ones((1, 2, 2, 1)) * 2, jnp.ones((1,), dtype=jnp.int32)
            return generator()

        def val_batches(self, **kwargs):
            """Yield a single validation batch."""
            return [(jnp.ones((1, 2, 2, 1)), jnp.zeros((1,), dtype=jnp.int32))]

        def test_batches(self, **kwargs):
            """Yield a single test batch."""
            return [(jnp.ones((1, 2, 2, 1)), jnp.zeros((1,), dtype=jnp.int32))]

    class StubWriter:
        """Stubbed TensorBoard writer that records the log directory."""

        def __init__(self, log_dir):
            """Store the target log directory."""
            self.log_dir = log_dir

        def add_scalars(self, *args, **kwargs):
            """Ignore scalar writes."""
            pass

        def add_text(self, *args, **kwargs):
            """Ignore text writes."""
            pass

        def close(self):
            """Ignore close requests."""
            pass

    class StubState:
        """Lightweight training state tracking gradient applications."""

        def __init__(self):
            """Initialize the state with an application counter."""
            self.apply_gradients_calls = 0

        def apply_gradients(self, *, grads, batch_stats):
            """Increment the application counter and return self."""
            self.apply_gradients_calls += 1
            return self

    def stub_create_state(rng, model_obj, config_obj, schedule):
        """Return a freshly constructed stub state."""
        return StubState()

    def stub_train_step(state, batch, rng):
        """Return synthetic metrics for the stubbed train step."""
        return state, {"loss": jnp.array(0.1), "accuracy": jnp.array(1.0)}

    val_losses = [jnp.array(0.2), jnp.array(0.2)]

    def stub_eval_step(state, batch):
        """Return deterministic validation metrics and zero predictions."""
        return {"loss": val_losses.pop(0), "accuracy": jnp.array(0.9)}, jnp.zeros(batch[0].shape[0], dtype=jnp.int32)

    def stub_predict_batches(state, model_obj, batches, config_obj):
        """Return zero-valued predictions and labels."""
        return jnp.zeros(2, dtype=jnp.int32), jnp.zeros(2, dtype=jnp.int32)

    monkeypatch.setattr(train, "EmotionDataModule", StubDataModule)
    monkeypatch.setattr(train, "SummaryWriter", StubWriter)
    class FakeResNet:
        """Stub ResNet carrying the required config interface."""

        config = SimpleNamespace(num_classes=2)

        def replace(self, **kwargs):
            """Return self because state mutation is not required."""
            return self

    monkeypatch.setattr(train, "create_resnet", lambda **kwargs: FakeResNet())
    monkeypatch.setattr(train, "create_train_state", stub_create_state)
    monkeypatch.setattr(train, "build_train_step", lambda *args, **kwargs: stub_train_step)
    monkeypatch.setattr(train, "build_eval_step", lambda *args, **kwargs: stub_eval_step)
    monkeypatch.setattr(train, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "predict_batches", stub_predict_batches)

    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path, batch_size=2),
        output_dir=tmp_path / "run",
        num_epochs=2,
        checkpoint_every=1,
        patience=1,
        log_every=1,
    )

    metrics = train_and_evaluate(config)
    assert "history" in metrics


def test_build_eval_step_mixed_precision_casts() -> None:
    """Test mixed precision evaluation casting behavior."""
    record: Dict[str, jnp.dtype] = {}

    class MixedModel:
        """Stub model capturing the dtype of evaluation inputs."""

        config = train.create_resnet(depth=18, num_classes=2).config

        @staticmethod
        def apply(variables: Dict[str, Any], images: jnp.ndarray, *, train: bool, mutable=None, rngs=None):
            """Record the dtype of incoming images and emit zero logits."""
            record["dtype"] = images.dtype
            logits = jnp.zeros((images.shape[0], 2), dtype=jnp.float32)
            return logits

    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("out"),
        use_mixed_precision=True,
    )
    eval_step = build_eval_step(cast(ResNet, MixedModel()), config)
    state = _make_train_state()
    eval_step(state, (jnp.ones((1, 2, 2, 1), dtype=jnp.float32), jnp.zeros((1,), dtype=jnp.int32)))
    assert record["dtype"] == jnp.float16


def test_predict_batches_mixed_precision_casts() -> None:
    """Test that mixed precision predictions cast inputs appropriately."""
    dtypes: list[jnp.dtype] = []

    class MixedModel:
        """Stub model recording dtypes observed during prediction."""

        config = train.create_resnet(depth=18, num_classes=2).config

        @staticmethod
        def apply(variables: Dict[str, Any], images: jnp.ndarray, *, train: bool, mutable=None, rngs=None):
            """Append the incoming dtype for verification and return logits."""
            dtypes.append(images.dtype)
            return jnp.zeros((images.shape[0], 2), dtype=jnp.float32)

    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("out"),
        use_mixed_precision=True,
    )
    state = _make_train_state()
    batches = [
        (jnp.ones((2, 2, 2, 1), dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.int32)),
        (jnp.ones((1, 2, 2, 1), dtype=jnp.float32), jnp.ones((1,), dtype=jnp.int32)),
    ]
    preds, labels = predict_batches(state, cast(ResNet, MixedModel()), batches=batches, config=config)
    assert preds.shape == labels.shape
    assert all(dtype == jnp.float16 for dtype in dtypes)


def test_train_and_evaluate_handles_restore_and_empty_metrics(monkeypatch, tmp_path: Path) -> None:
    """Test resume flow and metric defaults when no batches are produced."""
    class EmptyDataModule:
        """Stub data module that produces no batches."""

        def __init__(self, config):
            """Store config and synthetic class weights."""
            self.config = config
            self.class_weights = jnp.ones(2)

        def setup(self):
            """No-op setup method."""
            pass

        def split_counts(self):
            """Return minimal train and validation counts."""
            return {"train": {"neg": 1, "pos": 1}, "val": {"neg": 0, "pos": 0}}

        def train_batches(self, **kwargs):
            """Return an empty iterator for training batches."""
            return iter(())

        def val_batches(self, **kwargs):
            """Return an empty validation collection."""
            return []

        def test_batches(self, **kwargs):
            """Return an empty test collection."""
            return []

    class DummyWriter:
        """Stub writer capturing the log directory."""

        def __init__(self, log_dir):
            """Store the designated log directory."""
            self.log_dir = log_dir

        def add_scalars(self, *args, **kwargs):
            """Ignore scalar log calls."""
            pass

        def add_text(self, *args, **kwargs):
            """Ignore text log calls."""
            pass

        def close(self):
            """Ignore close operations."""
            pass

    def stub_create_state(rng, model_obj, config_obj, schedule):
        """Return a simple train state for resume testing."""
        return _make_train_state()

    class FakeResNet:
        config = SimpleNamespace(num_classes=2)

        def replace(self, **kwargs):
            return self

    monkeypatch.setattr(train, "EmotionDataModule", EmptyDataModule)
    monkeypatch.setattr(train, "SummaryWriter", DummyWriter)
    monkeypatch.setattr(train, "create_train_state", stub_create_state)
    monkeypatch.setattr(
        train,
        "maybe_restore_checkpoint",
        lambda config: {
            "params": {"w": jnp.array(0.0)},
            "batch_stats": freeze({}),
            "opt_state": None,
            "dynamic_scale": None,
        },
    )
    monkeypatch.setattr(train, "create_resnet", lambda **kwargs: FakeResNet())
    monkeypatch.setattr(train, "build_train_step", lambda *args, **kwargs: lambda state, batch, rng: (state, {}))
    monkeypatch.setattr(train, "build_eval_step", lambda *args, **kwargs: lambda state, batch: ({}, jnp.array([], dtype=jnp.int32)))
    monkeypatch.setattr(train, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "predict_batches", lambda *args, **kwargs: (jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32)))

    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path, batch_size=2),
        output_dir=tmp_path / "run",
        num_epochs=1,
        checkpoint_every=1,
        patience=1,
        log_every=1,
    )

    metrics = train_and_evaluate(config)
    assert math.isnan(metrics["history"]["train_loss"][0])
    assert math.isnan(metrics["history"]["val_loss"][0])
