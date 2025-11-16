"""NNX training tests for the emotion detection pipeline."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from src import train
from src.data import DataModuleConfig
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
    initialize_nnx_model,
    maybe_restore_checkpoint,
    predict_batches,
    save_checkpoint,
    train_and_evaluate,
)


class TinyNNXModel(nnx.Module):
    """Minimal NNX module used throughout the tests."""

    def __init__(self, rngs: nnx.Rngs):
        """Initialize the linear layer."""
        self.linear = nnx.Linear(1, 2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the linear mapping to flattened inputs."""
        flat = x.reshape(x.shape[0], -1)
        return self.linear(flat)

    def train(self) -> None:  # pragma: no cover - simple toggles
        """Mark the module as training."""
        return

    def eval(self) -> None:  # pragma: no cover - simple toggles
        """Mark the module as evaluating."""
        return


def _make_train_state() -> TrainState:
    rngs = nnx.Rngs(0)
    model = TinyNNXModel(rngs=rngs.fork())
    tx = optax.adam(1e-3)
    params_tree = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    opt_state = tx.init(params_tree)
    return TrainState(
        model=model,
        tx=tx,
        opt_state=opt_state,
        rngs=rngs,
        dynamic_scale=None,
    )


def test_learning_rate_schedule_shape() -> None:
    """Verify the learning-rate schedule behaviour."""
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
    """Exercise optimizer masking and accumulation."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("out"),
        gradient_accumulation_steps=2,
    )

    def schedule(_: int) -> float:
        return 0.1

    params = {"w": jnp.array(1.0)}
    mask = {"w": True}
    tx = create_optimizer(config, schedule, mask)
    opt_state = tx.init(params)
    assert hasattr(opt_state, "inner_state")


def test_create_train_state_initializes_optimizer(tmp_path: Path) -> None:
    """Ensure ``create_train_state`` wires the optimizer."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
        model_depth=18,
    )
    model, rngs = initialize_nnx_model(config, num_classes=2)
    schedule = create_learning_rate_schedule(config, steps_per_epoch=1)
    state = create_train_state(model, config, schedule, rngs=rngs)
    assert isinstance(state, TrainState)
    assert jax.tree_util.tree_leaves(state.opt_state)


def test_training_config_resolves_paths(tmp_path: Path) -> None:
    """TrainingConfig should normalize path-like fields."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path / "out",
        pretrained_checkpoint=tmp_path / "ckpt",
        resume_checkpoint=tmp_path / "resume",
    )
    assert config.output_dir.exists()
    assert isinstance(config.pretrained_checkpoint, Path)
    assert isinstance(config.resume_checkpoint, Path)


def test_create_train_state_respects_freeze_flags(tmp_path: Path) -> None:
    """Creating a train state should respect freeze/mask flags."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
        freeze_stem=True,
        freeze_classifier=True,
        frozen_stages=(1,),
    )
    model, rngs = initialize_nnx_model(config, num_classes=2)
    schedule = create_learning_rate_schedule(config, steps_per_epoch=1)
    state = create_train_state(model, config, schedule, rngs=rngs)
    assert isinstance(state, TrainState)


def test_cross_entropy_loss_with_smoothing_and_weights() -> None:
    """Cross entropy should handle smoothing and weighting."""
    logits = jnp.array([[2.0, 0.5]])
    labels = jnp.array([0])
    weights = jnp.array([0.7, 0.3])
    loss = cross_entropy_loss(
        logits, labels, label_smoothing=0.1, class_weights=weights
    )
    assert float(loss) > 0.0


def test_build_train_step_updates_optimizer() -> None:
    """Training step should advance the optimizer."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("out"),
        batch_size=1,
    )
    train_step = build_train_step(config, class_weights=None)
    state = _make_train_state()
    initial_state = state.opt_state
    batch = (jnp.ones((1, 1, 1, 1)), jnp.zeros((1,), dtype=jnp.int32))
    new_state, metrics = train_step(state, batch)
    before = jax.tree_util.tree_leaves(initial_state)
    after = jax.tree_util.tree_leaves(new_state.opt_state)
    assert any(np.any(before_leaf != after_leaf) for before_leaf, after_leaf in zip(before, after))
    chex.assert_tree_all_finite(metrics)


def test_build_eval_step_returns_predictions() -> None:
    """Eval step should emit metrics and predictions."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=Path(".")),
        output_dir=Path("out"),
    )
    eval_step = build_eval_step(config)
    state = _make_train_state()
    batch = (jnp.ones((1, 1, 1, 1)), jnp.zeros((1,), dtype=jnp.int32))
    metrics, preds = eval_step(state, batch)
    chex.assert_tree_all_finite(metrics)
    assert preds.shape == (1,)


def test_compute_confusion_matrix_and_format() -> None:
    """Confusion matrix helpers should remain consistent."""
    preds = jnp.array([0, 1, 1])
    labels = jnp.array([0, 0, 1])
    cm = compute_confusion_matrix(preds, labels, num_classes=2)
    assert cm.tolist() == [[1, 1], [0, 1]]
    table = format_confusion_matrix(cm, ["neg", "pos"])
    assert "neg" in table and "pos" in table


def test_compute_f1_metrics_edge_cases() -> None:
    """F1 metrics should handle empty or zero-valued matrices."""
    empty = np.zeros((0, 0), dtype=np.int32)
    micro, macro, scores = train.compute_f1_metrics(empty)
    assert math.isnan(micro) and math.isnan(macro) and scores == []
    zero_matrix = np.zeros((2, 2), dtype=np.int32)
    micro, macro, scores = train.compute_f1_metrics(zero_matrix)
    assert math.isnan(macro) and math.isnan(micro)
    skewed = np.array([[0, 1], [1, 0]], dtype=np.int32)
    _, _, scores = train.compute_f1_metrics(skewed)
    assert scores[0] == 0.0


def test_save_and_restore_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Saving and restoring a checkpoint should be lossless."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
    )
    state = _make_train_state()
    base_params = nnx.to_pure_dict(nnx.state(state.model, nnx.Param))
    ckpt_path = save_checkpoint(state, config, epoch=1)
    for _, variable in nnx.state(state.model, nnx.Param).flat_state():
        variable.value = variable.value + 1.0
    resume_config = replace(config, resume_checkpoint=ckpt_path)
    restored = maybe_restore_checkpoint(resume_config, state)
    assert restored is not None
    restored_params = nnx.to_pure_dict(nnx.state(restored.model, nnx.Param))
    chex.assert_trees_all_close(restored_params, base_params)
    before_opt = jax.tree_util.tree_leaves(state.opt_state)
    after_opt = jax.tree_util.tree_leaves(restored.opt_state)
    assert all(np.array_equal(a, b) for a, b in zip(before_opt, after_opt))


def test_predict_batches_handles_empty(tmp_path: Path) -> None:
    """Prediction helper should handle empty batches."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
    )
    state = _make_train_state()
    preds, labels = predict_batches(state, [], config)
    assert preds.size == 0 and labels.size == 0


def test_predict_batches_returns_predictions(tmp_path: Path) -> None:
    """Prediction helper should return logits for real batches."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
    )
    state = _make_train_state()
    batches = [
        (jnp.ones((1, 1, 1, 1)), jnp.zeros((1,), dtype=jnp.int32)),
        (jnp.ones((1, 1, 1, 1)), jnp.zeros((1,), dtype=jnp.int32)),
    ]
    preds, labels = predict_batches(state, batches, config)
    assert preds.shape == labels.shape == (2,)


def test_train_and_evaluate_runs_with_stubs(monkeypatch, tmp_path: Path) -> None:
    """The main training loop should run with stubbed helpers."""
    class MinimalDataModule:
        class_weights = jnp.ones(2)

        def __init__(self, *_args, **_kwargs):
            pass

        def setup(self):
            return

        def split_counts(self):
            return {
                "train": {"neg": 1, "pos": 1},
                "val": {"neg": 1, "pos": 1},
                "test": {"neg": 1, "pos": 1},
            }

        def train_batches(self, **kwargs):
            def generator():
                yield jnp.ones((1, 1, 1, 1)), jnp.zeros((1,), dtype=jnp.int32)

            return generator()

        def val_batches(self, **kwargs):
            return [(jnp.ones((1, 1, 1, 1)), jnp.zeros((1,), dtype=jnp.int32))]

        def test_batches(self, **kwargs):
            return [(jnp.ones((1, 1, 1, 1)), jnp.zeros((1,), dtype=jnp.int32))]

    monkeypatch.setattr(train, "EmotionDataModule", MinimalDataModule)

    class FakeModel(TinyNNXModel):
        pass

    fake_state = _make_train_state()

    monkeypatch.setattr(
        train,
        "initialize_nnx_model",
        lambda *args, **kwargs: (FakeModel(rngs=nnx.Rngs(0)), nnx.Rngs(1)),
    )
    monkeypatch.setattr(
        train,
        "create_train_state",
        lambda *args, **kwargs: fake_state,
    )

    step_counter = {"count": 0}

    def stub_train_step(state, batch):
        step_counter["count"] += 1
        return state, {
            "loss": jnp.array(0.1, dtype=jnp.float32),
            "accuracy": jnp.array(1.0, dtype=jnp.float32),
        }

    def stub_eval_step(state, batch):
        return (
            {
                "loss": jnp.array(0.1, dtype=jnp.float32),
                "accuracy": jnp.array(1.0, dtype=jnp.float32),
            },
            jnp.zeros(batch[0].shape[0], dtype=jnp.int32),
        )

    predict_calls = {"count": 0}

    def stub_predict_batches(state, batches, config_obj):
        predict_calls["count"] += 1
        return jnp.zeros((1,), dtype=jnp.int32), jnp.zeros((1,), dtype=jnp.int32)

    saved_epochs: list[int] = []

    def stub_save_checkpoint(state, config_obj, epoch):
        saved_epochs.append(epoch)
        path = tmp_path / f"epoch_{epoch:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(train, "build_train_step", lambda *a, **k: stub_train_step)
    monkeypatch.setattr(train, "build_eval_step", lambda *a, **k: stub_eval_step)
    monkeypatch.setattr(train, "predict_batches", stub_predict_batches)
    monkeypatch.setattr(train, "save_checkpoint", stub_save_checkpoint)
    monkeypatch.setattr(
        train, "maybe_restore_checkpoint", lambda config, state: _make_train_state()
    )
    monkeypatch.setattr(
        train.checkpointing,
        "restore_payload",
        lambda path, template: template,
    )

    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path, batch_size=1),
        output_dir=tmp_path / "run",
        num_epochs=2,
        checkpoint_every=1,
        patience=2,
        log_every=1,
    )

    metrics = train_and_evaluate(config)
    assert step_counter["count"] > 0
    assert predict_calls["count"] == 1
    assert metrics["best_checkpoint"] is not None
    assert saved_epochs, "checkpoints were not saved"


def test_train_and_evaluate_handles_empty_batches(monkeypatch, tmp_path: Path) -> None:
    """Training loop should tolerate empty training/validation batches."""
    class EmptyDataModule:
        class_weights = jnp.ones(2)

        def __init__(self, *_args, **_kwargs):
            pass

        def setup(self):
            return

        def split_counts(self):
            return {
                "train": {"neg": 1},
                "val": {"neg": 1},
                "test": {"neg": 1},
            }

        def train_batches(self, **kwargs):
            if False:
                yield None
            return iter(())

        def val_batches(self, **kwargs):
            return []

        def test_batches(self, **kwargs):
            return []

    monkeypatch.setattr(train, "EmotionDataModule", EmptyDataModule)
    monkeypatch.setattr(
        train,
        "initialize_nnx_model",
        lambda *a, **k: (TinyNNXModel(rngs=nnx.Rngs(0)), nnx.Rngs(1)),
    )
    monkeypatch.setattr(train, "create_train_state", lambda *a, **k: _make_train_state())
    monkeypatch.setattr(train, "build_train_step", lambda *a, **k: lambda s, b: (s, {"loss": jnp.array(0.0), "accuracy": jnp.array(0.0)}))
    monkeypatch.setattr(train, "build_eval_step", lambda *a, **k: lambda s, b: ({"loss": jnp.array(0.0), "accuracy": jnp.array(0.0)}, jnp.array([], dtype=jnp.int32)))
    monkeypatch.setattr(train, "predict_batches", lambda *a, **k: (jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32)))
    monkeypatch.setattr(train, "save_checkpoint", lambda *a, **k: tmp_path / "epoch_0001")
    monkeypatch.setattr(train, "maybe_restore_checkpoint", lambda config, state: None)

    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path, batch_size=1),
        output_dir=tmp_path / "empty",
        num_epochs=1,
        patience=0,
    )
    metrics = train_and_evaluate(config)
    assert math.isnan(metrics["train_loss"]) or metrics["train_loss"] == 0.0
def test_maybe_restore_checkpoint_none(tmp_path: Path, monkeypatch) -> None:
    """maybe_restore_checkpoint should handle absence or failure."""
    config = TrainingConfig(
        data=DataModuleConfig(data_dir=tmp_path),
        output_dir=tmp_path,
    )
    state = _make_train_state()
    assert train.maybe_restore_checkpoint(config, state) is None
    resume_config = replace(config, resume_checkpoint=tmp_path / "missing")
    monkeypatch.setattr(
        train.checkpointing, "restore_payload", lambda *args, **kwargs: None
    )
    assert train.maybe_restore_checkpoint(resume_config, state) is None
