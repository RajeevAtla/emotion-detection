from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import metrax as mx
import orbax.checkpoint as ocp
from flax.core import freeze
from pydantic import BaseModel, Field, ValidationError

from src.data import (
    AugmentationConfig,
    CLASS_NAMES,
    DataModuleConfig,
    EmotionDataModule,
    apply_augmentations,
    compute_class_weights,
    normalize_image,
)
from src.model import build_finetune_mask, create_resnet, maybe_load_pretrained_params
from src.train import (
    TrainingConfig,
    build_train_step,
    compute_confusion_matrix,
    create_learning_rate_schedule,
    create_train_state,
)


class TrainingConfigSchema(BaseModel):
    """Pydantic representation of the experiment configuration."""

    data_dir: str
    batch_size: int = Field(..., ge=1)
    learning_rate: float
    max_epochs: int
    val_ratio: float = Field(..., ge=0.0, le=0.5)


def _make_data_module(tmp_seed: int = 0) -> EmotionDataModule:
    cfg = DataModuleConfig(
        data_dir=Path("data"),
        batch_size=32,
        val_ratio=0.1,
        seed=tmp_seed,
    )
    cfg.mean = 0.5077
    cfg.std = 0.2550
    module = EmotionDataModule(cfg)
    module.setup()
    return module


def test_data_batch_shapes() -> None:
    module = _make_data_module()
    batch_images, batch_labels = next(module.train_batches(rng_seed=123))
    chex.assert_shape(batch_images, (32, 48, 48, 1))
    chex.assert_rank(batch_labels, 1)
    chex.assert_equal_shape([batch_labels, jnp.zeros((batch_labels.shape[0],), dtype=jnp.int32)])
    assert set(batch_labels.tolist()) <= set(range(len(CLASS_NAMES)))


def test_data_normalization_stats() -> None:
    module = _make_data_module(tmp_seed=1)
    batch_images, _ = next(module.val_batches(batch_size=64))
    assert jnp.isfinite(batch_images).all()
    assert abs(float(batch_images.mean())) < 1.0  # values should be roughly centered near 0
    assert 0.2 < float(batch_images.std()) < 1.5


def test_class_weights_sum_to_one() -> None:
    module = _make_data_module(tmp_seed=2)
    weights = module.class_weights
    chex.assert_shape(weights, (len(CLASS_NAMES),))
    np.testing.assert_allclose(np.asarray(weights).sum(), 1.0, atol=1e-5)


def test_augmentations_deterministic_with_seed() -> None:
    cfg = AugmentationConfig(
        horizontal_flip_prob=1.0,
        rotation_degrees=5.0,
        scale_range=(0.95, 1.05),
        elastic_blur_sigma=0.5,
    )
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    base = np.linspace(0, 1, num=48 * 48, dtype=np.float32).reshape((48, 48, 1))
    aug_a = apply_augmentations(base, rng_a, cfg)
    aug_b = apply_augmentations(base, rng_b, cfg)
    np.testing.assert_allclose(aug_a, aug_b, atol=1e-5)


def test_metrax_accuracy_metric() -> None:
    preds = jnp.asarray([0, 1, 2, 3, 4, 5, 6])
    labels = jnp.asarray([0, 1, 0, 3, 4, 0, 6])
    metric = mx.Accuracy.from_model_output(predictions=preds, labels=labels)
    result = metric.compute()
    assert abs(float(result) - (5 / 7)) < 1e-6


def test_pydantic_config_validation() -> None:
    good_payload = {
        "data_dir": "data",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "max_epochs": 50,
        "val_ratio": 0.1,
    }
    cfg = TrainingConfigSchema(**good_payload)
    assert cfg.batch_size == 64

    bad_payload = {**good_payload, "batch_size": -1}
    try:
        TrainingConfigSchema(**bad_payload)
    except ValidationError as exc:
        assert "batch_size" in str(exc)
    else:  # pragma: no cover - ensure failure if validation does not raise
        raise AssertionError("Negative batch_size should not validate.")


def test_normalize_image_is_noop_without_stats() -> None:
    dummy = np.ones((48, 48, 1), dtype=np.float32)
    out = normalize_image(dummy, None, None)
    np.testing.assert_array_equal(dummy, out)


def test_resnet_forward_and_features() -> None:
    model = create_resnet(depth=18, num_classes=len(CLASS_NAMES))
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((4, 48, 48, 1)), train=False)

    logits = model.apply(variables, jnp.ones((4, 48, 48, 1)), train=False)
    chex.assert_shape(logits, (4, len(CLASS_NAMES)))

    logits_feat, features = model.apply(
        variables, jnp.ones((4, 48, 48, 1)), train=False, return_features=True
    )
    chex.assert_shape(logits_feat, (4, len(CLASS_NAMES)))
    assert {"stem", "stage1", "stage2", "stage3", "stage4", "pooled", "logits"} <= features.keys()


def test_build_finetune_mask_respects_freeze_directives() -> None:
    base_model = create_resnet(depth=18, num_classes=len(CLASS_NAMES))
    variables = base_model.init(jax.random.PRNGKey(1), jnp.ones((1, 48, 48, 1)), train=False)
    cfg = replace(
        base_model.config,
        frozen_stages=(1, 2),
        freeze_stem=True,
        freeze_classifier=True,
    )
    mask = build_finetune_mask(variables, config=cfg)

    params_mask = mask["params"]
    assert params_mask["stem_conv"]["kernel"] is False
    assert params_mask["classifier"]["kernel"] is False
    assert params_mask["stage1_block1"]["conv1"]["kernel"] is False
    assert params_mask["stage3_block1"]["conv1"]["kernel"] is True


def test_maybe_load_pretrained_params_roundtrip(tmp_path: Path) -> None:
    model = create_resnet(depth=18, num_classes=len(CLASS_NAMES))
    variables = model.init(jax.random.PRNGKey(2), jnp.ones((1, 48, 48, 1)), train=False)

    ckpt_dir = tmp_path / "resnet_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(ckpt_dir), variables, force=True)

    restored = maybe_load_pretrained_params(
        variables, config=replace(model.config, checkpoint_path=ckpt_dir)
    )
    chex.assert_trees_all_close(restored, freeze(variables))


def test_compute_confusion_matrix_counts() -> None:
    preds = jnp.array([0, 1, 1, 2, 2])
    labels = jnp.array([0, 1, 2, 2, 1])
    cm = compute_confusion_matrix(preds, labels, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm[1, 1] + cm[1, 2] == 2


def test_train_step_gradients_finite(tmp_path: Path) -> None:
    data_cfg = DataModuleConfig(data_dir=Path("data"), batch_size=2, val_ratio=0.1, seed=0)
    train_cfg = TrainingConfig(
        data=data_cfg,
        output_dir=tmp_path / "runs",
        num_epochs=1,
        batch_size=2,
        log_every=10,
    )
    model = create_resnet(depth=18, num_classes=len(CLASS_NAMES))
    schedule = create_learning_rate_schedule(train_cfg, steps_per_epoch=1)
    state = create_train_state(jax.random.PRNGKey(0), model, train_cfg, schedule)
    train_step = build_train_step(model, train_cfg, class_weights=None)

    images = jnp.ones((train_cfg.batch_size, 48, 48, 1), dtype=jnp.float32)
    labels = jnp.zeros((train_cfg.batch_size,), dtype=jnp.int32)
    new_state, metrics = train_step(state, (images, labels), jax.random.PRNGKey(1))

    chex.assert_tree_all_equal_structs(state.params, new_state.params)
    chex.assert_tree_all_finite(new_state.params)
    assert jnp.isfinite(metrics["loss"])
