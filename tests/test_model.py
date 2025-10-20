"""Tests for model configuration and ResNet building blocks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from src import model


def test_resnet_config_presets_and_overrides() -> None:
    cfg = model.resnet_config(18, num_classes=5)
    assert cfg.depth == 18
    assert cfg.num_classes == 5
    with pytest.raises(ValueError):
        model.resnet_config(99)


def test_basic_and_bottleneck_blocks() -> None:
    key = jax.random.PRNGKey(0)
    basic = model.BasicBlock(features=32, strides=(2, 2), use_projection=True)
    x = jnp.ones((1, 32, 32, 32))
    variables = basic.init(key, x, train=True)
    y, _ = basic.apply(variables, x, train=True, mutable=["batch_stats"])
    assert y.shape[-1] == 32

    bottleneck = model.BottleneckBlock(features=64, strides=(1, 1), use_projection=False)
    x = jnp.ones((1, 16, 16, 64))
    variables = bottleneck.init(key, x, train=True)
    y, _ = bottleneck.apply(variables, x, train=True, mutable=["batch_stats"])
    assert y.shape[-1] == 64


def test_create_resnet_and_forward_features() -> None:
    resnet = model.create_resnet(depth=18, num_classes=3, include_top=False)
    variables = resnet.init(jax.random.PRNGKey(1), jnp.ones((1, 48, 48, 1)), train=False)
    logits, features = resnet.apply(variables, jnp.ones((1, 48, 48, 1)), train=False, return_features=True)
    assert features["stem"].ndim == 4
    assert "logits" not in features


def test_build_finetune_mask_variants() -> None:
    resnet = model.create_resnet(depth=18, num_classes=2)
    params = resnet.init(jax.random.PRNGKey(2), jnp.ones((1, 48, 48, 1)), train=False)["params"]

    no_freeze_mask = model.build_finetune_mask({"params": params}, config=resnet.config)
    assert np.all(np.asarray(no_freeze_mask["params"]["stem_conv"]["kernel"]))

    cfg = replace(resnet.config, freeze_stem=True, freeze_classifier=True, frozen_stages=(1,))
    mask = model.build_finetune_mask({"params": params}, config=cfg)
    assert not np.any(np.asarray(mask["params"]["stem_conv"]["kernel"]))
    assert not np.any(np.asarray(mask["params"]["classifier"]["kernel"]))
    assert not np.any(np.asarray(mask["params"]["stage1_block1"]["conv1"]["kernel"]))
    assert np.all(np.asarray(mask["params"]["stage3_block1"]["conv1"]["kernel"]))


def test_maybe_load_pretrained_params_roundtrip(tmp_path: Path) -> None:
    resnet = model.create_resnet(depth=18, num_classes=2)
    params = resnet.init(jax.random.PRNGKey(3), jnp.ones((1, 48, 48, 1)), train=False)["params"]

    checkpoint_path = tmp_path / "resnet_ckpt"
    ocp.PyTreeCheckpointer().save(str(checkpoint_path), params, force=True)

    restored = model.maybe_load_pretrained_params(
        params,
        config=replace(resnet.config, checkpoint_path=checkpoint_path),
    )
    orig_flat, _ = tree_util.tree_flatten(params)
    restored_flat, _ = tree_util.tree_flatten(restored)
    for original, replica in zip(orig_flat, restored_flat):
        np.testing.assert_allclose(np.asarray(original), np.asarray(replica))
