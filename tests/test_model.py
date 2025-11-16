"""Tests for model configuration and ResNet building blocks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from src import model


def _dummy_input(channels: int = 1) -> jnp.ndarray:
    return jnp.ones((1, 48, 48, channels), dtype=jnp.float32)


def test_resnet_config_presets_and_overrides() -> None:
    cfg = model.resnet_config(18, num_classes=5)
    assert cfg.depth == 18
    assert cfg.num_classes == 5
    with pytest.raises(ValueError):
        model.resnet_config(99)


def test_basic_and_bottleneck_blocks() -> None:
    rngs = nnx.Rngs(0)
    basic = model.BasicBlock(
        in_features=32,
        features=32,
        strides=(2, 2),
        use_projection=True,
        rngs=rngs,
    )
    basic.train()
    x = jnp.ones((1, 32, 32, 32), dtype=jnp.float32)
    y = basic(x)
    chex.assert_shape(y, (1, 16, 16, 32))

    bottleneck = model.BottleneckBlock(
        in_features=64,
        features=64,
        use_projection=False,
        rngs=nnx.Rngs(1),
    )
    bottleneck.eval()
    z = jnp.ones((1, 16, 16, 64), dtype=jnp.float32)
    out = bottleneck(z)
    chex.assert_shape(out, z.shape)


def test_create_resnet_and_forward_features() -> None:
    resnet = model.create_resnet(
        depth=18,
        num_classes=3,
        include_top=False,
        rngs=nnx.Rngs(2),
    )
    logits, features = resnet(
        _dummy_input(), train=False, return_features=True
    )
    expected_width = resnet.config.stage_widths[-1] * resnet.config.width_multiplier
    chex.assert_shape(logits, (1, expected_width))
    assert "stem" in features
    assert "pooled" in features
    assert "logits" not in features


def test_build_finetune_mask_variants() -> None:
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(3))
    params = nnx.state(resnet, nnx.Param)

    default_mask = model.build_finetune_mask(
        params,
        config=resnet.config,
    )
    assert default_mask["stem_conv"]["kernel"].value
    assert default_mask["classifier"]["kernel"].value

    frozen_cfg = replace(
        resnet.config,
        freeze_stem=True,
        freeze_classifier=True,
        frozen_stages=(1,),
    )
    frozen_mask = model.build_finetune_mask(params, config=frozen_cfg)
    assert not frozen_mask["stem_conv"]["kernel"].value
    assert not frozen_mask["classifier"]["kernel"].value
    assert not frozen_mask["stage1_block1"]["conv1"]["kernel"].value
    assert frozen_mask["stage3_block1"]["conv1"]["kernel"].value


def test_resnet_forward_updates_batchnorm_state() -> None:
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(4))
    params_before = nnx.state(resnet, nnx.BatchStat)
    before_snapshot = jax.tree_util.tree_map(
        lambda array: np.asarray(array).copy(),
        params_before,
    )
    resnet.train()
    _ = resnet(_dummy_input(), train=True)
    params_after = nnx.state(resnet, nnx.BatchStat)
    after_snapshot = jax.tree_util.tree_map(
        lambda array: np.asarray(array),
        params_after,
    )
    before_leaves = jax.tree_util.tree_leaves(before_snapshot)
    after_leaves = jax.tree_util.tree_leaves(after_snapshot)
    assert any(
        not np.allclose(before, after)
        for before, after in zip(before_leaves, after_leaves)
    )


def test_create_resnet_invalid_depth() -> None:
    with pytest.raises(ValueError):
        model.create_resnet(depth=99, rngs=nnx.Rngs(0))


def test_maybe_load_pretrained_params_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        model.maybe_load_pretrained_params(Path("dummy"))
