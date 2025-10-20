"""Tests for model configuration and ResNet building blocks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import warnings

import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
warnings.filterwarnings(
    "ignore",
    message="Sharding info not provided when restoring",
    category=UserWarning,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Sharding info not provided when restoring:UserWarning"
)

from flax.core import unfreeze

from src import model


def test_resnet_config_presets_and_overrides() -> None:
    """Test configuration presets and invalid depth handling."""
    cfg = model.resnet_config(18, num_classes=5)
    assert cfg.depth == 18
    assert cfg.num_classes == 5
    with pytest.raises(ValueError):
        model.resnet_config(99)


def test_basic_and_bottleneck_blocks() -> None:
    """Test initialization and forward pass across block variants."""
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
    """Test feature dictionary outputs when requesting intermediate tensors."""
    resnet = model.create_resnet(depth=18, num_classes=3, include_top=False)
    variables = resnet.init(jax.random.PRNGKey(1), jnp.ones((1, 48, 48, 1)), train=False)
    logits, features = resnet.apply(variables, jnp.ones((1, 48, 48, 1)), train=False, return_features=True)
    assert features["stem"].ndim == 4
    assert "logits" not in features


def test_build_finetune_mask_variants() -> None:
    """Test finetune masks for default and frozen stage configurations."""
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
    """Test that checkpoint round-trips reproduce identical parameters."""
    resnet = model.create_resnet(depth=18, num_classes=2)
    params = resnet.init(jax.random.PRNGKey(3), jnp.ones((1, 48, 48, 1)), train=False)["params"]

    checkpoint_path = tmp_path / "resnet_ckpt"
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = ocp.args.PyTreeSave(params)
    checkpointer.save(str(checkpoint_path), params, save_args=save_args, force=True)

    restored = model.maybe_load_pretrained_params(
        params,
        config=replace(resnet.config, checkpoint_path=checkpoint_path),
    )
    orig_flat, _ = tree_util.tree_flatten(params)
    restored_flat, _ = tree_util.tree_flatten(restored)
    for original, replica in zip(orig_flat, restored_flat):
        np.testing.assert_allclose(np.asarray(original), np.asarray(replica))


def test_residual_block_not_implemented() -> None:
    """Test that abstract residual block methods raise NotImplementedError."""
    block = model.ResidualBlock(features=16)
    with pytest.raises(NotImplementedError):
        block.init(jax.random.PRNGKey(0), jnp.ones((1, 4, 4, 16)), train=True)
    with pytest.raises(NotImplementedError):
        model.ResidualBlock.__call__(block, jnp.ones((1, 4, 4, 16)), train=True)


def test_basic_block_requires_projection() -> None:
    """Test that channel mismatches without projection raise an error."""
    block = model.BasicBlock(features=16, use_projection=False)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 8, 8, 8))
    with pytest.raises(ValueError):
        block.init(key, x, train=True)


def test_bottleneck_projection_paths() -> None:
    """Test bottleneck block projection branches with stride changes."""
    key = jax.random.PRNGKey(0)
    block = model.BottleneckBlock(features=32, strides=(2, 2), use_projection=True)
    x = jnp.ones((1, 8, 8, 16))
    variables = block.init(key, x, train=True)
    y, _ = block.apply(variables, x, train=True, mutable=["batch_stats"])
    assert y.shape[-1] == 32


def test_bottleneck_block_requires_projection() -> None:
    """Test that bottleneck blocks validate projection requirements."""
    key = jax.random.PRNGKey(1)
    block = model.BottleneckBlock(features=32, use_projection=False)
    x = jnp.ones((1, 8, 8, 16))
    with pytest.raises(ValueError):
        block.init(key, x, train=True)


def test_resnet_input_channel_mismatch_raises() -> None:
    """Test that input channel mismatches raise a ValueError."""
    resnet = model.create_resnet(depth=18, num_classes=2)
    variables = resnet.init(jax.random.PRNGKey(2), jnp.ones((1, 48, 48, 1)), train=False)
    with pytest.raises(ValueError):
        resnet.apply(variables, jnp.ones((1, 48, 48, 3)), train=False)


def test_resnet_input_projection_branch() -> None:
    """Test optional input projection handling during forward passes."""
    resnet = model.create_resnet(depth=18, num_classes=2, input_projection_channels=3)
    variables = resnet.init(jax.random.PRNGKey(9), jnp.ones((1, 48, 48, 3)), train=False)
    logits = resnet.apply(variables, jnp.ones((1, 48, 48, 3)), train=False)
    assert logits.shape == (1, 2)


def test_resnet_projection_on_width_mismatch() -> None:
    """Test that stage projection handles width mismatches."""
    base = model.create_resnet(depth=18, num_classes=2)
    altered = model.ResNet(config=replace(base.config, stem_width=32))
    variables = altered.init(jax.random.PRNGKey(3), jnp.ones((1, 48, 48, 1)), train=False)
    logits = altered.apply(variables, jnp.ones((1, 48, 48, 1)), train=False)
    assert logits.shape == (1, 2)


def test_resnet_dropout_applies_when_requested() -> None:
    """Test dropout application when requested during training."""
    base = model.create_resnet(depth=18, num_classes=2)
    resnet = model.ResNet(config=base.config, include_top=True, dropout_rate=0.5)
    variables = resnet.init(jax.random.PRNGKey(4), jnp.ones((1, 48, 48, 1)), train=True)
    logits, updates = resnet.apply(
        variables,
        jnp.ones((1, 48, 48, 1)),
        train=True,
        mutable=["batch_stats"],
        rngs={"dropout": jax.random.PRNGKey(5)},
    )
    assert logits.shape == (1, 2)
    assert "batch_stats" in updates


def test_create_resnet_invalid_depth() -> None:
    """Test that unsupported depths raise errors."""
    with pytest.raises(ValueError):
        model.create_resnet(depth=99)


def test_build_finetune_mask_without_container() -> None:
    """Test finetune mask creation when a params dict is supplied."""
    resnet = model.create_resnet(depth=18, num_classes=2)
    params = resnet.init(jax.random.PRNGKey(6), jnp.ones((1, 48, 48, 1)), train=False)["params"]
    unfrozen = unfreeze(params)
    mask = model.build_finetune_mask(unfrozen, config=resnet.config, freeze_classifier=True)
    assert isinstance(mask, model.FrozenDict)


def test_maybe_load_pretrained_params_no_checkpoint() -> None:
    """Test that absence of a checkpoint returns the original parameters."""
    resnet = model.create_resnet(depth=18, num_classes=2)
    params = resnet.init(jax.random.PRNGKey(7), jnp.ones((1, 48, 48, 1)), train=False)["params"]
    assert model.maybe_load_pretrained_params(params, config=resnet.config) is params


def test_build_finetune_mask_with_container() -> None:
    """Test finetune mask creation when parameters are wrapped in a dict."""
    resnet = model.create_resnet(depth=18, num_classes=2)
    params = resnet.init(jax.random.PRNGKey(11), jnp.ones((1, 48, 48, 1)), train=False)["params"]
    mask = model.build_finetune_mask({"params": params}, config=resnet.config)
    assert isinstance(mask, model.FrozenDict)


def test_maybe_load_pretrained_params_freezes_restored(monkeypatch) -> None:
    """Test checkpoint restoration with mocked checkpointer behavior."""
    class DummyCheckpointer:
        def restore(self, path: str, item, restore_args=None):
            return {"params": {"w": np.array([1.0])}}

    monkeypatch.setattr(model.ocp, "PyTreeCheckpointer", lambda: DummyCheckpointer())
    resnet = model.create_resnet(depth=18, num_classes=2)
    params = resnet.init(jax.random.PRNGKey(8), jnp.ones((1, 48, 48, 1)), train=False)["params"]
    restored = model.maybe_load_pretrained_params(
        params,
        config=replace(resnet.config, checkpoint_path=Path("dummy")),
    )
    assert isinstance(restored, model.FrozenDict)
