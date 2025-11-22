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

from src import checkpointing, model


def _dummy_input(channels: int = 1) -> jnp.ndarray:
    return jnp.ones((1, 48, 48, channels), dtype=jnp.float32)


def test_residual_block_call_not_implemented() -> None:
    """ResidualBlock base class should require overrides."""
    block = model.ResidualBlock(
        in_features=1,
        features=1,
        rngs=nnx.Rngs(0),
    )
    with pytest.raises(NotImplementedError):
        block(jnp.ones((1, 1, 1, 1), dtype=jnp.float32))


def test_resnet_config_presets_and_overrides() -> None:
    """Ensure resnet_config returns expected presets and failures."""
    cfg = model.resnet_config(18, num_classes=5)
    assert cfg.depth == 18
    assert cfg.num_classes == 5
    with pytest.raises(ValueError):
        model.resnet_config(99)


def test_basic_and_bottleneck_blocks() -> None:
    """Smoke test the residual blocks under train/eval modes."""
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


def test_basic_block_requires_projection_for_channel_mismatch() -> None:
    """Blocks should error when projection is omitted despite width changes."""
    block = model.BasicBlock(
        in_features=32,
        features=64,
        use_projection=False,
        rngs=nnx.Rngs(2),
    )
    with pytest.raises(ValueError):
        block(jnp.ones((1, 8, 8, 32), dtype=jnp.float32))


def test_bottleneck_block_projection_and_error_paths() -> None:
    """Bottleneck blocks should handle projection and mismatch scenarios."""
    projecting = model.BottleneckBlock(
        in_features=32,
        features=64,
        use_projection=True,
        rngs=nnx.Rngs(3),
    )
    y = projecting(jnp.ones((1, 8, 8, 32), dtype=jnp.float32))
    chex.assert_shape(y, (1, 8, 8, 64))

    mismatch = model.BottleneckBlock(
        in_features=32,
        features=64,
        use_projection=False,
        rngs=nnx.Rngs(4),
    )
    with pytest.raises(ValueError):
        mismatch(jnp.ones((1, 8, 8, 32), dtype=jnp.float32))


def test_create_resnet_and_forward_features() -> None:
    """Creating a ResNet should produce pooled features."""
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
    """Finetune masks should respect freezing configuration."""
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
    """Running the model in train mode should update batch statistics."""
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
    """Creating a ResNet with an invalid depth should fail."""
    with pytest.raises(ValueError):
        model.create_resnet(depth=99, rngs=nnx.Rngs(0))


def test_resnet_spatial_downsample_blocks_use_projection() -> None:
    """Stage transitions should enable projection for the first block."""
    resnet = model.create_resnet(
        depth=18,
        num_classes=2,
        rngs=nnx.Rngs(6),
    )
    stage2_block1 = getattr(resnet, "stage2_block1")
    assert stage2_block1.use_projection


def test_resnet_custom_stem_width_requires_projection() -> None:
    """Stage1 block uses projection when stem and stage widths differ."""
    config = model.ResNetConfig(
        depth=18,
        blocks_per_stage=(2, 2, 2, 2),
        block=model.BasicBlock,
        num_classes=2,
        stage_widths=(128, 128, 256, 512),
        stem_width=32,
        input_channels=1,
    )
    resnet = model.ResNet(config=config, rngs=nnx.Rngs(10))
    stage1_block1 = getattr(resnet, "stage1_block1")
    assert stage1_block1.use_projection


def test_resnet_input_projection_and_dropout_path() -> None:
    """Input projection and dropout branches should run without error."""
    resnet = model.create_resnet(
        depth=18,
        num_classes=2,
        input_projection_channels=3,
        dropout_rate=0.5,
        rngs=nnx.Rngs(7),
    )
    logits, features = resnet(
        _dummy_input(), train=True, return_features=True
    )
    chex.assert_shape(logits, (1, 2))
    assert "logits" in features


def test_resnet_rejects_invalid_input_channels() -> None:
    """Inputs with wrong channel counts should raise ValueError."""
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(8))
    bad_input = jnp.ones((1, 48, 48, 3), dtype=jnp.float32)
    with pytest.raises(ValueError):
        resnet(bad_input, train=False)


def test_maybe_load_pretrained_params_respects_none() -> None:
    """Passing ``None`` should be treated as a no-op."""
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(5))
    assert not model.maybe_load_pretrained_params(resnet, checkpoint_path=None)


def test_maybe_load_pretrained_params_missing_path(tmp_path: Path) -> None:
    """Missing paths should raise ``FileNotFoundError``."""
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(6))
    with pytest.raises(FileNotFoundError):
        model.maybe_load_pretrained_params(
            resnet,
            checkpoint_path=tmp_path / "missing",
        )


def test_maybe_load_pretrained_params_missing_model_entry(
    monkeypatch, tmp_path: Path
) -> None:
    """Missing ``model`` entries should raise ``ValueError``."""
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(10))
    ckpt_dir = tmp_path / "ckpt_missing"
    ckpt_dir.mkdir()

    def fake_restore(path, template):
        return {}

    monkeypatch.setattr(model.checkpointing, "restore_payload", fake_restore)
    with pytest.raises(ValueError):
        model.maybe_load_pretrained_params(resnet, checkpoint_path=ckpt_dir)


def test_maybe_load_pretrained_params_missing_model_key_on_disk(
    tmp_path: Path,
) -> None:
    """Checkpoints saved without a ``model`` key should raise ``ValueError``."""
    layout = checkpointing.CheckpointLayout(directory=tmp_path / "bad")
    payload = {"not_model": {}}
    checkpointing.save_payload(payload, layout=layout, epoch=1)
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(11))
    with pytest.raises(ValueError):
        model.maybe_load_pretrained_params(
            resnet,
            checkpoint_path=layout.directory / "epoch_0001",
        )


def test_maybe_load_pretrained_params_success(
    monkeypatch, tmp_path: Path
) -> None:
    """Valid checkpoints should load and update the model."""
    resnet = model.create_resnet(depth=18, num_classes=2, rngs=nnx.Rngs(9))
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    expected_state = model.checkpointing.nnx_state(resnet)

    def fake_restore(path, template):
        assert template["model"]
        return {"model": expected_state}

    applied: dict[str, object] = {}

    def fake_apply(module, state):
        applied["state"] = state

    monkeypatch.setattr(model.checkpointing, "restore_payload", fake_restore)
    monkeypatch.setattr(model.checkpointing, "apply_nnx_state", fake_apply)

    assert model.maybe_load_pretrained_params(
        resnet, checkpoint_path=ckpt_dir
    )
    assert applied["state"] == expected_state
