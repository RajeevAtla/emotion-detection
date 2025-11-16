"""Tests for checkpointing utilities."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from flax.core import freeze

from src import checkpointing


def test_convert_stringified_int_keys_handles_nested_dicts() -> None:
    """Ensure that Orbax stringified integer keys are converted to ints."""
    payload = {
        "0": {"inner": {"1": 2}},
        "layer": {"weights": jnp.ones((1,))},
    }
    converted = checkpointing.convert_stringified_int_keys(payload)
    assert 0 in converted
    assert 1 in converted[0]["inner"]


def test_save_and_restore_payload_round_trip(tmp_path: Path) -> None:
    """Test saving and restoring payloads via the helper."""
    layout = checkpointing.CheckpointLayout(
        directory=tmp_path, max_checkpoints=1
    )
    payload = {
        "params": freeze({"w": jnp.ones((1,), dtype=jnp.float32)}),
        "batch_stats": freeze({}),
        "opt_state": {"momentum": jnp.zeros((1,), dtype=jnp.float32)},
        "dynamic_scale": None,
    }
    checkpointing.save_payload(payload, layout=layout, epoch=1)
    restored = checkpointing.restore_payload(
        tmp_path / "epoch_0001", template=payload
    )
    assert restored is not None
    assert "params" in restored
