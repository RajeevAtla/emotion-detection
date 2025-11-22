"""Tests for checkpointing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.core import freeze

from src import checkpointing


def _normalize_tree(tree: Any) -> Any:
    """Convert PRNGKey leaves into comparable NumPy arrays."""
    if isinstance(tree, dict):
        return {key: _normalize_tree(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_normalize_tree(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_normalize_tree(value) for value in tree)
    if isinstance(tree, jax.Array):
        try:
            return np.asarray(tree)
        except TypeError:
            return np.asarray(jax.random.key_data(tree))
    return tree


def test_convert_stringified_int_keys_handles_nested_dicts() -> None:
    """Ensure that Orbax stringified integer keys are converted to ints."""
    payload = {
        "0": {"inner": {"1": 2}},
        "layer": {"weights": jnp.ones((1,))},
    }
    converted = checkpointing.convert_stringified_int_keys(payload)
    assert 0 in converted
    assert 1 in converted[0]["inner"]


def test_convert_stringified_int_keys_handles_sequences() -> None:
    """Ensure tuples, lists, and FrozenDict instances are converted."""
    payload = {
        "0": [("1", "two"), {"3": ["4"]}],
        "layer": freeze({"5": jnp.ones((1,))}),
    }
    converted = checkpointing.convert_stringified_int_keys(payload)
    assert isinstance(converted["layer"], type(freeze({})))
    assert converted[0][0][0] == "1"
    assert 3 in converted[0][1]


def test_convert_stringified_int_keys_handles_custom_mappings() -> None:
    """Mappings with restricted constructors should fall back to dicts."""

    class StrictMapping(dict):
        def __init__(self) -> None:
            super().__init__()

    payload = StrictMapping()
    payload["0"] = {"1": "2"}
    converted = checkpointing.convert_stringified_int_keys(payload)
    assert isinstance(converted, dict)
    assert 0 in converted


def test_convert_stringified_int_keys_preserves_failed_int_casts() -> None:
    """Keys that mimic digits but fail int() should remain unchanged."""

    class FakeDigit(str):
        def isdigit(self) -> bool:
            return True

    class OddStr(str):
        def strip(self, chars: str | None = None) -> str:
            return FakeDigit("abc")

    payload = {OddStr("10"): {"value": 1}}
    converted = checkpointing.convert_stringified_int_keys(payload)
    assert any(isinstance(key, OddStr) for key in converted.keys())


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


def test_save_payload_prunes_oldest_checkpoints(tmp_path: Path) -> None:
    """Saving more than max_checkpoints should delete the oldest."""
    layout = checkpointing.CheckpointLayout(
        directory=tmp_path, max_checkpoints=1
    )
    payload = {
        "params": freeze({"w": jnp.ones((1,), dtype=jnp.float32)}),
    }
    first = checkpointing.save_payload(payload, layout=layout, epoch=1)
    second = checkpointing.save_payload(payload, layout=layout, epoch=2)
    assert not first.exists()
    assert second.exists()


def test_apply_nnx_state_updates_module() -> None:
    """apply_nnx_state should mutate modules based on restored state."""

    class TinyLinear(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.linear = nnx.Linear(1, 1, rngs=rngs)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.linear(x)

    module = TinyLinear(nnx.Rngs(0))
    state = checkpointing.nnx_state(module)
    mutated = checkpointing.convert_stringified_int_keys(state)
    mutated["linear"]["kernel"] = mutated["linear"]["kernel"] + 1.0
    checkpointing.apply_nnx_state(module, mutated)
    updated = checkpointing.nnx_state(module)
    chex.assert_trees_all_close(mutated, updated)


def test_apply_nnx_state_to_object_handles_rngs() -> None:
    """Non-module nnx objects (like rngs) should be updatable."""
    rngs = nnx.Rngs(0)
    state = checkpointing.nnx_state(rngs)
    checkpointing.apply_nnx_state_to_object(rngs, state)
    refreshed = checkpointing.nnx_state(rngs)
    chex.assert_trees_all_close(
        _normalize_tree(state), _normalize_tree(refreshed)
    )


def test_restore_payload_returns_none_when_handler_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When Orbax returns None, the helper should propagate it."""
    layout = checkpointing.CheckpointLayout(directory=tmp_path)
    payload = {"dummy": 1}
    checkpointing.save_payload(payload, layout=layout, epoch=1)

    class DummyCheckpointer:
        def restore(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        checkpointing.ocp, "PyTreeCheckpointer", lambda: DummyCheckpointer()
    )
    restored = checkpointing.restore_payload(
        tmp_path / "epoch_0001", template=payload
    )
    assert restored is None


def test_restore_payload_returns_none(monkeypatch, tmp_path: Path) -> None:
    """restore_payload should return None when Orbax does."""

    class FakeCheckpointer:
        def restore(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        checkpointing.ocp, "PyTreeCheckpointer", lambda: FakeCheckpointer()
    )
    restored = checkpointing.restore_payload(tmp_path / "missing")
    assert restored is None
