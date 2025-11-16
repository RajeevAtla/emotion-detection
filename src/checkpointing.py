"""Orbax checkpoint helpers with Flax NNX-aware utilities."""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import orbax.checkpoint as ocp
from flax import nnx
from flax.core import FrozenDict, freeze

PyTree = Any


@dataclass(frozen=True)
class CheckpointLayout:
    """Metadata describing how checkpoints are stored on disk."""

    directory: Path
    max_checkpoints: int = 3
    filename_template: str = "epoch_{epoch:04d}"
    glob_pattern: str = "epoch_*"

    def target_path(self, epoch: int) -> Path:
        """Return the checkpoint directory for a particular epoch."""
        return self.directory / self.filename_template.format(epoch=epoch)


def save_payload(
    payload: Mapping[str, PyTree],
    *,
    layout: CheckpointLayout,
    epoch: int,
) -> Path:
    """Persist a PyTree payload using Orbax."""
    layout.directory.mkdir(parents=True, exist_ok=True)
    target_path = layout.target_path(epoch)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = ocp.args.PyTreeSave(payload)
    checkpointer.save(str(target_path), payload, save_args=save_args, force=True)

    existing = sorted(layout.directory.glob(layout.glob_pattern))
    excess = len(existing) - layout.max_checkpoints
    if excess > 0:
        for path in existing[:excess]:
            shutil.rmtree(path, ignore_errors=True)
    return target_path


def restore_payload(
    path: Path,
    *,
    template: Mapping[str, PyTree] | None = None,
) -> Mapping[str, PyTree] | None:
    """Restore a PyTree payload from disk."""
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = (
        ocp.args.PyTreeRestore(item=template)
        if template is not None
        else None
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sharding info not provided when restoring",
            category=UserWarning,
        )
        restored = checkpointer.restore(
            str(path),
            item=template,
            restore_args=restore_args,
        )
    if restored is None:
        return None
    return convert_stringified_int_keys(restored)


@overload
def convert_stringified_int_keys(tree: FrozenDict[str, PyTree]) -> FrozenDict[str, PyTree]:
    ...


@overload
def convert_stringified_int_keys(tree: Mapping[str, PyTree]) -> Mapping[str, PyTree]:
    ...


def convert_stringified_int_keys(tree: PyTree) -> PyTree:
    """Convert stringified integer keys produced by Orbax back to integers."""
    if isinstance(tree, FrozenDict):
        converted = {
            _maybe_int_key(key): convert_stringified_int_keys(value)
            for key, value in tree.items()
        }
        return freeze(converted)
    if isinstance(tree, Mapping):
        converted_dict = {
            _maybe_int_key(key): convert_stringified_int_keys(value)
            for key, value in tree.items()
        }
        try:
            return tree.__class__(converted_dict)
        except Exception:
            return converted_dict
    if isinstance(tree, tuple):
        return tuple(convert_stringified_int_keys(value) for value in tree)
    if isinstance(tree, list):
        return [convert_stringified_int_keys(value) for value in tree]
    return tree


def _state_to_pure_dict(state: nnx.State) -> dict[str, Any]:
    """Convert an ``nnx.State`` into a serialization-friendly mapping."""
    return state.to_pure_dict()


def nnx_state(module: nnx.Module | nnx.Optimizer | nnx.Rngs) -> PyTree:
    """Return the serialized state for an ``nnx`` object."""
    return _state_to_pure_dict(nnx.state(module))


def apply_nnx_state(module: nnx.Module, state: PyTree) -> None:
    """Update an ``nnx.Module`` using a restored checkpoint state."""
    pure_dict = convert_stringified_int_keys(state)
    current_state = nnx.state(module)
    current_state.replace_by_pure_dict(pure_dict)
    nnx.update(module, current_state)


def apply_nnx_state_to_object(obj: nnx.Optimizer | nnx.Rngs, state: PyTree) -> None:
    """Update non-module ``nnx`` objects (e.g., optimizers) with new state."""
    pure_dict = convert_stringified_int_keys(state)
    current_state = nnx.state(obj)
    current_state.replace_by_pure_dict(pure_dict)
    nnx.update(obj, current_state)


def _maybe_int_key(key: Any) -> Any:
    if isinstance(key, str):
        stripped = key.strip()
        if stripped.isdigit() or (
            stripped.startswith("-") and stripped[1:].isdigit()
        ):
            try:
                return int(stripped)
            except ValueError:
                return key
    return key
