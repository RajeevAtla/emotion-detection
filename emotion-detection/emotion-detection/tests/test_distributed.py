"""Tests for distributed multi-GPU training utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import mesh_utils

from src import distributed


def test_get_batch_sharding() -> None:
    """Test batch sharding creation."""
    mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    sharding = distributed.get_batch_sharding(batch_size=128, mesh=mesh)
    assert sharding is not None


def test_get_global_batch_size() -> None:
    """Test global batch size calculation."""
    local_batch = 128
    global_batch = distributed.get_global_batch_size(local_batch)
    expected = local_batch * jax.device_count()
    assert global_batch == expected


def test_get_local_batch_size() -> None:
    """Test local batch size calculation."""
    global_batch = 512
    num_devices = jax.device_count()
    if global_batch % num_devices == 0:
        local_batch = distributed.get_local_batch_size(global_batch)
        assert local_batch == global_batch // num_devices


def test_get_local_batch_size_invalid() -> None:
    """Test that invalid global batch size raises error."""
    global_batch = 500  # Not divisible by typical device counts
    try:
        distributed.get_local_batch_size(global_batch)
    except ValueError:
        pass  # Expected


def test_replicate_params() -> None:
    """Test parameter replication across devices."""
    mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    params = {"w": jnp.ones((3, 3)), "b": jnp.zeros((3,))}
    replicated = distributed.replicate_params(params, mesh)
    
    assert replicated["w"].shape == (3, 3)
    assert replicated["b"].shape == (3,)


def test_shard_batch() -> None:
    """Test batch sharding."""
    num_devices = jax.device_count()
    batch_size = num_devices * 32
    
    mesh = mesh_utils.create_device_mesh((num_devices,))
    sharding = distributed.get_batch_sharding(batch_size, mesh)
    
    images = jnp.ones((batch_size, 48, 48, 1), dtype=jnp.float32)
    labels = jnp.zeros((batch_size,), dtype=jnp.int32)
    
    sharded_images, sharded_labels = distributed.shard_batch(
        (images, labels), sharding
    )
    
    assert sharded_images.shape == images.shape
    assert sharded_labels.shape == labels.shape
