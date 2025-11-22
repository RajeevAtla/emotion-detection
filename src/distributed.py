"""Multi-GPU distributed training utilities for JAX NNX models."""

from __future__ import annotations

import os
from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax import nnx
import optax

PyTree = any


def setup_distributed_training() -> Tuple[Mesh, int, int]:
    """Initialize JAX distributed training environment.
    
    Returns:
        Tuple[Mesh, int, int]: Device mesh, num_processes, process_id
        
    Raises:
        RuntimeError: If distributed initialization fails.
    """
    try:
        if jax.device_count() > 1 and "NCCL_DEBUG" not in os.environ:
            os.environ.setdefault("NCCL_DEBUG", "INFO")
        
        num_processes = jax.process_count()
        process_id = jax.process_index()
        num_local_devices = jax.local_device_count()
        total_devices = jax.device_count()
        
        print(f"Process {process_id}/{num_processes}")
        print(f"Local devices: {num_local_devices}")
        print(f"Total devices: {total_devices}")
        print(f"Devices: {jax.devices()}")
        
        mesh = mesh_utils.create_device_mesh((total_devices,))
        return mesh, num_processes, process_id
    except Exception as e:
        raise RuntimeError(f"Distributed setup failed: {e}") from e


def get_batch_sharding(
    batch_size: int,
    mesh: Mesh,
) -> NamedSharding:
    """Create sharding for batch data across devices.
    
    Args:
        batch_size: Global batch size.
        mesh: Device mesh.
        
    Returns:
        NamedSharding: Sharding spec for batch axis.
    """
    return NamedSharding(mesh, P("batch"))


def shard_batch(
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    sharding: NamedSharding,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shard a batch across devices using specified sharding.
    
    Args:
        batch: Tuple of (images, labels).
        sharding: Target sharding specification.
        
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Sharded batch.
    """
    images, labels = batch
    images = jax.device_put(images, sharding)
    labels_sharding = NamedSharding(sharding.mesh, P("batch"))
    labels = jax.device_put(labels, labels_sharding)
    return images, labels


def replicate_params(
    params: PyTree,
    mesh: Mesh,
) -> PyTree:
    """Replicate parameters across all devices.
    
    Args:
        params: Parameter pytree.
        mesh: Device mesh.
        
    Returns:
        PyTree: Replicated parameters.
    """
    sharding = NamedSharding(mesh, P())
    return jax.tree_map(
        lambda x: jax.device_put(x, sharding),
        params,
    )


def create_distributed_optimizer(
    config: object,
    lr_schedule: optax.Schedule,
    mask: Optional[PyTree] = None,
    mesh: Optional[Mesh] = None,
) -> optax.GradientTransformation:
    """Create optimizer with distributed synchronization support.
    
    Args:
        config: Training configuration.
        lr_schedule: Learning rate schedule.
        mask: Optional parameter mask for freezing.
        mesh: Device mesh for distributed training.
        
    Returns:
        optax.GradientTransformation: Optimizer transformation.
    """
    tx = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
    )
    
    if config.gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.gradient_accumulation_steps)
    
    if mask is not None:
        tx = optax.masked(tx, mask)
    
    if mesh is not None:
        tx = optax.apply_every(config.gradient_accumulation_steps)
    
    return tx


def sync_gradients_across_devices(
    grads: PyTree,
) -> PyTree:
    """Synchronize gradients across all devices using all-reduce.
    
    Args:
        grads: Gradient pytree.
        
    Returns:
        PyTree: Synchronized gradients averaged across devices.
    """
    num_devices = jax.device_count()
    if num_devices <= 1:
        return grads
    
    return jax.tree_map(
        lambda g: jax.lax.psum(g, "i") / num_devices if isinstance(g, jnp.ndarray) else g,
        grads,
    )


def get_global_batch_size(
    local_batch_size: int,
) -> int:
    """Calculate global batch size from local batch size.
    
    Args:
        local_batch_size: Batch size per device.
        
    Returns:
        int: Global batch size across all devices.
    """
    return local_batch_size * jax.device_count()


def get_local_batch_size(
    global_batch_size: int,
) -> int:
    """Calculate local batch size from global batch size.
    
    Args:
        global_batch_size: Total batch size across devices.
        
    Returns:
        int: Batch size per device.
    """
    num_devices = jax.device_count()
    if global_batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"device count {num_devices}"
        )
    return global_batch_size // num_devices
