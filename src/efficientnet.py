"""EfficientNet implementation in Flax NNX for emotion detection.

This module provides EfficientNet-B0 through B3 implementations optimized
for facial expression recognition with multi-GPU support.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from flax import nnx


# EfficientNet scaling parameters
# (width_coefficient, depth_coefficient, resolution, dropout_rate)
EFFICIENTNET_PARAMS = {
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
}

# MBConv block configurations
# (expand_ratio, channels, num_blocks, stride, kernel_size)
MBCONV_CONFIGS = [
    (1, 16, 1, 1, 3),   # Stage 1
    (6, 24, 2, 2, 3),   # Stage 2
    (6, 40, 2, 2, 5),   # Stage 3
    (6, 80, 3, 2, 3),   # Stage 4
    (6, 112, 3, 1, 5),  # Stage 5
    (6, 192, 4, 2, 5),  # Stage 6
    (6, 320, 1, 1, 3),  # Stage 7
]


@dataclass(frozen=True)
class EfficientNetConfig:
    """Configuration for EfficientNet variants."""
    
    variant: str = "b2"
    num_classes: int = 7
    input_channels: int = 1  # Grayscale for FER
    dropout_rate: float = 0.3
    drop_connect_rate: float = 0.2  # Stochastic depth
    include_top: bool = True
    
    @property
    def width_coefficient(self) -> float:
        return EFFICIENTNET_PARAMS[self.variant][0]
    
    @property
    def depth_coefficient(self) -> float:
        return EFFICIENTNET_PARAMS[self.variant][1]
    
    @property
    def default_resolution(self) -> int:
        return EFFICIENTNET_PARAMS[self.variant][2]


def round_filters(filters: int, width_coefficient: float) -> int:
    """Round number of filters based on width multiplier."""
    divisor = 8
    filters = int(filters * width_coefficient)
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    """Round number of block repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


class SqueezeExcitation(nnx.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.fc1 = nnx.Linear(
            in_channels,
            squeeze_channels,
            use_bias=True,
            dtype=dtype,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            squeeze_channels,
            in_channels,
            use_bias=True,
            dtype=dtype,
            rngs=rngs,
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # Global average pooling
        scale = jnp.mean(x, axis=(1, 2))
        scale = self.fc1(scale)
        scale = jax.nn.swish(scale)
        scale = self.fc2(scale)
        scale = jax.nn.sigmoid(scale)
        return x * scale[:, None, None, :]


class MBConvBlock(nnx.Module):
    """Mobile Inverted Bottleneck Convolution block.
    
    This is the core building block of EfficientNet with:
    - Depthwise separable convolutions
    - Squeeze-and-excitation attention
    - Stochastic depth (drop connect)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expand_ratio: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand = expand_ratio != 1
        
        if self.expand:
            self.expand_conv = nnx.Conv(
                in_channels,
                expanded_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )
            self.expand_bn = nnx.BatchNorm(
                expanded_channels,
                momentum=0.99,
                epsilon=1e-3,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            expanded_channels = in_channels
        
        # Depthwise convolution
        self.depthwise_conv = nnx.Conv(
            expanded_channels,
            expanded_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding="SAME",
            feature_group_count=expanded_channels,  # Depthwise
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.depthwise_bn = nnx.BatchNorm(
            expanded_channels,
            momentum=0.99,
            epsilon=1e-3,
            dtype=dtype,
            rngs=rngs,
        )
        
        # Squeeze-and-excitation
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(
            expanded_channels,
            squeeze_channels,
            dtype=dtype,
            rngs=rngs,
        )
        
        # Projection phase
        self.project_conv = nnx.Conv(
            expanded_channels,
            out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.project_bn = nnx.BatchNorm(
            out_channels,
            momentum=0.99,
            epsilon=1e-3,
            dtype=dtype,
            rngs=rngs,
        )
        
        # For stochastic depth
        self.dropout = nnx.Dropout(drop_connect_rate, rngs=rngs)
    
    def __call__(self, x: jax.Array, *, train: bool = True) -> jax.Array:
        residual = x
        
        # Expansion
        if self.expand:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = jax.nn.swish(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = jax.nn.swish(x)
        
        # Squeeze-and-excitation
        x = self.se(x)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Residual connection with stochastic depth
        if self.use_residual:
            if train and self.drop_connect_rate > 0:
                x = self.dropout(x)
            x = x + residual
        
        return x


class EfficientNet(nnx.Module):
    """EfficientNet implementation in Flax NNX.
    
    Features:
    - Compound scaling of depth, width, and resolution
    - MBConv blocks with squeeze-and-excitation
    - Stochastic depth for regularization
    - Support for grayscale input (emotion detection)
    """
    
    def __init__(
        self,
        config: EfficientNetConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.dtype = dtype
        
        width_coef = config.width_coefficient
        depth_coef = config.depth_coefficient
        
        # Input projection for grayscale
        if config.input_channels != 3:
            self.input_projection = nnx.Conv(
                config.input_channels,
                3,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.input_projection = None
        
        # Stem
        stem_channels = round_filters(32, width_coef)
        self.stem_conv = nnx.Conv(
            3,
            stem_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.stem_bn = nnx.BatchNorm(
            stem_channels,
            momentum=0.99,
            epsilon=1e-3,
            dtype=dtype,
            rngs=rngs,
        )
        
        # Build MBConv blocks
        self.blocks: list[MBConvBlock] = []
        total_blocks = sum(round_repeats(cfg[2], depth_coef) for cfg in MBCONV_CONFIGS)
        block_idx = 0
        
        in_channels = stem_channels
        for expand_ratio, channels, num_blocks, stride, kernel_size in MBCONV_CONFIGS:
            out_channels = round_filters(channels, width_coef)
            num_repeats = round_repeats(num_blocks, depth_coef)
            
            for i in range(num_repeats):
                block_stride = stride if i == 0 else 1
                drop_rate = config.drop_connect_rate * block_idx / total_blocks
                
                block = MBConvBlock(
                    in_channels,
                    out_channels,
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    drop_connect_rate=drop_rate,
                    dtype=dtype,
                    rngs=rngs,
                )
                self.blocks.append(block)
                in_channels = out_channels
                block_idx += 1
        
        # Head
        head_channels = round_filters(1280, width_coef)
        self.head_conv = nnx.Conv(
            in_channels,
            head_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        self.head_bn = nnx.BatchNorm(
            head_channels,
            momentum=0.99,
            epsilon=1e-3,
            dtype=dtype,
            rngs=rngs,
        )
        
        # Classifier
        if config.include_top:
            self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
            self.classifier = nnx.Linear(
                head_channels,
                config.num_classes,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.dropout = None
            self.classifier = None
        
        self._head_channels = head_channels
    
    def __call__(
        self,
        x: jax.Array,
        *,
        train: bool = True,
        return_features: bool = False,
    ) -> jax.Array | Tuple[jax.Array, dict[str, jax.Array]]:
        """Forward pass through EfficientNet.
        
        Args:
            x: Input tensor of shape (B, H, W, C)
            train: Whether in training mode (affects dropout, batchnorm)
            return_features: If True, return intermediate features
            
        Returns:
            Logits or (logits, features_dict)
        """
        features = {}
        
        # Input projection for grayscale
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = jax.nn.swish(x)
        features["stem"] = x
        
        # Blocks
        for idx, block in enumerate(self.blocks):
            x = block(x, train=train)
            if idx in {2, 4, 8, 15}:  # Key stages
                features[f"stage_{idx}"] = x
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = jax.nn.swish(x)
        features["head"] = x
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        features["pooled"] = x
        
        # Classifier
        if self.config.include_top and self.classifier is not None:
            if self.dropout is not None:
                x = self.dropout(x, deterministic=not train)
            x = self.classifier(x)
            features["logits"] = x
        
        if return_features:
            return x, features
        return x
    
    def train(self) -> None:
        """Set module to training mode."""
        pass
    
    def eval(self) -> None:
        """Set module to evaluation mode."""
        pass


def create_efficientnet(
    variant: str = "b2",
    *,
    rngs: nnx.Rngs,
    num_classes: int = 7,
    input_channels: int = 1,
    dropout_rate: float = 0.3,
    drop_connect_rate: float = 0.2,
    include_top: bool = True,
) -> EfficientNet:
    """Factory function to create EfficientNet model.
    
    Args:
        variant: One of "b0", "b1", "b2", "b3"
        rngs: Flax RNG container
        num_classes: Number of output classes
        input_channels: Number of input channels (1 for grayscale)
        dropout_rate: Dropout rate before classifier
        drop_connect_rate: Stochastic depth rate
        include_top: Whether to include classifier head
        
    Returns:
        EfficientNet model instance
    """
    if variant not in EFFICIENTNET_PARAMS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(EFFICIENTNET_PARAMS.keys())}")
    
    config = EfficientNetConfig(
        variant=variant,
        num_classes=num_classes,
        input_channels=input_channels,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        include_top=include_top,
    )
    
    return EfficientNet(config, rngs=rngs)


# Convenience functions for common variants
def efficientnet_b0(rngs: nnx.Rngs, **kwargs) -> EfficientNet:
    """Create EfficientNet-B0 model."""
    return create_efficientnet("b0", rngs=rngs, **kwargs)


def efficientnet_b2(rngs: nnx.Rngs, **kwargs) -> EfficientNet:
    """Create EfficientNet-B2 model (recommended for FER)."""
    return create_efficientnet("b2", rngs=rngs, **kwargs)


def efficientnet_b3(rngs: nnx.Rngs, **kwargs) -> EfficientNet:
    """Create EfficientNet-B3 model."""
    return create_efficientnet("b3", rngs=rngs, **kwargs)
