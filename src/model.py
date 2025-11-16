"""Flax ResNet building blocks tailored for emotion detection tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx

PyTree = Union[jax.Array, jnp.ndarray, Mapping[str, "PyTree"], Sequence["PyTree"]]
BoolTree = Union[bool, Mapping[str, "BoolTree"], Sequence["BoolTree"]]


@dataclass(frozen=True)
class ResNetConfig:
    """Configuration bundle describing a ResNet variant.

    Attributes:
        depth: Depth indicator (e.g., 18 or 34).
        blocks_per_stage: Number of residual blocks per stage.
        block: Residual block class used to construct the network.
        num_classes: Number of output classes in the classifier head.
        stage_widths: Channel widths per residual stage.
        stem_width: Channel width of the initial convolutional stem.
        width_multiplier: Multiplier applied to stage widths.
        input_channels: Number of channels expected in the input tensor.
        input_projection_channels: Optional projection width when adapting inputs.
        checkpoint_path: Optional checkpoint to initialize parameters from.
        frozen_stages: Stages to freeze during fine-tuning.
        freeze_stem: Whether to freeze the stem parameters.
        freeze_classifier: Whether to freeze the classifier head.
    """

    depth: int
    blocks_per_stage: Tuple[int, ...]
    block: Type["ResidualBlock"]
    num_classes: int = 7
    stage_widths: Tuple[int, ...] = (64, 128, 256, 512)
    stem_width: int = 64
    width_multiplier: int = 1
    input_channels: int = 1
    input_projection_channels: Optional[int] = None
    checkpoint_path: Optional[Path] = None
    frozen_stages: Tuple[int, ...] = ()
    freeze_stem: bool = False
    freeze_classifier: bool = False


def resnet_config(depth: int, **overrides: object) -> ResNetConfig:
    """Construct a ``ResNetConfig`` for a canonical depth."""

    presets: dict[int, Tuple[Tuple[int, ...], str]] = {
        18: ((2, 2, 2, 2), "basic"),
        34: ((3, 4, 6, 3), "basic"),
        50: ((3, 4, 6, 3), "bottleneck"),
    }
    if depth not in presets:
        raise ValueError(
            f"Unsupported ResNet depth {depth}; choose from {sorted(presets)}."
        )

    blocks_per_stage, block_key = presets[depth]
    block_cls = BasicBlock if block_key == "basic" else BottleneckBlock
    config = ResNetConfig(
        depth=depth,
        blocks_per_stage=blocks_per_stage,
        block=block_cls,
    )
    return replace(config, **overrides)  # type: ignore[arg-type]


class ResidualBlock(nnx.Module):
    """Abstract base for residual blocks."""

    def __init__(
        self,
        in_features: int,
        features: int,
        *,
        strides: Tuple[int, int] = (1, 1),
        dtype: jnp.dtype = jnp.float32,
        use_projection: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_features = in_features
        self.features = features
        self.strides = strides
        self.dtype = dtype
        self.use_projection = use_projection

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the block to a tensor."""
        raise NotImplementedError


class BasicBlock(ResidualBlock):
    """Standard 3x3 + 3x3 residual block used by ResNet-18/34."""

    def __init__(
        self,
        in_features: int,
        features: int,
        *,
        strides: Tuple[int, int] = (1, 1),
        dtype: jnp.dtype = jnp.float32,
        use_projection: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            in_features,
            features,
            strides=strides,
            dtype=dtype,
            use_projection=use_projection,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_features,
            features,
            kernel_size=(3, 3),
            strides=strides,
            padding="SAME",
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs.fork(),
        )
        self.bn1 = nnx.BatchNorm(
            features,
            momentum=0.9,
            epsilon=1e-5,
            dtype=dtype,
            rngs=rngs.fork(),
        )
        self.conv2 = nnx.Conv(
            features,
            features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs.fork(),
        )
        self.bn2 = nnx.BatchNorm(
            features,
            momentum=0.9,
            epsilon=1e-5,
            dtype=dtype,
            rngs=rngs.fork(),
        )
        if self.use_projection or self.strides != (1, 1):
            self.proj_conv = nnx.Conv(
                in_features,
                features,
                kernel_size=(1, 1),
                strides=strides,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                rngs=rngs.fork(),
            )
            self.proj_bn = nnx.BatchNorm(
                features,
                momentum=0.9,
                epsilon=1e-5,
                dtype=dtype,
                rngs=rngs.fork(),
            )
        else:
            self.proj_conv = None
            self.proj_bn = None

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = jax.nn.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.proj_conv is not None and self.proj_bn is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_bn(residual)
        elif residual.shape[-1] != self.features:
            raise ValueError(
                "Residual channel mismatch without projection; set use_projection=True."
            )

        return jax.nn.relu(y + residual)


class BottleneckBlock(ResidualBlock):
    """1x1-3x3-1x1 bottleneck block used by deeper ResNets."""

    bottleneck_ratio: int = 4

    def __init__(
        self,
        in_features: int,
        features: int,
        *,
        strides: Tuple[int, int] = (1, 1),
        dtype: jnp.dtype = jnp.float32,
        use_projection: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            in_features,
            features,
            strides=strides,
            dtype=dtype,
            use_projection=use_projection,
            rngs=rngs,
        )
        inner_features = features // self.bottleneck_ratio
        self.conv1 = nnx.Conv(
            in_features,
            inner_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs.fork(),
        )
        self.bn1 = nnx.BatchNorm(
            inner_features,
            momentum=0.9,
            epsilon=1e-5,
            dtype=dtype,
            rngs=rngs.fork(),
        )
        self.conv2 = nnx.Conv(
            inner_features,
            inner_features,
            kernel_size=(3, 3),
            strides=strides,
            padding="SAME",
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs.fork(),
        )
        self.bn2 = nnx.BatchNorm(
            inner_features,
            momentum=0.9,
            epsilon=1e-5,
            dtype=dtype,
            rngs=rngs.fork(),
        )
        self.conv3 = nnx.Conv(
            inner_features,
            features,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs.fork(),
        )
        self.bn3 = nnx.BatchNorm(
            features,
            momentum=0.9,
            epsilon=1e-5,
            dtype=dtype,
            rngs=rngs.fork(),
        )
        if self.use_projection or self.strides != (1, 1):
            self.proj_conv = nnx.Conv(
                in_features,
                features,
                kernel_size=(1, 1),
                strides=strides,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                rngs=rngs.fork(),
            )
            self.proj_bn = nnx.BatchNorm(
                features,
                momentum=0.9,
                epsilon=1e-5,
                dtype=dtype,
                rngs=rngs.fork(),
            )
        else:
            self.proj_conv = None
            self.proj_bn = None

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = jax.nn.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = jax.nn.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.proj_conv is not None and self.proj_bn is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_bn(residual)
        elif residual.shape[-1] != self.features:
            raise ValueError(
                "Residual channel mismatch without projection; set use_projection=True."
            )

        return jax.nn.relu(y + residual)


class ResNet(nnx.Module):
    """Flax NNX implementation of a CIFAR-style ResNet."""

    def __init__(
        self,
        config: ResNetConfig,
        *,
        include_top: bool = True,
        dropout_rate: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dtype = dtype

        stem_in_features = config.input_channels
        if config.input_projection_channels is not None:
            self.input_projection = nnx.Conv(
                config.input_channels,
                config.input_projection_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                rngs=rngs.fork(),
            )
            stem_in_features = config.input_projection_channels
        else:
            self.input_projection = None

        stem_width = config.stem_width * config.width_multiplier
        self.stem_conv = nnx.Conv(
            stem_in_features,
            stem_width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs.fork(),
        )
        self.stem_bn = nnx.BatchNorm(
            stem_width,
            momentum=0.9,
            epsilon=1e-5,
            dtype=dtype,
            rngs=rngs.fork(),
        )

        self.stage_block_names: list[list[str]] = []
        current_channels = stem_width
        for stage_index, (width, num_blocks) in enumerate(
            zip(config.stage_widths, config.blocks_per_stage)
        ):
            stage_width = width * config.width_multiplier
            block_names: list[str] = []
            for block_index in range(num_blocks):
                strides = (1, 1)
                use_projection = False
                if stage_index > 0 and block_index == 0:
                    strides = (2, 2)
                    use_projection = True
                elif block_index == 0 and current_channels != stage_width:
                    use_projection = True

                block = config.block(
                    current_channels,
                    stage_width,
                    strides=strides,
                    dtype=dtype,
                    use_projection=use_projection,
                    rngs=rngs.fork(),
                )
                block_name = f"stage{stage_index + 1}_block{block_index + 1}"
                setattr(self, block_name, block)
                block_names.append(block_name)
                current_channels = stage_width
            self.stage_block_names.append(block_names)

        if include_top:
            self.classifier = nnx.Linear(
                current_channels,
                config.num_classes,
                dtype=dtype,
                rngs=rngs.fork(),
            )
            self.dropout = (
                nnx.Dropout(dropout_rate, rngs=rngs.fork())
                if dropout_rate > 0.0
                else None
            )
        else:
            self.classifier = None
            self.dropout = None

    def __call__(
        self,
        x: jax.Array,
        *,
        train: bool = True,
        return_features: bool = False,
    ) -> jax.Array | Tuple[jax.Array, dict[str, jax.Array]]:
        if train:
            self.train()
        else:
            self.eval()

        cfg = self.config
        if self.input_projection is not None:
            x = self.input_projection(x)
        elif x.shape[-1] != cfg.input_channels:
            raise ValueError(
                f"Expected input with {cfg.input_channels} channels but received {x.shape[-1]}."
            )

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = jax.nn.relu(x)

        features: dict[str, jax.Array] = {"stem": x}
        for stage_index, block_names in enumerate(self.stage_block_names):
            for block_name in block_names:
                block = getattr(self, block_name)
                x = block(x)
            features[f"stage{stage_index + 1}"] = x

        x = jnp.mean(x, axis=(1, 2))
        features["pooled"] = x

        if self.include_top and self.classifier is not None:
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.classifier(x)
            features["logits"] = x

        if return_features:
            return x, features
        return x


def create_resnet(
    depth: int = 34,
    *,
    rngs: nnx.Rngs,
    num_classes: int = 7,
    input_channels: int = 1,
    width_multiplier: int = 1,
    include_top: bool = True,
    input_projection_channels: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    frozen_stages: Tuple[int, ...] = (),
    freeze_stem: bool = False,
    freeze_classifier: bool = False,
    dropout_rate: float = 0.0,
) -> ResNet:
    """Factory returning a configured ResNet model."""

    blocks_per_stage = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
    }
    if depth not in blocks_per_stage:
        raise ValueError(f"Unsupported depth {depth}; choose 18 or 34.")

    config = ResNetConfig(
        depth=depth,
        blocks_per_stage=blocks_per_stage[depth],
        block=BasicBlock,
        num_classes=num_classes,
        stage_widths=(64, 128, 256, 512),
        stem_width=64,
        width_multiplier=width_multiplier,
        input_channels=input_channels,
        input_projection_channels=input_projection_channels,
        checkpoint_path=checkpoint_path,
        frozen_stages=frozen_stages,
        freeze_stem=freeze_stem,
        freeze_classifier=freeze_classifier,
    )
    return ResNet(
        config=config,
        include_top=include_top,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )


def _path_to_names(path: tuple[object, ...]) -> list[str]:
    names: list[str] = []
    for entry in path:
        key = getattr(entry, "key", None)
        if isinstance(key, str) and key != ".value":
            names.append(key)
    return names


def build_finetune_mask(
    params: PyTree,
    *,
    config: ResNetConfig,
    freeze_classifier: Optional[bool] = None,
) -> BoolTree:
    """Return a boolean mask marking which parameters remain trainable."""

    if freeze_classifier is None:
        freeze_classifier = config.freeze_classifier

    leaves_with_path, treedef = jtu.tree_flatten_with_path(params)

    mask_leaves: list[bool] = []
    frozen_prefixes = {f"stage{stage}_" for stage in config.frozen_stages}

    for path, _ in leaves_with_path:
        names = _path_to_names(path)
        top_level = names[0] if names else ""
        trainable = True

        if config.freeze_stem and (
            top_level.startswith("stem_") or top_level == "input_projection"
        ):
            trainable = False

        if any(top_level.startswith(prefix) for prefix in frozen_prefixes):
            trainable = False

        if freeze_classifier and top_level.startswith("classifier"):
            trainable = False

        mask_leaves.append(trainable)

    return jtu.tree_unflatten(treedef, mask_leaves)


def maybe_load_pretrained_params(*args, **kwargs) -> None:
    """Placeholder until Agent 3 wires Orbax checkpoint loading for NNX."""

    raise NotImplementedError(
        "NNX migration replaces FrozenDict checkpoints; use the forthcoming "
        "Agent 3 checkpoint utilities instead."
    )
