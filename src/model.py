from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import jax.numpy as jnp
from flax import linen as nn
from flax import traverse_util
from flax.core import FrozenDict, freeze, unfreeze
import orbax.checkpoint as ocp

ModuleDef = Type[nn.Module]


@dataclass(frozen=True)
class ResNetConfig:
    """Configuration bundle describing a ResNet variant."""

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
    """Convenience factory returning standard ResNet-18/34 configs."""
    presets: Dict[int, Tuple[Tuple[int, ...], str]] = {
        18: ((2, 2, 2, 2), "basic"),
        34: ((3, 4, 6, 3), "basic"),
        50: ((3, 4, 6, 3), "bottleneck"),
    }
    if depth not in presets:
        raise ValueError(f"Unsupported ResNet depth {depth}; choose from {sorted(presets)}.")

    blocks_per_stage, block_key = presets[depth]
    block_cls = BasicBlock if block_key == "basic" else BottleneckBlock

    config = ResNetConfig(depth=depth, blocks_per_stage=blocks_per_stage, block=block_cls)
    return replace(config, **overrides)  # type: ignore[arg-type]


class ResidualBlock(nn.Module):
    """Abstract base for residual blocks."""

    features: int
    strides: Tuple[int, int] = (1, 1)
    dtype: jnp.dtype = jnp.float32
    norm: ModuleDef = nn.BatchNorm
    conv: ModuleDef = nn.Conv
    use_projection: bool = False

    def setup(self) -> None:
        raise NotImplementedError

    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        raise NotImplementedError


class BasicBlock(ResidualBlock):
    """Standard 3x3 + 3x3 residual block used by ResNet-18/34."""

    def setup(self) -> None:
        self.conv1 = self.conv(
            self.features,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=False,
            dtype=self.dtype,
            name="conv1",
        )
        self.bn1 = self.norm(momentum=0.9, epsilon=1e-5, dtype=self.dtype, name="bn1")
        self.conv2 = self.conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=False,
            dtype=self.dtype,
            name="conv2",
        )
        self.bn2 = self.norm(momentum=0.9, epsilon=1e-5, dtype=self.dtype, name="bn2")
        if self.use_projection or self.strides != (1, 1):
            self.proj_conv = self.conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=False,
                dtype=self.dtype,
                name="proj_conv",
            )
            self.proj_bn = self.norm(momentum=0.9, epsilon=1e-5, dtype=self.dtype, name="proj_bn")
        else:
            self.proj_conv = None
            self.proj_bn = None

    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        residual = x
        y = self.conv1(x)
        y = self.bn1(y, use_running_average=not train)
        y = nn.relu(y)

        y = self.conv2(y)
        y = self.bn2(y, use_running_average=not train)

        if self.proj_conv is not None and self.proj_bn is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_bn(residual, use_running_average=not train)
        else:
            if residual.shape[-1] != self.features:
                raise ValueError(
                    "Residual channel mismatch without projection; ensure use_projection=True."
                )

        return nn.relu(y + residual)


class BottleneckBlock(ResidualBlock):
    """1x1-3x3-1x1 bottleneck block used by deeper ResNets."""

    bottleneck_ratio: int = 4

    def setup(self) -> None:
        inner_features = self.features // self.bottleneck_ratio
        self.conv1 = self.conv(
            inner_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            dtype=self.dtype,
            name="conv1",
        )
        self.bn1 = self.norm(momentum=0.9, epsilon=1e-5, dtype=self.dtype, name="bn1")
        self.conv2 = self.conv(
            inner_features,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=False,
            dtype=self.dtype,
            name="conv2",
        )
        self.bn2 = self.norm(momentum=0.9, epsilon=1e-5, dtype=self.dtype, name="bn2")
        self.conv3 = self.conv(
            self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            dtype=self.dtype,
            name="conv3",
        )
        self.bn3 = self.norm(momentum=0.9, epsilon=1e-5, dtype=self.dtype, name="bn3")
        if self.use_projection or self.strides != (1, 1):
            self.proj_conv = self.conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=False,
                dtype=self.dtype,
                name="proj_conv",
            )
            self.proj_bn = self.norm(momentum=0.9, epsilon=1e-5, dtype=self.dtype, name="proj_bn")
        else:
            self.proj_conv = None
            self.proj_bn = None

    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        residual = x
        y = self.conv1(x)
        y = self.bn1(y, use_running_average=not train)
        y = nn.relu(y)

        y = self.conv2(y)
        y = self.bn2(y, use_running_average=not train)
        y = nn.relu(y)

        y = self.conv3(y)
        y = self.bn3(y, use_running_average=not train)

        if self.proj_conv is not None and self.proj_bn is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_bn(residual, use_running_average=not train)
        else:
            if residual.shape[-1] != self.features:
                raise ValueError(
                    "Residual channel mismatch without projection; ensure use_projection=True."
                )

        return nn.relu(y + residual)


class ResNet(nn.Module):
    """Flax implementation of a CIFAR-style ResNet."""

    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32
    norm: ModuleDef = nn.BatchNorm
    conv: ModuleDef = nn.Conv
    include_top: bool = True
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool = True,
        return_features: bool = False,
    ) -> jnp.ndarray | Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        cfg = self.config
        if cfg.input_projection_channels is not None:
            x = self.conv(
                cfg.input_projection_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=False,
                dtype=self.dtype,
                name="input_projection",
            )(x)
        elif x.shape[-1] != cfg.input_channels:
            raise ValueError(
                f"Expected input with {cfg.input_channels} channels but received {x.shape[-1]}."
            )

        stem_width = cfg.stem_width * cfg.width_multiplier
        x = self.conv(
            stem_width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            dtype=self.dtype,
            name="stem_conv",
        )(x)
        x = self.norm(
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            name="stem_bn",
        )(x, use_running_average=not train)
        x = nn.relu(x)

        features: Dict[str, jnp.ndarray] = {"stem": x}
        for stage_index, (width, num_blocks) in enumerate(
            zip(cfg.stage_widths, cfg.blocks_per_stage)
        ):
            stage_width = width * cfg.width_multiplier
            for block_index in range(num_blocks):
                strides = (1, 1)
                use_projection = False
                if stage_index > 0 and block_index == 0:
                    strides = (2, 2)
                    use_projection = True
                elif block_index == 0 and x.shape[-1] != stage_width:
                    use_projection = True

                block = cfg.block(
                    stage_width,
                    strides=strides,
                    dtype=self.dtype,
                    norm=self.norm,
                    conv=self.conv,
                    use_projection=use_projection,
                    name=f"stage{stage_index+1}_block{block_index+1}",
                )
                x = block(x, train=train)

            features[f"stage{stage_index+1}"] = x

        x = jnp.mean(x, axis=(1, 2))
        features["pooled"] = x

        if self.include_top:
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train, name="dropout")(x)
            x = nn.Dense(cfg.num_classes, dtype=self.dtype, name="classifier")(x)
            features["logits"] = x

        if return_features:
            return x, features
        return x


def create_resnet(
    depth: int = 34,
    *,
    num_classes: int = 7,
    input_channels: int = 1,
    width_multiplier: int = 1,
    include_top: bool = True,
    input_projection_channels: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    frozen_stages: Tuple[int, ...] = (),
    freeze_stem: bool = False,
    freeze_classifier: bool = False,
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
    return ResNet(config=config, include_top=include_top)


def build_finetune_mask(
    params: FrozenDict[str, Any] | Dict[str, Any],
    *,
    config: ResNetConfig,
    freeze_classifier: Optional[bool] = None,
) -> FrozenDict[str, Any]:
    """Return a boolean mask marking which parameters remain trainable."""
    if freeze_classifier is None:
        freeze_classifier = config.freeze_classifier

    has_params_container = isinstance(params, (dict, FrozenDict)) and "params" in params
    target_tree = params["params"] if has_params_container else params
    if isinstance(target_tree, FrozenDict):
        target_tree_unfrozen = unfreeze(target_tree)
    else:
        target_tree_unfrozen = target_tree

    flat = traverse_util.flatten_dict(target_tree_unfrozen)

    frozen_stages = {f"stage{stage}_" for stage in config.frozen_stages}
    mask_flat: Dict[Tuple[str, ...], bool] = {}
    for path, value in flat.items():
        top_level = path[0] if isinstance(path, tuple) and path else path
        trainable = True

        if config.freeze_stem and (
            isinstance(top_level, str) and (top_level.startswith("stem_") or top_level == "input_projection")
        ):
            trainable = False

        if isinstance(top_level, str) and any(top_level.startswith(prefix) for prefix in frozen_stages):
            trainable = False

        if freeze_classifier and isinstance(top_level, str) and top_level.startswith("classifier"):
            trainable = False

        mask_flat[path] = trainable

    mask_tree = traverse_util.unflatten_dict(mask_flat)
    mask_tree = freeze(mask_tree)
    if has_params_container:
        return freeze({"params": mask_tree})  # type: ignore[return-value]
    return mask_tree  # type: ignore[return-value]


def maybe_load_pretrained_params(
    params: FrozenDict[str, Any],
    *,
    config: ResNetConfig,
) -> FrozenDict[str, Any]:
    """Load parameters from a checkpoint when configured."""
    if config.checkpoint_path is None:
        return params

    checkpointer = ocp.PyTreeCheckpointer()
    target = unfreeze(params) if isinstance(params, FrozenDict) else params
    restored = checkpointer.restore(str(config.checkpoint_path), item=target)
    if isinstance(restored, FrozenDict):
        return restored
    return freeze(restored)
