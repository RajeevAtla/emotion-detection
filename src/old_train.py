"""Training loops, utilities, and evaluation helpers for emotion detection.

Optimized version with:
- Fixed syntax errors
- JIT-compiled training steps
- Clear epoch logging
- Proper multi-GPU support
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeAlias, Union, cast

import jax
import jax.numpy as jnp
import metrax as mx
import optax
import numpy as np
from flax import nnx
from tensorboardX import SummaryWriter

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src import checkpointing
from src.data import DataModuleConfig, EmotionDataModule
from src.model import (
    ResNet,
    build_finetune_mask,
    create_resnet,
    maybe_load_pretrained_params,
)

BoolTree: TypeAlias = Union[bool, Mapping[str, "BoolTree"], Sequence["BoolTree"]]
TrainBatch = Tuple[jnp.ndarray, jnp.ndarray]
TrainStepFn = Callable[
    ["TrainState", TrainBatch],
    Tuple["TrainState", dict[str, jnp.ndarray]],
]
EvalStepFn = Callable[
    ["TrainState", TrainBatch], Tuple[dict[str, jnp.ndarray], jnp.ndarray]
]


def _cast_precision(images: jnp.ndarray, *, use_mixed_precision: bool) -> jnp.ndarray:
    """Cast input images to the configured precision."""
    return images.astype(jnp.float16 if use_mixed_precision else jnp.float32)


TrainingHistory = dict[str, list[float]]
TrainingSummary = dict[str, Union[float, int, None, str, TrainingHistory]]


@dataclass
class TrainingConfig:
    """Collection of knobs driving the training loop."""

    data: DataModuleConfig
    output_dir: Path
    model_depth: int = 34
    width_multiplier: int = 1
    dropout_rate: float = 0.0
    num_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    label_smoothing: float = 0.0
    seed: int = 0
    log_every: int = 50
    log_to_console: bool = True  # Changed default to True
    checkpoint_every: int = 5
    max_checkpoints: int = 3
    use_mixed_precision: bool = False
    patience: Optional[int] = None
    freeze_stem: bool = False
    freeze_classifier: bool = False
    frozen_stages: Tuple[int, ...] = ()
    pretrained_checkpoint: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None
    num_gpus: int = 1
    distributed: bool = False

    def __post_init__(self) -> None:
        """Resolve path-like attributes immediately after initialization."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.pretrained_checkpoint is not None:
            self.pretrained_checkpoint = Path(self.pretrained_checkpoint)
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(self.resume_checkpoint)


@dataclass
class TrainState:
    """Container describing the mutable pieces of training."""

    model: ResNet
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    rngs: nnx.Rngs
    dynamic_scale: Optional[object] = None


def create_learning_rate_schedule(
    config: TrainingConfig,
    steps_per_epoch: int,
    global_batch_size: Optional[int] = None,
) -> optax.Schedule:
    """Build a learning rate schedule with warmup followed by cosine decay."""
    warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)
    total_steps = max(1, config.num_epochs * steps_per_epoch)
    if warmup_steps >= total_steps:
        warmup_steps = max(1, total_steps - 1)
    total_steps = max(warmup_steps + 1, total_steps)
    decay_steps = max(1, total_steps - warmup_steps)

    peak_lr = config.learning_rate
    # Scale learning rate with batch size (linear scaling rule)
    if global_batch_size is not None and global_batch_size > 128:
        scale_factor = global_batch_size / 128.0
        peak_lr = peak_lr * math.sqrt(scale_factor)  # Square root scaling
        
    min_lr = config.min_learning_rate
    warmup_steps_f = float(warmup_steps)
    decay_steps_f = float(decay_steps)

    def schedule(step: int) -> jnp.ndarray:
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup_progress = jnp.minimum(1.0, step_f / warmup_steps_f)
        cosine_progress = jnp.clip(
            (step_f - warmup_steps_f) / decay_steps_f, 0.0, 1.0
        )
        cosine_value = 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_progress))
        decay_lr = min_lr + (peak_lr - min_lr) * cosine_value
        warmup_lr = peak_lr * warmup_progress
        return jnp.where(step_f < warmup_steps_f, warmup_lr, decay_lr)

    return schedule


def create_optimizer(
    config: TrainingConfig,
    lr_schedule: optax.Schedule,
    mask: Optional[BoolTree] = None,
) -> optax.GradientTransformation:
    """Create the optimizer transformation for training."""
    tx: optax.GradientTransformation = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
    )
    if config.gradient_accumulation_steps > 1:
        tx = cast(
            optax.GradientTransformation,
            optax.MultiSteps(
                tx, every_k_schedule=config.gradient_accumulation_steps
            ),
        )
    if mask is not None:
        tx = cast(optax.GradientTransformation, optax.masked(tx, mask))
    return tx


def initialize_nnx_model(
    config: TrainingConfig,
    *,
    num_classes: int,
    seed: Optional[int] = None,
    include_top: bool = True,
) -> tuple[ResNet, nnx.Rngs]:
    """Instantiate an NNX ResNet alongside its RNG container."""
    rng_seed = config.seed if seed is None else seed
    rngs = nnx.Rngs(rng_seed)
    model = create_resnet(
        depth=config.model_depth,
        rngs=rngs.fork(),
        num_classes=num_classes,
        input_channels=getattr(config.data, "input_channels", 1),
        width_multiplier=config.width_multiplier,
        include_top=include_top,
        input_projection_channels=None,
        checkpoint_path=config.pretrained_checkpoint,
        frozen_stages=config.frozen_stages,
        freeze_stem=config.freeze_stem,
        freeze_classifier=config.freeze_classifier,
        dropout_rate=config.dropout_rate,
    )
    maybe_load_pretrained_params(
        model,
        checkpoint_path=config.pretrained_checkpoint,
    )
    return model, rngs


def create_train_state(
    model: ResNet,
    config: TrainingConfig,
    lr_schedule: optax.Schedule,
    *,
    rngs: Optional[nnx.Rngs] = None,
) -> TrainState:
    """Return a populated ``TrainState`` ready for the training loop."""
    base_rngs = rngs or nnx.Rngs(config.seed)
    params_tree = nnx.to_pure_dict(nnx.state(model, nnx.Param))

    mask_tree: Optional[BoolTree] = None
    if config.freeze_stem or config.freeze_classifier or config.frozen_stages:
        mask_tree_result = build_finetune_mask(
            params_tree,
            config=replace(
                model.config,
                freeze_stem=config.freeze_stem,
                freeze_classifier=config.freeze_classifier,
                frozen_stages=config.frozen_stages,
            ),
        )
        mask_tree = mask_tree_result

    optimizer = create_optimizer(config, lr_schedule, mask_tree)
    opt_state = optimizer.init(params_tree)
    return TrainState(
        model=model,
        tx=optimizer,
        opt_state=opt_state,
        rngs=base_rngs,
        dynamic_scale=None,
    )


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    label_smoothing: float = 0.0,
    class_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute label-smoothed cross-entropy with optional class weighting."""
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    if label_smoothing > 0.0:
        smoothing = jnp.asarray(label_smoothing, dtype=logits.dtype)
        one_hot = one_hot * (1.0 - smoothing) + smoothing / num_classes
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    if class_weights is not None:
        weights = jnp.take(class_weights, labels)
        loss = loss * weights
    return jnp.mean(loss)


def compute_confusion_matrix(
    preds: jnp.ndarray, labels: jnp.ndarray, num_classes: int
) -> np.ndarray:
    """Return confusion matrix with shape ``(num_classes, num_classes)``."""
    preds_np = np.asarray(preds)
    labels_np = np.asarray(labels)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for label, pred in zip(labels_np, preds_np):
        cm[int(label), int(pred)] += 1
    return cm


def format_confusion_matrix(cm: np.ndarray, class_names: Iterable[str]) -> str:
    """Format confusion matrix as a Markdown table string."""
    header = [" "] + list(class_names)
    lines = [" | ".join(header)]
    lines.append(" | ".join(["---"] * len(header)))
    for idx, row in enumerate(cm):
        row_vals = [str(header[idx + 1])] + [str(int(val)) for val in row]
        lines.append(" | ".join(row_vals))
    return "\n".join(lines)


def compute_f1_metrics(cm: np.ndarray) -> tuple[float, float, list[float]]:
    """Compute micro and macro F1 along with per-class F1 scores."""
    if cm.size == 0:
        return float("nan"), float("nan"), []

    num_classes = cm.shape[0]
    per_class_scores = np.full(num_classes, np.nan, dtype=np.float32)
    for idx in range(num_classes):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            continue
        precision_den = tp + fp
        recall_den = tp + fn
        precision = tp / precision_den if precision_den > 0.0 else 0.0
        recall = tp / recall_den if recall_den > 0.0 else 0.0
        if precision + recall == 0.0:
            per_class_scores[idx] = 0.0
        else:
            per_class_scores[idx] = (
                2.0 * precision * recall / (precision + recall)
            )

    if np.all(np.isnan(per_class_scores)):
        macro_f1 = float("nan")
    else:
        macro_f1 = float(np.nanmean(per_class_scores))

    total = float(cm.sum())
    if total <= 0.0:
        micro_f1 = float("nan")
    else:
        micro_f1 = float(np.trace(cm) / total)

    return micro_f1, macro_f1, per_class_scores.tolist()


def build_train_step(
    config: TrainingConfig,
    *,
    class_weights: Optional[jnp.ndarray],
) -> TrainStepFn:
    """Create the stateful training function for NNX models."""

    def loss_with_metrics(
        model: ResNet, images: jnp.ndarray, labels: jnp.ndarray
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        model.train()
        logits = cast(jnp.ndarray, model(images))
        loss = cross_entropy_loss(
            logits,
            labels,
            label_smoothing=config.label_smoothing,
            class_weights=class_weights,
        )
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == labels)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return loss, metrics

    grad_fn = nnx.value_and_grad(
        loss_with_metrics,
        argnums=nnx.DiffState(0, nnx.Param),
        has_aux=True,
    )

    def train_step(
        state: TrainState, batch: TrainBatch
    ) -> Tuple[TrainState, dict[str, jnp.ndarray]]:
        images, labels = batch
        images = _cast_precision(
            images, use_mixed_precision=config.use_mixed_precision
        )
        (_, metrics), grads = grad_fn(state.model, images, labels)
        params_tree = nnx.to_pure_dict(nnx.state(state.model, nnx.Param))
        grads_tree = nnx.to_pure_dict(nnx.state(grads, nnx.Param))
        updates, new_opt_state = state.tx.update(
            grads_tree, state.opt_state, params_tree
        )
        new_params = optax.apply_updates(params_tree, updates)
        nnx.update(state.model, new_params)
        state.opt_state = new_opt_state
        return state, metrics

    return train_step


def build_eval_step(config: TrainingConfig) -> EvalStepFn:
    """Create the evaluation step for NNX models."""

    def eval_step(
        state: TrainState, batch: TrainBatch
    ) -> Tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        images, labels = batch
        images = _cast_precision(
            images, use_mixed_precision=config.use_mixed_precision
        )
        state.model.eval()
        logits = cast(jnp.ndarray, state.model(images))
        loss = cross_entropy_loss(
            logits,
            labels,
            label_smoothing=config.label_smoothing,
        )
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == labels)
        state.model.train()
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return metrics, preds

    return eval_step


def save_checkpoint(
    state: TrainState, config: TrainingConfig, epoch: int
) -> Path:
    """Persist an ``NNXTrainState`` to disk."""
    payload = {
        "model": checkpointing.nnx_state(state.model),
        "opt_state": state.opt_state,
        "rngs": checkpointing.nnx_state(state.rngs),
        "dynamic_scale": state.dynamic_scale,
    }
    layout = checkpointing.CheckpointLayout(
        directory=config.output_dir / "checkpoints",
        max_checkpoints=config.max_checkpoints,
    )
    return checkpointing.save_payload(payload, layout=layout, epoch=epoch)


def maybe_restore_checkpoint(
    config: TrainingConfig, state: TrainState
) -> Optional[TrainState]:
    """Restore an NNX checkpoint if ``resume_checkpoint`` is provided."""
    if config.resume_checkpoint is None:
        return None
    template = {
        "model": checkpointing.nnx_state(state.model),
        "opt_state": state.opt_state,
        "rngs": checkpointing.nnx_state(state.rngs),
        "dynamic_scale": state.dynamic_scale,
    }
    restored = checkpointing.restore_payload(
        config.resume_checkpoint, template=template
    )
    if restored is None:
        return None
    model_state = restored.get("model", template["model"])
    opt_state = restored.get("opt_state", template["opt_state"])
    rng_state = restored.get("rngs", template["rngs"])
    checkpointing.apply_nnx_state(state.model, model_state)
    checkpointing.apply_nnx_state_to_object(state.rngs, rng_state)
    dynamic_scale = restored.get("dynamic_scale", state.dynamic_scale)
    return replace(
        state,
        opt_state=opt_state,
        dynamic_scale=dynamic_scale,
    )


def predict_batches(
    state: TrainState,
    batches: Iterable[Tuple[jnp.ndarray, jnp.ndarray]],
    config: TrainingConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate predictions using the NNX model."""
    preds: list[jnp.ndarray] = []
    labels: list[jnp.ndarray] = []
    state.model.eval()
    for images, batch_labels in batches:
        batch_images = _cast_precision(
            images, use_mixed_precision=config.use_mixed_precision
        )
        logits = state.model(batch_images)
        logits = cast(jnp.ndarray, logits)
        preds.append(jnp.argmax(logits, axis=-1))
        labels.append(batch_labels)
    state.model.train()
    if not preds:
        return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32)
    return jnp.concatenate(preds, axis=0), jnp.concatenate(labels, axis=0)


def print_header(text: str, char: str = "=", width: int = 70) -> None:
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}", flush=True)


def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_f1: float,
    val_macro_f1: float,
    epoch_time: float,
    best_val_loss: float,
    is_best: bool,
) -> None:
    """Print a formatted epoch summary."""
    status = "★ BEST" if is_best else ""
    print(f"\n{'─' * 70}")
    print(f"│ Epoch {epoch:3d}/{num_epochs} │ Time: {epoch_time:.1f}s │ {status}")
    print(f"{'─' * 70}")
    print(f"│ Train Loss: {train_loss:.4f}  │  Train Acc: {train_acc*100:.2f}%")
    print(f"│ Val Loss:   {val_loss:.4f}  │  Val Acc:   {val_acc*100:.2f}%")
    print(f"│ Val F1:     {val_f1:.4f}  │  Macro F1:  {val_macro_f1:.4f}")
    print(f"│ Best Loss:  {best_val_loss:.4f}")
    print(f"{'─' * 70}", flush=True)


def train_and_evaluate(config: TrainingConfig, mesh=None) -> TrainingSummary:
    """Run the training pipeline using the NNX-native state."""
    
    # =========================================================================
    # Setup
    # =========================================================================
    num_devices = jax.device_count()
    print_header("EMOTION DETECTION TRAINING")
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {num_devices}")
    
    if num_devices > 1 and mesh is None:
        mesh = mesh_utils.create_device_mesh((num_devices,))
        print(f"Created device mesh: {mesh}")

    # Calculate batch sizes
    global_batch_size = config.batch_size * num_devices
    print(f"Local batch size: {config.batch_size}")
    print(f"Global batch size: {global_batch_size}")

    # =========================================================================
    # Data Setup
    # =========================================================================
    print_header("DATA SETUP", char="-")
    
    rng = jax.random.PRNGKey(config.seed)
    data_config = replace(config.data)
    data_module = EmotionDataModule(data_config)
    data_module.setup()
    
    class_weights = data_module.class_weights
    split_counts = data_module.split_counts()
    train_counts = split_counts["train"]
    val_counts = split_counts["val"]
    test_counts = split_counts["test"]
    
    num_train_examples = sum(train_counts.values())
    num_val_examples = sum(val_counts.values())
    num_test_examples = sum(test_counts.values())
    steps_per_epoch = max(1, num_train_examples // config.batch_size)
    
    print(f"Training samples:   {num_train_examples}")
    print(f"Validation samples: {num_val_examples}")
    print(f"Test samples:       {num_test_examples}")
    print(f"Steps per epoch:    {steps_per_epoch}")
    print(f"Classes: {list(train_counts.keys())}")
    print(f"Class distribution (train): {train_counts}")

    # =========================================================================
    # Model Setup
    # =========================================================================
    print_header("MODEL SETUP", char="-")
    
    train_schedule = create_learning_rate_schedule(
        config, steps_per_epoch, global_batch_size
    )
    
    model, rngs = initialize_nnx_model(
        config,
        num_classes=len(train_counts),
        include_top=True,
    )
    
    # Count parameters
    params = nnx.state(model, nnx.Param)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model: ResNet-{config.model_depth}")
    print(f"Parameters: {num_params:,}")
    print(f"Width multiplier: {config.width_multiplier}")
    print(f"Dropout rate: {config.dropout_rate}")
    
    state = create_train_state(model, config, train_schedule, rngs=rngs)
    
    # Check for resume
    restored_state = maybe_restore_checkpoint(config, state)
    if restored_state is not None:
        state = restored_state
        print("✓ Restored from checkpoint")

    # =========================================================================
    # Training Setup
    # =========================================================================
    print_header("TRAINING CONFIG", char="-")
    print(f"Epochs:          {config.num_epochs}")
    print(f"Learning rate:   {config.learning_rate}")
    print(f"Min LR:          {config.min_learning_rate}")
    print(f"Warmup epochs:   {config.warmup_epochs}")
    print(f"Weight decay:    {config.weight_decay}")
    print(f"Label smoothing: {config.label_smoothing}")
    print(f"Mixed precision: {config.use_mixed_precision}")
    print(f"Patience:        {config.patience}")
    
    train_step = build_train_step(config, class_weights=class_weights)
    eval_step = build_eval_step(config)

    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
    
    best_val_loss = float('inf')
    history: TrainingHistory = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_macro_f1": [],
    }
    epochs_without_improvement = 0
    best_checkpoint_path: Optional[Path] = None
    best_epoch: Optional[int] = None

    # =========================================================================
    # Training Loop
    # =========================================================================
    print_header("TRAINING START")
    total_start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Generate epoch seed
        epoch_rng, rng = jax.random.split(rng)
        epoch_seed = int(jax.random.randint(epoch_rng, (), 0, 2**31 - 1))
        
        # ---------------------------------------------------------------------
        # Training Phase
        # ---------------------------------------------------------------------
        train_iter = data_module.train_batches(
            rng_seed=epoch_seed,
            batch_size=config.batch_size,
        )
        
        train_losses = []
        train_accs = []
        
        for step, (images, labels) in enumerate(train_iter, start=1):
            state, metrics = train_step(state, (images, labels))
            train_losses.append(float(metrics["loss"]))
            train_accs.append(float(metrics["accuracy"]))
            
            # Step logging
            if step % config.log_every == 0:
                global_step = (epoch - 1) * steps_per_epoch + step
                avg_loss = np.mean(train_losses[-config.log_every:])
                avg_acc = np.mean(train_accs[-config.log_every:])
                
                writer.add_scalars(
                    "train_step",
                    {"loss": avg_loss, "accuracy": avg_acc},
                    global_step=global_step,
                )
                
                if config.log_to_console:
                    print(
                        f"  [Epoch {epoch:02d} | Step {step:04d}/{steps_per_epoch}] "
                        f"loss={avg_loss:.4f}, acc={avg_acc*100:.1f}%",
                        flush=True,
                    )

        # Compute epoch training metrics
        epoch_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_train_acc = float(np.mean(train_accs)) if train_accs else float("nan")

        # ---------------------------------------------------------------------
        # Validation Phase
        # ---------------------------------------------------------------------
        val_losses = []
        val_accs = []
        val_preds_list: list[jnp.ndarray] = []
        val_labels_list: list[jnp.ndarray] = []
        
        for images, labels in data_module.val_batches(batch_size=config.batch_size):
            metrics_dict, preds = eval_step(state, (images, labels))
            val_losses.append(float(metrics_dict["loss"]))
            val_accs.append(float(metrics_dict["accuracy"]))
            val_preds_list.append(preds)
            val_labels_list.append(labels)

        epoch_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        epoch_val_acc = float(np.mean(val_accs)) if val_accs else float("nan")

        # Compute F1 metrics
        val_f1 = float("nan")
        val_macro_f1 = float("nan")
        per_class_f1: list[float] = []
        
        if val_preds_list:
            val_preds = jnp.concatenate(val_preds_list, axis=0)
            val_labels = jnp.concatenate(val_labels_list, axis=0)
            cm = compute_confusion_matrix(
                preds=val_preds,
                labels=val_labels,
                num_classes=len(train_counts),
            )
            val_f1, val_macro_f1, per_class_f1 = compute_f1_metrics(cm)
            
            # Log confusion matrix
            class_names = list(train_counts.keys())
            writer.add_text(
                "epoch/confusion_matrix",
                format_confusion_matrix(cm, class_names),
                global_step=epoch,
            )

        # ---------------------------------------------------------------------
        # Logging & Checkpointing
        # ---------------------------------------------------------------------
        epoch_time = time.time() - epoch_start_time
        
        # Check if this is the best epoch
        is_best = not math.isnan(epoch_val_loss) and epoch_val_loss < best_val_loss
        
        # Print epoch summary
        print_epoch_summary(
            epoch=epoch,
            num_epochs=config.num_epochs,
            train_loss=epoch_train_loss,
            train_acc=epoch_train_acc,
            val_loss=epoch_val_loss,
            val_acc=epoch_val_acc,
            val_f1=val_f1,
            val_macro_f1=val_macro_f1,
            epoch_time=epoch_time,
            best_val_loss=best_val_loss if not math.isinf(best_val_loss) else epoch_val_loss,
            is_best=is_best,
        )

        # TensorBoard logging
        writer.add_scalars(
            "epoch",
            {
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
                "val_f1": val_f1,
                "val_macro_f1": val_macro_f1,
            },
            global_step=epoch,
        )

        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)
        history["val_f1"].append(val_f1)
        history["val_macro_f1"].append(val_macro_f1)

        # Checkpointing
        if is_best:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            best_checkpoint_path = save_checkpoint(state, config, epoch)
            best_epoch = epoch
            print(f"  → Saved best checkpoint: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epoch % config.checkpoint_every == 0:
                ckpt_path = save_checkpoint(state, config, epoch)
                print(f"  → Saved checkpoint: {ckpt_path}")

        # Early stopping check
        if config.patience is not None and epochs_without_improvement >= config.patience:
            print(f"\n⚠ Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    total_time = time.time() - total_start_time
    print_header("TRAINING COMPLETE")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Restore best checkpoint for final evaluation
    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        print(f"\nRestoring best checkpoint from epoch {best_epoch}...")
        template = {
            "model": checkpointing.nnx_state(state.model),
            "opt_state": state.opt_state,
            "rngs": checkpointing.nnx_state(state.rngs),
            "dynamic_scale": state.dynamic_scale,
        }
        restored_best = checkpointing.restore_payload(
            best_checkpoint_path, template=template
        )
        if restored_best is not None:
            checkpointing.apply_nnx_state(
                state.model, restored_best.get("model", template["model"])
            )

    # Test evaluation
    print_header("TEST EVALUATION", char="-")
    test_predictions, test_labels = predict_batches(
        state,
        data_module.test_batches(batch_size=config.batch_size),
        config,
    )
    
    test_accuracy = None
    test_f1: Optional[float] = None
    test_macro_f1: Optional[float] = None
    
    if test_predictions.size > 0:
        test_accuracy = float(jnp.mean(test_predictions == test_labels))
        test_cm = compute_confusion_matrix(
            preds=test_predictions,
            labels=test_labels,
            num_classes=len(train_counts),
        )
        test_f1, test_macro_f1, test_per_class_f1 = compute_f1_metrics(test_cm)
        
        print(f"Test Accuracy:  {test_accuracy*100:.2f}%")
        print(f"Test F1 (micro): {test_f1:.4f}")
        print(f"Test F1 (macro): {test_macro_f1:.4f}")
        
        # Print per-class F1
        print("\nPer-class F1 scores:")
        class_names = list(train_counts.keys())
        for name, score in zip(class_names, test_per_class_f1):
            score_str = f"{score:.4f}" if not math.isnan(score) else "N/A"
            print(f"  {name:12s}: {score_str}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(format_confusion_matrix(test_cm, class_names))

    writer.close()
    
    # =========================================================================
    # Return Summary
    # =========================================================================
    return {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "train_accuracy": history["train_accuracy"][-1] if history["train_accuracy"] else float("nan"),
        "val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
        "val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else float("nan"),
        "val_f1": history["val_f1"][-1] if history["val_f1"] else float("nan"),
        "val_macro_f1": history["val_macro_f1"][-1] if history["val_macro_f1"] else float("nan"),
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_macro_f1": test_macro_f1,
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "best_epoch": best_epoch,
        "history": history,
    }
