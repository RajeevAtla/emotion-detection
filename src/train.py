"""Training loops, utilities, and evaluation helpers for emotion detection."""

from __future__ import annotations

import math
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
    log_every: int = 100
    log_to_console: bool = False
    checkpoint_every: int = 5
    max_checkpoints: int = 3
    use_mixed_precision: bool = False
    patience: Optional[int] = None
    freeze_stem: bool = False
    freeze_classifier: bool = False
    frozen_stages: Tuple[int, ...] = ()
    pretrained_checkpoint: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None

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
    config: TrainingConfig, steps_per_epoch: int
) -> optax.Schedule:
    """Build a learning rate schedule with warmup followed by cosine decay.

    Args:
        config: Training configuration supplying learning rate hyperparameters.
        steps_per_epoch: Number of optimizer steps executed per epoch.

    Returns:
        optax.Schedule: Callable returning a scalar learning rate.
    """
    warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)
    total_steps = max(1, config.num_epochs * steps_per_epoch)
    if warmup_steps >= total_steps:
        warmup_steps = max(1, total_steps - 1)
    total_steps = max(warmup_steps + 1, total_steps)
    decay_steps = max(1, total_steps - warmup_steps)

    peak_lr = config.learning_rate
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
    """Create the optimizer transformation for training.

    Args:
        config: Training configuration supplying optimizer hyperparameters.
        lr_schedule: Learning rate schedule callable.
        mask: Optional mask marking trainable parameters.

    Returns:
        optax.GradientTransformation: Configured optimizer chain.
    """
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
    """Compute label-smoothed cross-entropy with optional class weighting.

    Args:
        logits: Logit tensor of shape ``(batch, num_classes)``.
        labels: Integer labels of shape ``(batch,)``.
        label_smoothing: Amount of label smoothing to apply.
        class_weights: Optional class weights broadcast across the batch.

    Returns:
        jnp.ndarray: Scalar loss averaged across the batch.
    """
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
    """Return confusion matrix with shape ``(num_classes, num_classes)``.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.
        num_classes: Total number of classes.

    Returns:
        np.ndarray: Confusion matrix where rows correspond to labels.
    """
    preds_np = np.asarray(preds)
    labels_np = np.asarray(labels)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for label, pred in zip(labels_np, preds_np):
        cm[int(label), int(pred)] += 1
    return cm


def format_confusion_matrix(cm: np.ndarray, class_names: Iterable[str]) -> str:
    """Format confusion matrix as a Markdown table string.

    Args:
        cm: Confusion matrix array.
        class_names: Iterable of class names aligning with matrix rows.

    Returns:
        str: Markdown-formatted confusion matrix.
    """
    header = [" "] + list(class_names)
    lines = [" | ".join(header)]
    lines.append(" | ".join(["---"] * len(header)))
    for idx, row in enumerate(cm):
        row_vals = [str(header[idx + 1])] + [str(int(val)) for val in row]
        lines.append(" | ".join(row_vals))
    return "\n".join(lines)


def compute_f1_metrics(cm: np.ndarray) -> tuple[float, float, list[float]]:
    """Compute micro and macro F1 along with per-class F1 scores.

    Args:
        cm: Confusion matrix where rows correspond to ground-truth labels.

    Returns:
        tuple[float, float, list[float]]: Micro F1, macro F1, per-class F1 list.
    """
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
        accuracy = mx.Accuracy.from_model_output(
            predictions=preds,
            labels=labels,
        ).compute()
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
        accuracy = mx.Accuracy.from_model_output(
            predictions=preds,
            labels=labels,
        ).compute()
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


def train_and_evaluate(config: TrainingConfig) -> TrainingSummary:
    """Run the training pipeline using the NNX-native state."""
    rng = jax.random.PRNGKey(config.seed)
    data_config = replace(config.data)
    data_module = EmotionDataModule(data_config)
    data_module.setup()
    class_weights = data_module.class_weights
    train_counts = data_module.split_counts()["train"]
    num_train_examples = sum(train_counts.values())
    steps_per_epoch = max(1, num_train_examples // config.batch_size)
    train_schedule = create_learning_rate_schedule(config, steps_per_epoch)

    model, rngs = initialize_nnx_model(
        config,
        num_classes=len(train_counts),
        include_top=True,
    )
    state = create_train_state(
        model, config, train_schedule, rngs=rngs
    )
    restored_state = maybe_restore_checkpoint(config, state)
    if restored_state is not None:
        state = restored_state

    train_step = build_train_step(
        config,
        class_weights=class_weights,
    )
    eval_step = build_eval_step(config)

    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
    best_val_loss = jnp.inf
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

    for epoch in range(1, config.num_epochs + 1):
        epoch_rng, rng = jax.random.split(rng)
        epoch_seed = int(jax.random.randint(epoch_rng, (), 0, 2**31 - 1))
        train_iter = data_module.train_batches(
            rng_seed=epoch_seed,
            batch_size=config.batch_size,
        )
        train_metrics = []
        for step, (images, labels) in enumerate(train_iter, start=1):
            state, metrics = train_step(state, (images, labels))
            train_metrics.append(metrics)
            if step % config.log_every == 0:
                global_step = (epoch - 1) * steps_per_epoch + step
                writer.add_scalars(
                    "train_step",
                    {
                        "loss": float(metrics["loss"]),
                        "accuracy": float(metrics["accuracy"]),
                    },
                    global_step=global_step,
                )
                if config.log_to_console:
                    print(
                        "[epoch "
                        f"{epoch:03d} step {step:05d}] "
                        f"train loss={float(metrics['loss']):.4f} "
                        f"accuracy={float(metrics['accuracy']):.4f}",
                        flush=True,
                    )

        if train_metrics:
            epoch_train_loss = float(
                jnp.mean(jnp.asarray([m["loss"] for m in train_metrics]))
            )
            epoch_train_acc = float(
                jnp.mean(jnp.asarray([m["accuracy"] for m in train_metrics]))
            )
        else:
            epoch_train_loss = float("nan")
            epoch_train_acc = float("nan")

        val_metrics: list[dict[str, jnp.ndarray]] = []
        val_preds_list: list[jnp.ndarray] = []
        val_labels_list: list[jnp.ndarray] = []
        for images, labels in data_module.val_batches(
            batch_size=config.batch_size
        ):
            metrics_dict, preds = eval_step(state, (images, labels))
            val_metrics.append(metrics_dict)
            val_preds_list.append(preds)
            val_labels_list.append(labels)

        if val_metrics:
            epoch_val_loss = float(
                jnp.mean(jnp.asarray([m["loss"] for m in val_metrics]))
            )
            epoch_val_acc = float(
                jnp.mean(jnp.asarray([m["accuracy"] for m in val_metrics]))
            )
        else:
            epoch_val_loss = float("nan")
            epoch_val_acc = float("nan")

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
            class_names = list(train_counts.keys())
            writer.add_text(
                "epoch/confusion_matrix",
                format_confusion_matrix(cm, class_names),
                global_step=epoch,
            )
            per_class_line = ", ".join(
                f"{name}: "
                f"{'nan' if math.isnan(float(score)) else f'{float(score):.4f}'}"
                for name, score in zip(class_names, per_class_f1)
            )
            if per_class_line:
                writer.add_text(
                    "epoch/val_per_class_f1",
                    per_class_line,
                    global_step=epoch,
                )

        writer.add_scalars(
            "epoch_nnx",
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

        if config.log_to_console:
            print(
                "[epoch "
                f"{epoch:03d}] train_loss={epoch_train_loss:.4f} "
                f"train_accuracy={epoch_train_acc:.4f} "
                f"val_loss={epoch_val_loss:.4f} "
                f"val_accuracy={epoch_val_acc:.4f} "
                f"val_f1={val_f1:.4f} "
                f"val_macro_f1={val_macro_f1:.4f}",
                flush=True,
            )

        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)
        history["val_f1"].append(val_f1)
        history["val_macro_f1"].append(val_macro_f1)

        improved = (
            not math.isnan(epoch_val_loss) and epoch_val_loss < best_val_loss
        )
        if improved:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            best_checkpoint_path = save_checkpoint(state, config, epoch)
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
            if epoch % config.checkpoint_every == 0:
                save_checkpoint(state, config, epoch)

        if (
            config.patience is not None
            and epochs_without_improvement >= config.patience
        ):
            break

    if best_checkpoint_path is not None and best_checkpoint_path.exists():
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
            nnx.update(
                state.rngs, restored_best.get("rngs", template["rngs"])
            )
            state = replace(
                state,
                opt_state=restored_best.get("opt_state", state.opt_state),
                dynamic_scale=restored_best.get(
                    "dynamic_scale", state.dynamic_scale
                ),
            )

    test_predictions, test_labels = predict_batches(
        state,
        data_module.test_batches(batch_size=config.batch_size),
        config,
    )
    test_accuracy = None
    test_f1: Optional[float] = None
    test_macro_f1: Optional[float] = None
    if test_predictions.size > 0:
        test_metric = mx.Accuracy.from_model_output(
            predictions=test_predictions,
            labels=test_labels,
        )
        test_accuracy = float(test_metric.compute())
        test_cm = compute_confusion_matrix(
            preds=test_predictions,
            labels=test_labels,
            num_classes=len(train_counts),
        )
        micro, macro, _ = compute_f1_metrics(test_cm)
        test_f1 = micro
        test_macro_f1 = macro

    writer.close()
    return {
        "train_loss": history["train_loss"][-1],
        "train_accuracy": history["train_accuracy"][-1],
        "val_loss": history["val_loss"][-1],
        "val_accuracy": history["val_accuracy"][-1],
        "val_f1": history["val_f1"][-1],
        "val_macro_f1": history["val_macro_f1"][-1],
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_macro_f1": test_macro_f1,
        "best_checkpoint": str(best_checkpoint_path)
        if best_checkpoint_path is not None
        else None,
        "best_epoch": best_epoch if best_epoch is not None else None,
        "history": history,
    }
