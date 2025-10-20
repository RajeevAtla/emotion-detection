"""Training loops, utilities, and evaluation helpers for emotion detection."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import metrax as mx
import optax
import orbax.checkpoint as ocp
import numpy as np
from flax.core import FrozenDict, freeze
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale
from tensorboardX import SummaryWriter

from src.data import DataModuleConfig, EmotionDataModule
from src.model import ResNet, build_finetune_mask, create_resnet, maybe_load_pretrained_params


@dataclass
class TrainingConfig:
    """Collection of knobs driving the training loop.

    Attributes:
        data: Data module configuration describing dataset handling.
        output_dir: Directory for checkpoints, logs, and metrics.
        model_depth: ResNet depth to instantiate.
        width_multiplier: Width multiplier applied to stage channels.
        dropout_rate: Dropout probability applied before classification.
        num_epochs: Number of training epochs.
        batch_size: Number of samples per batch.
        learning_rate: Peak learning rate used after warmup.
        min_learning_rate: Minimum learning rate reached after decay.
        warmup_epochs: Count of epochs used for linear warmup.
        weight_decay: Weight decay applied by the optimizer.
        gradient_accumulation_steps: Gradient accumulation factor.
        label_smoothing: Label smoothing factor for cross entropy.
        seed: Global PRNG seed.
        log_every: Training-step logging interval.
        checkpoint_every: Epoch interval for writing checkpoints.
        max_checkpoints: Maximum number of checkpoints to retain.
        use_mixed_precision: Whether to enable mixed precision training.
        patience: Optional early-stopping patience in epochs.
        freeze_stem: Whether to freeze stem parameters.
        freeze_classifier: Whether to freeze classifier parameters.
        frozen_stages: Tuple of stage indices to freeze.
        pretrained_checkpoint: Optional checkpoint path for initialization.
        resume_checkpoint: Optional checkpoint path to resume training.
    """

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


class TrainState(train_state.TrainState):
    """TrainState with batch statistics and optional dynamic loss scaling."""

    batch_stats: FrozenDict[str, Any]
    dynamic_scale: Optional[DynamicScale] = None


def create_learning_rate_schedule(config: TrainingConfig, steps_per_epoch: int) -> optax.Schedule:
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
        cosine_progress = jnp.clip((step_f - warmup_steps_f) / decay_steps_f, 0.0, 1.0)
        cosine_value = 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_progress))
        decay_lr = min_lr + (peak_lr - min_lr) * cosine_value
        warmup_lr = peak_lr * warmup_progress
        return jnp.where(step_f < warmup_steps_f, warmup_lr, decay_lr)

    return schedule


def create_optimizer(config: TrainingConfig, lr_schedule: optax.Schedule, mask: Optional[FrozenDict] = None) -> optax.GradientTransformation:
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
            optax.MultiSteps(tx, every_k_schedule=config.gradient_accumulation_steps),
        )
    if mask is not None:
        tx = cast(optax.GradientTransformation, optax.masked(tx, mask))
    return tx


def create_train_state(
    rng: jax.Array,
    model: ResNet,
    config: TrainingConfig,
    lr_schedule: optax.Schedule,
) -> TrainState:
    """Initialize model parameters and optimizer state.

    Args:
        rng: PRNG key used for parameter initialization.
        model: ResNet module to initialize.
        config: Training configuration controlling optimizer behavior.
        lr_schedule: Learning rate schedule used by the optimizer.

    Returns:
        TrainState: Populated training state ready for training.
    """
    example = jnp.zeros((1, 48, 48, model.config.input_channels), dtype=jnp.float32)
    variables = model.init(rng, example, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", freeze({}))

    if config.pretrained_checkpoint is not None:
        params = maybe_load_pretrained_params(
            params,
            config=replace(
                model.config,
                checkpoint_path=config.pretrained_checkpoint,
            ),
        )

    mask_tree = None
    if config.freeze_stem or config.freeze_classifier or config.frozen_stages:
        mask_tree = build_finetune_mask(
            {"params": params},
            config=replace(
                model.config,
                freeze_stem=config.freeze_stem,
                freeze_classifier=config.freeze_classifier,
                frozen_stages=config.frozen_stages,
            ),
        )["params"]
        mask_tree = freeze(mask_tree)

    optimizer = create_optimizer(config, lr_schedule, mask_tree)
    dynamic_scale = DynamicScale() if config.use_mixed_precision else None

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        dynamic_scale=dynamic_scale,
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


def compute_confusion_matrix(preds: jnp.ndarray, labels: jnp.ndarray, num_classes: int) -> np.ndarray:
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


def build_train_step(
    model: ResNet,
    config: TrainingConfig,
    class_weights: Optional[jnp.ndarray],
):
    """Create a jit-compiled training step function.

    Args:
        model: ResNet module to evaluate.
        config: Training configuration controlling behaviors.
        class_weights: Optional class weights applied to the loss.

    Returns:
        Callable: A function executing a single training step.
    """

    def loss_fn(params, batch_stats, images, labels, rng):
        """Compute loss and metrics for a single batch."""
        variables = {"params": params, "batch_stats": batch_stats}
        logits, new_model_state = model.apply(
            variables,
            images,
            train=True,
            mutable=["batch_stats"],
            rngs={"dropout": rng},
        )
        logits = cast(jnp.ndarray, logits)
        loss = cross_entropy_loss(
            logits,
            labels,
            label_smoothing=config.label_smoothing,
            class_weights=class_weights,
        )
        accuracy = mx.Accuracy.from_model_output(
            predictions=jnp.argmax(logits, axis=-1),
            labels=labels,
        ).compute()
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return loss, (metrics, new_model_state)

    def train_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray], rng: jax.Array) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Execute one optimizer update using the provided batch."""
        images, labels = batch
        if config.use_mixed_precision:
            images = images.astype(jnp.float16)
        else:
            images = images.astype(jnp.float32)

        if config.use_mixed_precision and state.dynamic_scale is not None:
            dynamic_scale = cast(Any, state.dynamic_scale)

            def scaled_loss_fn(params):
                loss, (metrics, new_model_state) = loss_fn(params, state.batch_stats, images, labels, rng)
                return dynamic_scale.scale(loss), (metrics, new_model_state)  # type: ignore[call-arg]

            grad_fn = dynamic_scale.value_and_grad(scaled_loss_fn, has_aux=True)
            (_, (metrics, new_model_state)), grads = grad_fn(state.params)
            loss_scale = getattr(dynamic_scale, "loss_scale", jnp.array(1.0))
            grads = jax.tree_util.tree_map(lambda g: g / loss_scale, grads)
            state = state.apply_gradients(
                grads=grads,
                batch_stats=new_model_state["batch_stats"],
            )
            state = state.replace(dynamic_scale=dynamic_scale)
            return state, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, new_model_state)), grads = grad_fn(
            state.params,
            state.batch_stats,
            images,
            labels,
            rng,
        )
        state = state.apply_gradients(
            grads=grads,
            batch_stats=new_model_state["batch_stats"],
        )
        return state, metrics

    return jax.jit(train_step)


def build_eval_step(model: ResNet, config: TrainingConfig):
    """Create a JIT-compiled evaluation function.

    Args:
        model: ResNet module to evaluate.
        config: Training configuration controlling behaviors.

    Returns:
        Callable: Function computing evaluation metrics for a batch.
    """
    def eval_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """Evaluate a batch and return metrics and predictions."""
        images, labels = batch
        if config.use_mixed_precision:
            images = images.astype(jnp.float16)
        else:
            images = images.astype(jnp.float32)
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        logits = cast(jnp.ndarray, model.apply(variables, images, train=False, mutable=False))
        loss = cross_entropy_loss(
            logits,
            labels,
            label_smoothing=config.label_smoothing,
        )
        preds = jnp.argmax(logits, axis=-1)
        metrics = {
            "loss": loss,
            "accuracy": mx.Accuracy.from_model_output(
                predictions=preds,
                labels=labels,
            ).compute(),
        }
        return metrics, preds

    return jax.jit(eval_step)


def save_checkpoint(state: TrainState, config: TrainingConfig, epoch: int) -> None:
    """Persist the current training state to disk.

    Args:
        state: Training state containing parameters and optimizer state.
        config: Training configuration providing output directory metadata.
        epoch: Epoch number associated with the checkpoint.
    """
    payload = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "opt_state": state.opt_state,
        "dynamic_scale": state.dynamic_scale,
    }
    ckpt_dir = config.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    target_path = ckpt_dir / f"epoch_{epoch:04d}"
    save_args = ocp.args.PyTreeSave(payload)
    checkpointer.save(str(target_path), payload, save_args=save_args, force=True)

    existing = sorted(ckpt_dir.glob("epoch_*"))
    excess = len(existing) - config.max_checkpoints
    if excess > 0:
        for path in existing[:excess]:
            shutil.rmtree(path, ignore_errors=True)


def maybe_restore_checkpoint(config: TrainingConfig, state: TrainState) -> Optional[Dict[str, Any]]:
    """Restore a checkpoint if ``resume_checkpoint`` is specified.

    Args:
        config: Training configuration referencing an optional checkpoint.
        state: Current training state providing structure for restoration.

    Returns:
        Optional[Dict[str, Any]]: Restored checkpoint payload or ``None``.
    """
    if config.resume_checkpoint is None:
        return None
    checkpointer = ocp.PyTreeCheckpointer()
    template = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "opt_state": state.opt_state,
        "dynamic_scale": state.dynamic_scale,
    }
    restore_args = ocp.args.PyTreeRestore(item=template)
    return checkpointer.restore(str(config.resume_checkpoint), item=template, restore_args=restore_args)


def predict_batches(
    state: TrainState,
    model: ResNet,
    batches: Iterable[Tuple[jnp.ndarray, jnp.ndarray]],
    config: TrainingConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate predictions for an iterator of batches.

    Args:
        state: Training state containing model parameters.
        model: ResNet module used for inference.
        batches: Iterable yielding image and label tensors.
        config: Training configuration controlling precision.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Prediction and label arrays.
    """
    preds = []
    labels_list = []
    for images, labels in batches:
        if config.use_mixed_precision:
            images = images.astype(jnp.float16)
        else:
            images = images.astype(jnp.float32)
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        logits = cast(jnp.ndarray, model.apply(variables, images, train=False, mutable=False))
        preds.append(jnp.argmax(logits, axis=-1))
        labels_list.append(labels)
    if not preds:
        return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32)
    return jnp.concatenate(preds, axis=0), jnp.concatenate(labels_list, axis=0)


def train_and_evaluate(config: TrainingConfig) -> Dict[str, Any]:
    """Run the full training, validation, and test evaluation pipeline.

    Args:
        config: Training configuration describing the experiment.

    Returns:
        Dict[str, Any]: Metrics summarizing the training run.
    """

    rng = jax.random.PRNGKey(config.seed)

    data_config = replace(config.data)
    data_module = EmotionDataModule(data_config)
    data_module.setup()
    class_weights = data_module.class_weights
    train_counts = data_module.split_counts()["train"]
    num_train_examples = sum(train_counts.values())
    steps_per_epoch = max(1, num_train_examples // config.batch_size)
    train_schedule = create_learning_rate_schedule(config, steps_per_epoch)

    model = create_resnet(
        depth=config.model_depth,
        num_classes=len(train_counts),
        width_multiplier=config.width_multiplier,
        include_top=True,
        input_projection_channels=None,
    )
    model = model.replace(dropout_rate=config.dropout_rate)  # type: ignore[arg-type]

    state = create_train_state(rng, model, config, train_schedule)
    restored_state = maybe_restore_checkpoint(config, state)
    if restored_state is not None:
        state = state.replace(
            params=restored_state["params"],
            batch_stats=restored_state.get("batch_stats", state.batch_stats),
            opt_state=restored_state.get("opt_state", state.opt_state),
            dynamic_scale=restored_state.get("dynamic_scale", state.dynamic_scale),
        )
    train_step = build_train_step(model, config, class_weights=class_weights)
    eval_step = build_eval_step(model, config)

    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
    best_val_loss = jnp.inf
    history: Dict[str, list[float]] = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    epochs_without_improvement = 0
    for epoch in range(1, config.num_epochs + 1):
        epoch_rng, rng = jax.random.split(rng)
        epoch_seed = int(jax.random.randint(epoch_rng, (), 0, 2**31 - 1))
        train_iter = data_module.train_batches(
            rng_seed=epoch_seed,
            batch_size=config.batch_size,
        )
        train_metrics = []

        for step, (images, labels) in enumerate(train_iter, start=1):
            state, metrics = train_step(
                state,
                (images, labels),
                jax.random.fold_in(epoch_rng, step),
            )
            train_metrics.append(metrics)
            if step % config.log_every == 0:
                global_step = (epoch - 1) * steps_per_epoch + step
                writer.add_scalars(
                    "train_step",
                    {"loss": float(metrics["loss"]), "accuracy": float(metrics["accuracy"])},
                    global_step=global_step,
                )

        if train_metrics:
            epoch_train_loss = float(jnp.mean(jnp.asarray([m["loss"] for m in train_metrics])))
            epoch_train_acc = float(jnp.mean(jnp.asarray([m["accuracy"] for m in train_metrics])))
        else:
            epoch_train_loss = float("nan")
            epoch_train_acc = float("nan")

        val_metrics = []
        val_preds_list = []
        val_labels_list = []
        for images, labels in data_module.val_batches(batch_size=config.batch_size):
            metrics_dict, preds = eval_step(state, (images, labels))
            val_metrics.append(metrics_dict)
            val_preds_list.append(preds)
            val_labels_list.append(labels)
        if val_metrics:
            epoch_val_loss = float(jnp.mean(jnp.asarray([m["loss"] for m in val_metrics])))
            epoch_val_acc = float(jnp.mean(jnp.asarray([m["accuracy"] for m in val_metrics])))
        else:
            epoch_val_loss = float("nan")
            epoch_val_acc = float("nan")

        if val_preds_list:
            val_preds = jnp.concatenate(val_preds_list, axis=0)
            val_labels = jnp.concatenate(val_labels_list, axis=0)
            cm = compute_confusion_matrix(
                preds=val_preds,
                labels=val_labels,
                num_classes=len(train_counts),
            )
            class_names = list(train_counts.keys())
            writer.add_text(
                "epoch/confusion_matrix",
                format_confusion_matrix(cm, class_names),
                global_step=epoch,
            )

        writer.add_scalars(
            "epoch",
            {
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
            },
            global_step=epoch,
        )

        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        improved = not math.isnan(epoch_val_loss) and epoch_val_loss < best_val_loss
        if improved:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % config.checkpoint_every == 0 or improved:
            save_checkpoint(state, config, epoch)

        if config.patience is not None and epochs_without_improvement >= config.patience:
            break

    test_predictions, test_labels = predict_batches(
        state,
        model,
        data_module.test_batches(batch_size=config.batch_size),
        config,
    )
    test_accuracy = None
    if test_predictions.size > 0:
        test_metric = mx.Accuracy.from_model_output(
            predictions=test_predictions,
            labels=test_labels,
        )
        test_accuracy = float(test_metric.compute())

    writer.close()
    return {
        "train_loss": history["train_loss"][-1],
        "train_accuracy": history["train_accuracy"][-1],
        "val_loss": history["val_loss"][-1],
        "val_accuracy": history["val_accuracy"][-1],
        "test_accuracy": test_accuracy,
        "history": history,
    }
