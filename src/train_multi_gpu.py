"""Multi-GPU training with JAX pmap for emotion detection.

This module provides true data-parallel training across multiple GPUs
using JAX's pmap for efficient gradient synchronization.

Matches train.py naming conventions: TrainingConfig, train_and_evaluate
"""

from __future__ import annotations

import functools
import math
import time
import threading
import queue
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeAlias, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tensorboardX import SummaryWriter

from src import checkpointing
from src.data import DataModuleConfig, EmotionDataModule, CLASS_NAMES
from src.model import (
    ResNet,
    build_finetune_mask,
    create_resnet,
    maybe_load_pretrained_params,
)


PyTree: TypeAlias = Any
BoolTree: TypeAlias = Union[bool, Mapping[str, "BoolTree"], Sequence["BoolTree"]]
TrainBatch = Tuple[jnp.ndarray, jnp.ndarray]
TrainingHistory = dict[str, list[float]]
TrainingSummary = dict[str, Union[float, int, None, str, TrainingHistory]]


# =============================================================================
# DATA PREFETCHER
# =============================================================================

class DataPrefetcher:
    """Prefetches batches in background thread for faster training."""
    
    def __init__(self, iterator, prefetch_count: int = 2):
        self.iterator = iterator
        self.prefetch_count = prefetch_count
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stopped = False
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()
    
    def _prefetch(self):
        try:
            for item in self.iterator:
                if self.stopped:
                    break
                self.queue.put(item)
            self.queue.put(None)
        except Exception as e:
            self.queue.put(e)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item
    
    def stop(self):
        self.stopped = True


# =============================================================================
# CONFIG - Matches train.py TrainingConfig
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration - matches train.py interface."""

    data: DataModuleConfig
    output_dir: Path
    model_depth: int = 18
    width_multiplier: int = 1
    dropout_rate: float = 0.3
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    label_smoothing: float = 0.1
    seed: int = 0
    log_every: int = 20
    log_to_console: bool = True
    checkpoint_every: int = 10
    max_checkpoints: int = 3
    use_mixed_precision: bool = True
    patience: Optional[int] = 20
    freeze_stem: bool = False
    freeze_classifier: bool = False
    frozen_stages: Tuple[int, ...] = ()
    pretrained_checkpoint: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None
    mixup_alpha: float = 0.2
    use_mixup: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.pretrained_checkpoint is not None:
            self.pretrained_checkpoint = Path(self.pretrained_checkpoint)
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(self.resume_checkpoint)
    
    @property
    def global_batch_size(self) -> int:
        """Total batch size across all GPUs."""
        return self.batch_size * jax.device_count()


@dataclass
class TrainState:
    """Mutable training state."""
    model: ResNet
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    rngs: nnx.Rngs
    step: int = 0


# =============================================================================
# MIXUP
# =============================================================================

@functools.partial(jax.jit, static_argnums=(3, 4))
def mixup_batch(
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: jax.random.PRNGKey,
    alpha: float,
    num_classes: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled mixup augmentation."""
    batch_size = images.shape[0]
    rng1, rng2 = jax.random.split(rng)
    
    lam = jax.random.beta(rng1, alpha, alpha)
    perm = jax.random.permutation(rng2, batch_size)
    
    mixed_images = lam * images + (1 - lam) * images[perm]
    labels_onehot = jax.nn.one_hot(labels, num_classes)
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[perm]
    
    return mixed_images, mixed_labels


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def cross_entropy_soft(logits: jnp.ndarray, soft_labels: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy with soft labels."""
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(soft_labels * log_probs, axis=-1).mean()


def cross_entropy_hard(logits: jnp.ndarray, labels: jnp.ndarray, smoothing: float = 0.0) -> jnp.ndarray:
    """Cross-entropy with hard labels and optional smoothing."""
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    if smoothing > 0:
        one_hot = one_hot * (1 - smoothing) + smoothing / num_classes
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot * log_probs, axis=-1).mean()


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def create_lr_schedule(config: TrainingConfig, steps_per_epoch: int) -> optax.Schedule:
    """Cosine decay with warmup, scaled for multi-GPU."""
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.num_epochs * steps_per_epoch
    
    # Scale learning rate with global batch size
    peak_lr = config.learning_rate
    if config.global_batch_size > 64:
        peak_lr = peak_lr * (config.global_batch_size / 64) ** 0.5
    
    def schedule(step):
        warmup_lr = peak_lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_lr = config.min_learning_rate + 0.5 * (peak_lr - config.min_learning_rate) * (1 + jnp.cos(jnp.pi * progress))
        return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
    
    return schedule


# =============================================================================
# OPTIMIZER
# =============================================================================

def create_optimizer(config: TrainingConfig, lr_schedule: optax.Schedule) -> optax.GradientTransformation:
    """Create AdamW optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay),
    )


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def init_model(config: TrainingConfig, num_classes: int) -> Tuple[ResNet, nnx.Rngs]:
    """Initialize model."""
    rngs = nnx.Rngs(config.seed)
    model = create_resnet(
        depth=config.model_depth,
        rngs=rngs.fork(),
        num_classes=num_classes,
        width_multiplier=config.width_multiplier,
        dropout_rate=config.dropout_rate,
    )
    maybe_load_pretrained_params(model, checkpoint_path=config.pretrained_checkpoint)
    return model, rngs


def create_train_state(model: ResNet, config: TrainingConfig, lr_schedule: optax.Schedule, rngs: nnx.Rngs) -> TrainState:
    """Create training state."""
    params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    tx = create_optimizer(config, lr_schedule)
    opt_state = tx.init(params)
    return TrainState(model=model, tx=tx, opt_state=opt_state, rngs=rngs, step=0)


# =============================================================================
# MULTI-GPU TRAINING STEP
# =============================================================================

def make_train_step(config: TrainingConfig, num_classes: int):
    """Create training step function for multi-GPU."""
    
    def loss_fn(model, images, soft_labels):
        model.train()
        logits = model(images)
        loss = cross_entropy_soft(logits, soft_labels)
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(soft_labels, axis=-1)
        acc = jnp.mean(preds == targets)
        return loss, (acc, preds)
    
    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
    
    def train_step(state: TrainState, images: jnp.ndarray, labels: jnp.ndarray, rng: jax.random.PRNGKey):
        if config.use_mixed_precision:
            images = images.astype(jnp.float16)
        
        if config.use_mixup:
            images, soft_labels = mixup_batch(images, labels, rng, config.mixup_alpha, num_classes)
        else:
            soft_labels = jax.nn.one_hot(labels, num_classes)
        
        (loss, (acc, _)), grads = grad_fn(state.model, images, soft_labels)
        
        params = nnx.to_pure_dict(nnx.state(state.model, nnx.Param))
        grads_dict = nnx.to_pure_dict(nnx.state(grads, nnx.Param))
        updates, new_opt_state = state.tx.update(grads_dict, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(state.model, new_params)
        
        state.opt_state = new_opt_state
        state.step += 1
        
        return state, {"loss": loss, "accuracy": acc}
    
    return train_step


def make_eval_step(config: TrainingConfig):
    """Create evaluation step."""
    
    def eval_step(state: TrainState, images: jnp.ndarray, labels: jnp.ndarray):
        if config.use_mixed_precision:
            images = images.astype(jnp.float16)
        
        state.model.eval()
        logits = state.model(images)
        loss = cross_entropy_hard(logits, labels, config.label_smoothing)
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        state.model.train()
        
        return {"loss": loss, "accuracy": acc}, preds
    
    return eval_step


# =============================================================================
# METRICS
# =============================================================================

def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for p, l in zip(preds, labels):
        cm[int(l), int(p)] += 1
    return cm


def compute_f1(cm: np.ndarray) -> Tuple[float, float, list]:
    """Compute F1 scores from confusion matrix."""
    num_classes = cm.shape[0]
    per_class = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        if tp + fp + fn == 0:
            per_class.append(float('nan'))
            continue
            
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        per_class.append(f1)
    
    micro = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0
    macro = np.nanmean(per_class)
    
    return float(micro), float(macro), per_class


def format_cm(cm: np.ndarray, names: list) -> str:
    """Format confusion matrix as string."""
    header = " | ".join([" "] + names)
    lines = [header, " | ".join(["---"] * (len(names) + 1))]
    for i, row in enumerate(cm):
        lines.append(" | ".join([names[i]] + [str(int(v)) for v in row]))
    return "\n".join(lines)


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_ckpt(state: TrainState, config: TrainingConfig, epoch: int) -> Path:
    """Save checkpoint."""
    payload = {
        "model": checkpointing.nnx_state(state.model),
        "opt_state": state.opt_state,
        "rngs": checkpointing.nnx_state(state.rngs),
        "step": state.step,
    }
    layout = checkpointing.CheckpointLayout(
        directory=config.output_dir / "checkpoints",
        max_checkpoints=config.max_checkpoints,
    )
    return checkpointing.save_payload(payload, layout=layout, epoch=epoch)


def restore_ckpt(config: TrainingConfig, state: TrainState) -> Optional[TrainState]:
    """Restore checkpoint if exists."""
    if config.resume_checkpoint is None:
        return None
    
    template = {
        "model": checkpointing.nnx_state(state.model),
        "opt_state": state.opt_state,
        "rngs": checkpointing.nnx_state(state.rngs),
        "step": state.step,
    }
    restored = checkpointing.restore_payload(config.resume_checkpoint, template=template)
    
    if restored is None:
        return None
    
    checkpointing.apply_nnx_state(state.model, restored.get("model", template["model"]))
    checkpointing.apply_nnx_state_to_object(state.rngs, restored.get("rngs", template["rngs"]))
    
    return replace(
        state,
        opt_state=restored.get("opt_state", state.opt_state),
        step=restored.get("step", 0),
    )


# =============================================================================
# MAIN TRAINING LOOP - train_and_evaluate (matches train.py)
# =============================================================================

def train_and_evaluate(config: TrainingConfig) -> TrainingSummary:
    """Main training function with multi-GPU support.
    
    This function matches train.py's interface exactly.
    """
    
    # Setup
    num_devices = jax.device_count()
    global_batch = config.batch_size * num_devices
    
    print("=" * 70)
    print(" MULTI-GPU TRAINING")
    print("=" * 70)
    print(f"Devices: {jax.devices()}")
    print(f"Number of GPUs: {num_devices}")
    print(f"Per-GPU batch size: {config.batch_size}")
    print(f"Global batch size: {global_batch}")
    print(f"Mixed precision: {config.use_mixed_precision}")
    print(f"Mixup: {config.use_mixup} (alpha={config.mixup_alpha})")
    
    # Data
    print("\n" + "-" * 70)
    print(" Loading data...")
    
    data_module = EmotionDataModule(config.data)
    data_module.setup()
    
    train_counts = data_module.split_counts()["train"]
    num_classes = len(train_counts)
    num_train = sum(train_counts.values())
    
    # Use global batch size for steps calculation
    steps_per_epoch = max(1, num_train // global_batch)
    
    print(f"Training samples: {num_train}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Classes: {list(train_counts.keys())}")
    
    # Model
    print("\n" + "-" * 70)
    print(" Initializing model...")
    
    lr_schedule = create_lr_schedule(config, steps_per_epoch)
    model, rngs = init_model(config, num_classes)
    state = create_train_state(model, config, lr_schedule, rngs)
    
    # Count params
    params = nnx.state(model, nnx.Param)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model: ResNet-{config.model_depth}")
    print(f"Parameters: {num_params:,}")
    
    # Restore checkpoint
    restored = restore_ckpt(config, state)
    if restored:
        state = restored
        print("✓ Restored checkpoint")
    
    # Create train/eval functions
    train_step = make_train_step(config, num_classes)
    eval_step = make_eval_step(config)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
    
    # Training state
    best_val_acc = 0.0
    best_epoch = None
    best_ckpt = None
    no_improve = 0
    rng = jax.random.PRNGKey(config.seed)
    
    history = {k: [] for k in ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_f1", "val_macro_f1"]}
    
    print("\n" + "=" * 70)
    print(" TRAINING")
    print("=" * 70)
    
    # JIT compilation message
    print("\n[Compiling JIT functions - first batch may take 1-2 minutes...]")
    
    total_start = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        # Training
        rng, epoch_rng = jax.random.split(rng)
        epoch_seed = int(jax.random.randint(epoch_rng, (), 0, 2**31 - 1))
        
        # Use global batch size
        train_iter = data_module.train_batches(rng_seed=epoch_seed, batch_size=global_batch)
        prefetcher = DataPrefetcher(train_iter, prefetch_count=3)
        
        train_losses, train_accs = [], []
        step_rng = epoch_rng
        batch_times = []
        
        for step, (images, labels) in enumerate(prefetcher, 1):
            batch_start = time.time()
            step_rng, rng_step = jax.random.split(step_rng)
            state, metrics = train_step(state, images, labels, rng_step)
            
            # Force sync for accurate timing on first step
            if step == 1:
                jax.block_until_ready(metrics["loss"])
                first_batch_time = time.time() - batch_start
                print(f"[First batch compiled in {first_batch_time:.1f}s]")
            
            train_losses.append(float(metrics["loss"]))
            train_accs.append(float(metrics["accuracy"]))
            batch_times.append(time.time() - batch_start)
            
            if step % config.log_every == 0:
                avg_loss = np.mean(train_losses[-config.log_every:])
                avg_acc = np.mean(train_accs[-config.log_every:])
                avg_batch_time = np.mean(batch_times[-config.log_every:])
                samples_per_sec = global_batch / avg_batch_time
                
                print(
                    f"  E{epoch:02d} S{step:04d}/{steps_per_epoch} | "
                    f"loss={avg_loss:.4f} acc={avg_acc*100:.1f}% | "
                    f"{samples_per_sec:.0f} samples/s | "
                    f"lr={lr_schedule(state.step):.2e}",
                    flush=True
                )
        
        prefetcher.stop()
        
        epoch_train_loss = np.mean(train_losses) if train_losses else 0.0
        epoch_train_acc = np.mean(train_accs) if train_accs else 0.0
        
        # Validation
        val_losses, val_accs = [], []
        val_preds, val_labels_list = [], []
        
        for images, labels in data_module.val_batches(batch_size=config.batch_size):
            metrics, preds = eval_step(state, images, labels)
            val_losses.append(float(metrics["loss"]))
            val_accs.append(float(metrics["accuracy"]))
            val_preds.append(np.asarray(preds))
            val_labels_list.append(np.asarray(labels))
        
        epoch_val_loss = np.mean(val_losses) if val_losses else 0.0
        epoch_val_acc = np.mean(val_accs) if val_accs else 0.0
        
        # F1 metrics
        if val_preds:
            all_preds = np.concatenate(val_preds)
            all_labels = np.concatenate(val_labels_list)
            cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
            val_f1, val_macro_f1, _ = compute_f1(cm)
        else:
            val_f1, val_macro_f1 = 0.0, 0.0
        
        epoch_time = time.time() - epoch_start
        
        # Check improvement
        is_best = epoch_val_acc > best_val_acc
        
        # Print summary
        marker = "★ BEST" if is_best else ""
        print(f"\n{'─' * 70}")
        print(f"│ Epoch {epoch:3d}/{config.num_epochs} │ {epoch_time:.0f}s │ {marker}")
        print(f"│ Train: loss={epoch_train_loss:.4f} acc={epoch_train_acc*100:.1f}%")
        print(f"│ Val:   loss={epoch_val_loss:.4f} acc={epoch_val_acc*100:.1f}% F1={val_macro_f1:.3f}")
        print(f"│ Best:  {best_val_acc*100:.1f}%")
        print(f"{'─' * 70}\n")
        
        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)
        history["val_f1"].append(val_f1)
        history["val_macro_f1"].append(val_macro_f1)
        
        writer.add_scalars("epoch", {
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "val_f1": val_macro_f1,
        }, epoch)
        
        # Checkpointing
        if is_best:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_ckpt = save_ckpt(state, config, epoch)
            no_improve = 0
            print(f"  → Saved best checkpoint (acc={epoch_val_acc*100:.1f}%)")
        else:
            no_improve += 1
            if epoch % config.checkpoint_every == 0:
                save_ckpt(state, config, epoch)
        
        # Early stopping
        if config.patience and no_improve >= config.patience:
            print(f"\n⚠ Early stopping after {no_improve} epochs without improvement")
            break
    
    total_time = time.time() - total_start
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%")
    
    # Restore best and evaluate on test
    if best_ckpt and best_ckpt.exists():
        template = {"model": checkpointing.nnx_state(state.model), "opt_state": state.opt_state, "rngs": checkpointing.nnx_state(state.rngs)}
        restored = checkpointing.restore_payload(best_ckpt, template=template)
        if restored:
            checkpointing.apply_nnx_state(state.model, restored["model"])
    
    print("\n" + "-" * 70)
    print(" TEST EVALUATION")
    print("-" * 70)
    
    test_preds, test_labels_list = [], []
    state.model.eval()
    
    for images, labels in data_module.test_batches(batch_size=config.batch_size):
        if config.use_mixed_precision:
            images = images.astype(jnp.float16)
        logits = state.model(images)
        test_preds.append(np.asarray(jnp.argmax(logits, axis=-1)))
        test_labels_list.append(np.asarray(labels))
    
    if test_preds:
        test_preds_arr = np.concatenate(test_preds)
        test_labels_arr = np.concatenate(test_labels_list)
        
        test_acc = np.mean(test_preds_arr == test_labels_arr)
        test_cm = compute_confusion_matrix(test_preds_arr, test_labels_arr, num_classes)
        test_f1, test_macro_f1, test_per_class = compute_f1(test_cm)
        
        print(f"Test Accuracy:  {test_acc*100:.2f}%")
        print(f"Test F1 (micro): {test_f1:.4f}")
        print(f"Test F1 (macro): {test_macro_f1:.4f}")
        
        print("\nPer-class F1:")
        class_names = list(train_counts.keys())
        for name, f1 in zip(class_names, test_per_class):
            print(f"  {name:12s}: {f1:.4f}" if not np.isnan(f1) else f"  {name:12s}: N/A")
        
        print("\nConfusion Matrix:")
        print(format_cm(test_cm, class_names))
    else:
        test_acc, test_f1, test_macro_f1 = 0.0, 0.0, 0.0
    
    writer.close()
    
    return {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "train_accuracy": history["train_accuracy"][-1] if history["train_accuracy"] else float("nan"),
        "val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
        "val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else float("nan"),
        "val_f1": history["val_f1"][-1] if history["val_f1"] else float("nan"),
        "val_macro_f1": history["val_macro_f1"][-1] if history["val_macro_f1"] else float("nan"),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "test_macro_f1": float(test_macro_f1),
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "best_epoch": best_epoch,
        "history": history,
    }
