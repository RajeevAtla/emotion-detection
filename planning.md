Emotion Detection ResNet Plan
================================

- [x] Review He et al. (2015/2016) ResNet paper and map components to emotion recognition task — residual bottleneck blocks with identity shortcuts mitigate vanishing gradients; final global average pooling simplifies adapting classifier head to emotion classes; plan to start from ResNet-34/18 depending on dataset size and consider freezing early layers during warm-up.
- [x] Inspect dataset structure in `data/` (class labels, splits, file formats) — train/test folders each contain seven emotion classes (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`) stored as 48x48 8-bit grayscale PNGs; class counts are imbalanced (e.g., `disgusted` is only 436 images in train vs. `happy` 7215); need to synthesize validation split from training data.
- [x] Define target image resolution, normalization stats, and augmentation strategy aligned with ResNet preprocessing and JAX expectations - retain native 48x48 resolution (grayscale) to avoid over-upscaling; convert to float32 in `[0,1]`, normalize with dataset mean 0.5077 and std 0.2550 (computed over train split); augment training with random horizontal flip, +/- 15 deg rotation, random resized crop (scale 0.9-1.1) with optional elastic blur; use center-crop-only pipeline for validation/test.
- [x] Decide on ResNet depth (e.g., ResNet-34) balancing dataset size and compute budget — target a CIFAR-style ResNet-34 (3,4,6,3 blocks) with 3x3 stride-1 stem and no initial maxpool to preserve 48x48 spatial detail; expose config switch to downgrade to ResNet-18 for ablations; first conv adapted to single-channel input with optional channel-expansion for transfer learning.
- [x] Decide on JAX ecosystem tooling (Flax for modules and training config via `TrainState`, Optax for optimizers + schedulers, Orbax for checkpointing, MetraX for metrics, Chex for testing, Pydantic for config validation, TensorBoard summaries for experiment tracking, and `uv` for dependency management).
- [x] Outline experiment tracking, checkpointing, and evaluation metrics (accuracy, F1, macro-F1, class confusion matrix) - use MetraX metric collections updated each epoch, log scalars/figures to TensorBoard and structured JSON; Orbax-managed checkpoints storing `TrainState` and config snapshots.
- [x] Schedule implementation order across `src/data.py`, `src/model.py`, `src/train.py`, and `src/main.py` - Phase 1: implement `data.py` loaders/augmentations + associated Chex tests; Phase 2: build `model.py` ResNet modules with config options; Phase 3: wire `train.py` loops (optimizer, metrics, checkpoints); Phase 4: finalize `main.py` orchestration + CLI; Phase 5: harden `src/tests.py` and integration smoke tests.
- [x] Outline Chex-based invariants/tests for data, model, and training loops to validate shapes and dtypes - data pipeline asserts `(batch, 48, 48, 1)` float32 tensors with deterministic augmentations under fixed seeds; model forward pass enforces logits `(batch, 7)` and finite params; training steps check gradient tree structure, non-NaN losses, and optimizer state alignment.
- [x] Define scope for unified `src/tests.py` covering Chex property tests and Pydantic schema validation - consolidate module smoke tests, config validation, and metric sanity checks into a single entrypoint runnable via `uv run python -m pytest src/tests.py` or direct invocation.
- [x] Confirm PyPI package name/source for MetraX metrics library (current `uv add metrax` fails; need guidance before wiring metrics integration). — Installed as `google-metrax`; added `clu` dependency per package requirements.

`src/data.py` Subplan
---------------------

- [x] Audit raw dataset layout and class distribution; document findings - `EmotionDataModule.split_counts()` surfaces per-split class totals for monitoring imbalance.
- [x] Implement dataset loader producing NumPy/JAX arrays with configurable transforms, label mapping, and optional stratified split - `EmotionDataModule` batches yield normalized `jnp` tensors with label indices derived from `CLASS_NAMES`.
- [x] Add Pillow (or equivalent) dependency for PNG decode within UV-managed environment - `Pillow` now managed via `uv`; `_load_image` handles consistent grayscale decoding.
- [x] Add augmentation pipeline (random horizontal flip, +/- 15 deg rotation, random resized crop 0.9-1.1 scale, optional elastic blur) using JAX-friendly preprocessing - `AugmentationConfig` + `apply_augmentations` provide configurable stochastic transforms with deterministic seeding.
- [x] Provide normalization constants (mean=0.5077, std=0.2550) and ensure tensor shapes align with model expectations and JAX layout - per-sample normalization performed in `_iter_batches`, defaulting to computed stats.
- [x] Include utility to compute and cache dataset statistics if not provided - `compute_dataset_statistics` persists per-channel stats to JSON for reuse.
- [x] Create stratified train/validation split from training data with configurable hold-out ratio - `stratified_split` balances validation sampling across classes.
- [x] Evaluate and mitigate class imbalance (e.g., class weighting, sampling) based on observed counts - `compute_class_weights` yields class-balancing weights for loss functions.
- [x] Draft Chex tests covering dataset output shapes, dtypes, and deterministic behavior under fixed seeds — see `src/tests.py::test_data_batch_shapes` and `test_augmentations_deterministic_with_seed`.

`src/model.py` Subplan
----------------------

- [x] Specify ResNet backbone configuration (CIFAR-style stem: 3x3 stride-1, no maxpool; stage depths 3-4-6-3) based on chosen depth - implemented via `ResNetConfig` and staged loops in `src/model.py`.
- [x] Implement residual blocks with bottleneck/basic variants using Flax modules and sanity-check parameter counts - `BasicBlock` and `BottleneckBlock` now define projection-aware modules.
- [x] Expose depth/width configurations (ResNet-18 vs ResNet-34) through shared module factory - `create_resnet` factory selects block counts with width multipliers.
- [x] Integrate adaptive pooling and classifier head sized to number of emotion classes within the JAX module stack; support optional projection to 3-channel inputs for transfer scenarios - `ResNet.__call__` performs global average pooling, optional dropout, classifier head, and input projection when configured.
- [x] Add options for pretrained initialization and fine-tuning strategies compatible with JAX checkpoints - checkpoint loading via `maybe_load_pretrained_params` and optimizer masks from `build_finetune_mask`.
- [x] Expose forward pass hooks for feature extraction or attention visualization as needed - `return_features=True` returns intermediate feature maps.
- [x] Plan Chex module tests to confirm residual block shape/parameter invariants - ResNet unit tests now validate logits/features and finetuning masks in `src/tests.py`.

`src/train.py` Subplan
----------------------

- [x] Define JIT-compiled training/eval step functions with mixed precision support and gradient accumulation toggles, leveraging Flax `TrainState` - `build_train_step`/`build_eval_step` provide JIT-run logic; mixed precision via `DynamicScale`, gradient accumulation handled with `optax.MultiSteps`.
- [x] Configure Optax optimizer (AdamW or SGD w/ momentum), learning rate schedule, and weight decay - `create_optimizer` wires AdamW + optional masking and multi-step; `create_learning_rate_schedule` supplies warmup/cosine decay.
- [x] Implement checkpointing, early stopping, and best-model tracking using JAX-compatible serialization - `save_checkpoint` persists `TrainState`, prunes old checkpoints, and patience-based early stopping tracks best validation loss.
- [x] Compute evaluation metrics per epoch plus detailed validation logging using MetraX - accuracy computed via `metrax.Accuracy` within train/eval steps; TensorBoard scalars record loss/accuracy each epoch and log per-epoch confusion matrices.
- [x] Add support for test-time evaluation / inference over holdout split using pure JAX/NumPy functions - `predict_batches` runs inference across test iterator and reports final accuracy.
- [x] Prepare Chex-based training step tests (e.g., gradient/NaN checks, tree alignment) for later automation - see `src/tests.py::test_train_step_gradients_finite`.
- [x] Integrate TensorBoard summary writing for loss/metric curves and confusion matrix visualizations - `SummaryWriter` logs step/epoch metrics; confusion-matrix support pending once metrics exist.

`src/main.py` Subplan
---------------------

- [x] Parse CLI/Config file for hyperparameters, dataset paths, and runtime options - `src/main.py` loads JSON configs with CLI overrides for seeds, epochs, resume paths, and output destinations.
- [x] Initialize deterministic seeds, device selection, and logging directories - `prepare_environment` seeds Python/NumPy/JAX; timestamped run directories created under the selected output root.
- [x] Wire together data module, model, and trainer components while managing JAX PRNG keys and functional patterns - CLI resolves `TrainingConfig`/`DataModuleConfig` instances before invoking `train_and_evaluate`.
- [x] Handle experiment lifecycle: train -> validate -> test, with optional resume flag - main delegates to `train_and_evaluate`, forwarding resume checkpoints and reporting train/val/test metrics.
- [x] Surface concise experiment summary (metrics, checkpoints) to stdout/log file - `summarize` prints final losses/accuracies plus checkpoint directory.
- [x] Centralize hyperparameter/training configuration management using Flax config utilities or dataclasses - runtime config built from dataclasses with serialization back to JSON.
- [x] Ensure runtime configuration validated via Pydantic models - `src/main.py` now validates JSON/CLI payloads through `RuntimeTrainingModel`/`RuntimeDataModel` before dataclass construction.
- [x] Persist config/metrics artifacts (JSON + TensorBoard) per run directory for reproducibility - resolved config and metric history written to `<run>/config_resolved.json` and `<run>/metrics.json`.

`src/tests.py` Subplan
------------------

- [x] Design Pydantic schemas for experiment configuration, dataset metadata, and training hyperparameters — `TrainingConfigSchema` in `src/tests.py`.
- [x] Implement Chex shape/dtype tests for data loaders, augmentations, and batching utilities — `src/tests.py::test_data_batch_shapes`, `test_data_normalization_stats`, and `test_augmentations_deterministic_with_seed`.
- [x] Add Chex module/property tests verifying ResNet block outputs, parameter trees, and initialization behavior - see `src/tests.py::test_resnet_forward_and_features`, `test_build_finetune_mask_respects_freeze_directives`, and `test_maybe_load_pretrained_params_roundtrip`.
- [x] Create Chex-assisted training step assertions (loss finite, gradients not NaN/Inf, optimizer state structure) - covered by \src/tests.py::test_train_step_gradients_finite\.
- [x] Validate evaluation metrics integration by comparing MetraX outputs against handcrafted samples - `src/tests.py::test_metrax_accuracy_metric`.
- [x] Provide CLI entry or pytest-style harness (e.g., `uv run pytest src/tests.py`) to run targeted JIT-safe tests without side effects - project now uses pytest via `uv run python -m pytest src/tests.py`.
- [x] Testing notes - current pytest suite (`uv run python -m pytest src/tests.py`) covers data loaders, augmentation determinism, ResNet forward shapes, finetuning masks, checkpoint round-trips, confusion-matrix utilities, and training-step gradient sanity checks.

Next Test Enhancements
----------------------

- [ ] Move tests into dedicated `tests/` package to mirror production modules and ease discovery.
- [ ] Introduce additional integration tests covering end-to-end CLI flows (`src/main.py`) and train/eval loops for higher coverage.
- [ ] Increase unit test granularity across `src/train.py` (scheduler, checkpointing, logging) and `src/main.py` (serialization, overrides) to reach 100% coverage.
- [ ] Add regression tests for data augmentation edge cases (extreme scales, disabled augment, cached statistics) to cover remaining branches.
- [ ] Track coverage progress via `pytest-cov` in CI, gating the suite on 100% statement coverage.

