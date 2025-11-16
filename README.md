# Emotion Detection with JAX ResNet

This repository implements a complete emotion-recognition pipeline using JAX/Flax.
It covers data ingestion, training, evaluation, and reproducible experimentation for
facial expression classification on the FER-style 48x48 grayscale dataset.

---

## Features

- **JAX/Flax NNX ResNet**: configurable CIFAR-style ResNet-18/34 backbones with fine-tuning support. All modules now use Flax NNX state semantics (no Linen `apply_fn`), so you can control train/eval by calling `module.train()` / `module.eval()`.
- **Data Module**: deterministic preprocessing, augmentation, and stratified splitting.
- **Training Loop**: Fully NNX-native train/eval steps powered by Optax, mixed precision, checkpointing, early stopping, and TensorBoard logging.
- **Checkpointing Helpers**: Orbax-based checkpoint saves with automatic pruning plus a shim that reverses stringified integer keys when resuming (workaround for google/orbax#2561 and aligned with the Flax NNX migration plan).
- **Metrics**: Accuracy, F1, macro-F1, and confusion matrices via MetraX.
- **Testing**: Extensive pytest/Chex coverage (unit + integration) with 100% statement coverage.
- **Tooling**: Managed by `uv` for reproducible dependency resolution.

---

## Quick Start

### Prerequisites

- Python **3.13.x** (the project pins `pyproject.toml` to `==3.13.*`; install via `uv python install 3.13` or your preferred environment manager).
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv#installation)).

### 1. Clone & Enter

```bash
git clone https://github.com/RajeevAtla/emotion-detection.git
cd emotion-detection
```

### 2. Install Dependencies (uv + venv)

```bash
pip install uv
uv python install 3.13
uv sync
```

Run `uv sync --group cuda` instead when you specifically want the CUDA-enabled
`jax[cuda12]` wheels (skip the group on CPU-only machines or on systems without
CUDA drivers).
(Only runs on Linux).

### 3. Prepare Dataset

Place FER-style data under `data/` with the following structure:

```
data/
  train/
    happy/
      img0.png
      ...
    sad/
      ...
  test/
    happy/
    sad/
    ...
```

Each class directory should use one of the canonical labels (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`). Files may be PNG/JPG/JPEG; the loader always converts them to single-channel float32 tensors and assumes the FER 48x48 resolution (no automatic resizing beyond the augmentation pipeline), so keep inputs at that size. The training split automatically produces a stratified validation set controlled by `data.val_ratio` (10% by default).

---

## Running Training

`configs/example.toml` mirrors the CLI schema that `src.main` consumes. Each config defines a top-level `[training]` table with nested `[training.data]` and `[training.data.augmentation]` sections—update `training.data.data_dir` plus any hyperparameters before running if your setup differs from the defaults.

```bash
uv run python -m src.main \
  --config configs/example.toml \
  --output-dir runs \
  --seed 42 \
  --experiment-name baseline
```

Key configuration options (via TOML or CLI overrides):

- `data.data_dir`: path to dataset.
- `model_depth`: `18` or `34`.
- `num_epochs`, `batch_size`, `learning_rate`, `warmup_epochs`.
- `use_mixed_precision`: enable float16 training on compatible accelerators.
- `pretrained_checkpoint` / `resume_checkpoint`: point at `checkpoints/epoch_XXXX`
  directories written by Orbax to warm start or resume training.
- `freeze_stem`, `freeze_classifier`, `frozen_stages`: compatible with the new `build_finetune_mask` traversal over `nnx.state(model)` so you can freeze arbitrary blocks while keeping optimizer masking in sync.

Training outputs:

- A timestamped run directory `<output-root>/<timestamp>` (or `<timestamp-experiment>` when `--experiment-name` is set).
- `checkpoints/` containing Orbax snapshots (best validation checkpoints are automatically reloaded before final testing).
- `tensorboard/` with scalar curves, micro/macro-F1 traces, and confusion-matrix summaries.
- `config_resolved.toml` and `metrics.toml` capturing the exact hyperparameters and the per-epoch history (accuracy, micro/macro-F1, per-class F1 text payloads, etc.).

### Checkpointing & Resuming

Checkpoint IO is centralized in `src/checkpointing.py`, which keeps the latest
`max_checkpoints` directories (default three) and now targets Flax NNX modules,
optimizers, and RNG containers. Orbax currently stringifies integer dict keys
when restoring (see [google/orbax#2561](https://github.com/google/orbax/issues/2561)),
so the helper automatically converts those keys back to integers before the state
is consumed or passed into `nnx.update`. To resume a run, either set
`training.resume_checkpoint` in your TOML or pass
`--resume /path/to/checkpoints/epoch_XXXX` on the CLI -- the helper will pick up
the payload (model params, optimizer PyTrees, RNGs, dynamic-scale state), restore
everything into the current `TrainState`, and trim any stale checkpoints after the next
save. Set `training.pretrained_checkpoint` to a matching directory when you only
need the model weights (the optimizer state is reinitialized for fine-tuning).

---

## Testing & Linting

All automation uses uv:

```bash
uv run ruff check
uv run ruff format
uv run ty check src
uv run pytest --cov=src
```

The pytest suite mirrors every major NNX code path (model modules, optimizer masking,
checkpoint round-trips, CLI integration) and enforces **100%** statement coverage.
For a quick local run you can also invoke the convenience harness:

```bash
uv run python scripts/run_tests.py --cov
```

### Smoke CI Workflow

The GitHub smoke workflow stages a synthetic FER dataset inside the runner's
temporary directory before kicking off a one-epoch training run. This test hits
the exact Flax NNX codepath (model state + optimizer + RNG streams) used for
full experiments, so reproducing it locally is the fastest way to sanity-check
the migration.

**Important:** The workflow intentionally recreates its staging directory from scratch. Do **not** point it at your real FER dataset. When running the smoke scenario locally:

1. Place your production dataset outside this repository (or keep a separate backup).
2. Create a scratch directory (for example `RUNNER_TEMP=$(mktemp -d)`).
3. Copy `configs/smoke.toml` into that scratch directory and edit `data.data_dir` to reference the scratch path.
4. Populate the scratch directory with a handful of tiny grayscale images per class (one or two is enough).
5. Execute `uv run python -m src.main --config <scratch>/smoke.toml --output-dir runs --seed 0 --experiment-name smoke-local`.

Following these steps mirrors the CI behaviour while ensuring the repository's `data/` folder-and your real dataset-remain untouched.

Ruff enforces an 80-character max line length (see `pyproject.toml`).
Run `uv run ruff format` before committing to keep the repo consistent.

These mirror the GitHub Actions workflow located in `.github/workflows/ci.yml`.

---

## Project Structure

```
configs/
  example.toml    # Baseline configuration referenced in the README
  smoke.toml      # CI smoke-test configuration (patch its paths before use)
scripts/
  run_tests.py    # Convenience entry point for pytest/coverage
src/
  data.py      # Data loading/augmentation utilities
  model.py     # ResNet architectures and helpers
  train.py     # Training loop, checkpointing, evaluation
  main.py      # CLI entry point
tests/
  test_data.py
  test_model.py
  test_train.py
  test_main.py
```

---

## CI Pipeline

The GitHub Actions workflow performs:

1. `uv sync --dev`
2. `uv run ruff check`
3. `uv run ty check src`
4. `uv run pytest --cov=src`

## License

MIT — see [LICENSE](LICENSE).
