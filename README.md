# Emotion Detection with JAX ResNet

This repository implements a complete emotion-recognition pipeline using JAX/Flax.
It covers data ingestion, training, evaluation, and reproducible experimentation for
facial expression classification on the FER-style 48x48 grayscale dataset.

---

## Features
- **JAX/Flax ResNet**: configurable CIFAR-style ResNet-18/34 backbones with fine-tuning support.
- **Data Module**: deterministic preprocessing, augmentation, and stratified splitting.
- **Training Loop**: Optax optimizers, mixed precision, checkpointing, early stopping, and TensorBoard logging.
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
Images must be 48x48 grayscale PNGs. The train split is used to create an internal validation set.

---

## Running Training
```bash
uv run python -m src.main \
  --config configs/example.json \
  --output-dir runs \
  --seed 42 \
  --experiment-name baseline
```

Key configuration options (via JSON or CLI overrides):
- `data.data_dir`: path to dataset.
- `model_depth`: `18` or `34`.
- `num_epochs`, `batch_size`, `learning_rate`, `warmup_epochs`.
- `use_mixed_precision`: enable float16 training on compatible accelerators.

Training outputs:
- Checkpoints under `<output-dir>/<timestamp>/<experiment>/checkpoints/`.
- TensorBoard logs under `<output-dir>/<timestamp>/<experiment>/tensorboard/`.
- JSON summaries (`config_resolved.json`, `metrics.json`).
- Metrics include validation/test accuracy, micro-F1, and macro-F1 plus
  per-class F1 breakdown text in TensorBoard.

---

## Testing & Linting

All automation uses uv:
```bash
uv tool run ruff check
uv tool run ruff format
uv tool run ty check src
uv run pytest --cov=src
```

For a quick local run you can also invoke the convenience harness:
```bash
uv run python scripts/run_tests.py --cov
```

### Smoke CI Workflow

The GitHub smoke workflow stages a synthetic FER dataset inside the runner's temporary directory before kicking off a one-epoch training run.

**Important:** The workflow intentionally recreates its staging directory from scratch. Do **not** point it at your real FER dataset. When running the smoke scenario locally:

1. Place your production dataset outside this repository (or keep a separate backup).
2. Create a scratch directory (for example `RUNNER_TEMP=$(mktemp -d)`).
3. Copy `configs/smoke.json` into that scratch directory and edit `data.data_dir` to reference the scratch path.
4. Populate the scratch directory with a handful of tiny grayscale images per class (one or two is enough).
5. Execute `uv run python -m src.main --config <scratch>/smoke.json --output-dir runs --seed 0 --experiment-name smoke-local`.

Following these steps mirrors the CI behaviour while ensuring the repository’s `data/` folder—and your real dataset—remain untouched.

Ruff enforces a 79-character max line length.
Run `uv tool run ruff format` before committing to keep the repo consistent.

These mirror the GitHub Actions workflow located in `.github/workflows/ci.yml`.

---

## Project Structure
```
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
planning.md     # High-level roadmap & completed tasks
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


