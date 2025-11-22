#!/bin/bash
#SBATCH --job-name=emotion-2gpu-working
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[017,018]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err

echo "=========================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Local slurm log directory (note: SLURM will still create the stdout/err files itself)
mkdir -p slurm

module purge
module use /projects/community/modulefiles
module load apptainer
module load git/2.35.1-ez82

# -------- Scratch + cache setup on HOST (outside container) --------
mkdir -p /scratch/${USER}/apptainer
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer

export INSIGHTFACE_HOME=/scratch/${USER}/.insightface
mkdir -p "${INSIGHTFACE_HOME}"
echo "InsightFace cache (host): ${INSIGHTFACE_HOME}"
echo ""

JAX_IMAGE=jax_25.10-py3.sif

if [ ! -f "${JAX_IMAGE}" ]; then
  echo "Pulling JAX container..."
  apptainer pull "${JAX_IMAGE}" docker://nvcr.io/nvidia/jax:25.10-py3
else
  echo "Using cached container"
fi

echo ""
echo "Starting container..."
echo ""

# -------- Run everything inside container via heredoc --------
apptainer exec --nv --bind /scratch:/scratch "${JAX_IMAGE}" bash << 'INNERSH'
set -euo pipefail

echo "=== Environment Setup ==="
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_IO_ENABLE_OPENEXR=0
unset DISPLAY || true
echo "✓ Headless mode enabled"
echo ""

echo "INSIGHTFACE_HOME (inside): ${INSIGHTFACE_HOME:-not set}"
echo ""

echo "=== Installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
  # Use bash explicitly for the installer to avoid /bin/sh quirks
  curl -LsSf https://astral.sh/uv/install.sh | bash
  if [ -f "$HOME/.local/bin/env" ]; then
    # env file is created by uv installer
    # shellcheck disable=SC1090
    . "$HOME/.local/bin/env"
  else
    # If env file doesn't exist, add ~/.local/bin to PATH manually
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
echo "✓ uv $(uv --version)"
echo ""

echo "=== Setting Python 3.12 ==="
uv python install 3.12
uv python pin 3.12
PYTHON_VER=$(uv run python --version)
echo "✓ ${PYTHON_VER}"

if [[ "$PYTHON_VER" != *"3.12"* ]]; then
  echo "ERROR: Expected Python 3.12, got ${PYTHON_VER}"
  exit 1
fi
echo ""

echo "=== Syncing Dependencies ==="
uv sync --group cuda
echo "✓ Base dependencies installed"
echo ""

echo "=== Installing InsightFace Stack ==="
uv pip install opencv-python-headless
uv pip install onnxruntime-gpu
uv pip install insightface
echo "✓ InsightFace stack installed"
echo ""

echo "=== Verification ==="
uv run python << "PYEOF"
import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

print(f"Python: {sys.version}")

import cv2
print(f"✓ OpenCV: {cv2.__version__}")

import onnxruntime as ort
print(f"✓ ONNX Runtime: {ort.__version__}")
providers = ort.get_available_providers()
print(f"  Providers: {providers[:2]}")

import insightface
print(f"✓ InsightFace: {insightface.__version__}")

from insightface.app import FaceAnalysis
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
print(f"✓ FaceAnalysis initialized")

import jax
print(f"✓ JAX devices: {jax.devices()}")

print("\n✓ All checks passed!")
PYEOF

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

uv run python -m src.main \
  --config configs/example.toml \
  --output-dir runs \
  --experiment-name 2gpu-insightface-fer2013 \
  --seed 42 \
  --num-epochs 50

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "✓ Training completed successfully!"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "✗ Training failed with exit code: $TRAIN_EXIT"
  echo "=========================================="
fi

exit $TRAIN_EXIT
INNERSH

CONTAINER_EXIT=$?

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "Exit code: $CONTAINER_EXIT"
echo "=========================================="

exit $CONTAINER_EXIT

