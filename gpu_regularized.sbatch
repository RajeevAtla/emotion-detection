#!/bin/bash
#SBATCH --job-name=emotion-regularized
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[017,018]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/train_%j.out
#SBATCH --error=slurm/train_%j.err

echo "========================================================================"
echo "EMOTION DETECTION - REGULARIZED TRAINING (Anti-Overfitting)"
echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"
echo ""
echo "Key changes from previous run:"
echo "  - ResNet-18 instead of ResNet-34 (smaller model)"
echo "  - Dropout: 0.5 (was 0.1)"
echo "  - Weight decay: 0.01 (was 0.0001)"
echo "  - Label smoothing: 0.2 (was 0.1)"
echo "  - Learning rate: 0.0003 (was 0.001)"
echo "  - Stronger augmentation"
echo "========================================================================"

mkdir -p slurm

module purge
module use /projects/community/modulefiles
module load apptainer
module load git/2.35.1-ez82

mkdir -p /scratch/${USER}/apptainer
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer

echo ""
echo "Current git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo ""

JAX_IMAGE=jax_25.10-py3.sif

if [ ! -f "${JAX_IMAGE}" ]; then
  echo "Pulling JAX container..."
  apptainer pull "${JAX_IMAGE}" docker://nvcr.io/nvidia/jax:25.10-py3
else
  echo "Using cached container: ${JAX_IMAGE}"
fi

echo ""
echo "Starting training container..."
echo ""

apptainer exec --nv \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  --env TF_FORCE_GPU_ALLOW_GROWTH=true \
  --env TF_GPU_ALLOCATOR=cuda_malloc_async \
  --env JAX_PLATFORMS=cuda \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  "${JAX_IMAGE}" bash -c '
  set -euo pipefail
  
  echo "========================================================================"
  echo "CONTAINER SETUP"
  echo "========================================================================"
  
  if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  source $HOME/.local/bin/env 2>/dev/null || true
  
  uv sync --group cuda
  
  echo ""
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
  echo ""
  
  uv run python -c "
import jax
print(f\"JAX devices: {jax.devices()}\")
print(f\"Device count: {jax.device_count()}\")
"
  
  echo ""
  echo "========================================================================"
  echo "STARTING REGULARIZED TRAINING"
  echo "========================================================================"
  echo ""
  
  uv run python -m src.main \
    --config configs/regularized_v1.toml \
    --output-dir runs \
    --experiment-name regularized-$(date +%Y%m%d) \
    --seed 42
  
  TRAIN_EXIT=$?
  
  echo ""
  if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ TRAINING COMPLETED SUCCESSFULLY"
  else
    echo "✗ TRAINING FAILED (exit code: $TRAIN_EXIT)"
  fi
  
  exit $TRAIN_EXIT
'

echo ""
echo "Job completed at: $(date)"
exit $?
