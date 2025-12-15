#!/bin/bash
#SBATCH --job-name=emotion-efficientnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[017,018]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/train_%j.out
#SBATCH --error=slurm/train_%j.err

echo "========================================================================"
echo "EMOTION DETECTION - EfficientNet-B2 Multi-Dataset Training"
echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Model: EfficientNet-B2 (9M params)"
echo "  - Datasets: FER-2013 + CK+ + RAF-DB"
echo "  - GPUs: 4x NVIDIA A100 40GB"
echo "  - Per-GPU Batch Size: 64"
echo "  - Global Batch Size: 256"
echo "  - Mixed Precision: Enabled"
echo "  - Augmentation: Mixup + CutMix + RandAugment"
echo "========================================================================"

mkdir -p slurm

# Module setup
module purge
module use /projects/community/modulefiles
module load apptainer
module load git/2.35.1-ez82

# Cache directories
mkdir -p /scratch/${USER}/apptainer
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer

echo ""
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo ""

# Container
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
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  --env TF_FORCE_GPU_ALLOW_GROWTH=true \
  --env TF_GPU_ALLOCATOR=cuda_malloc_async \
  --env JAX_PLATFORMS=cuda \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "${JAX_IMAGE}" bash -c '
  set -euo pipefail
  
  echo "========================================================================"
  echo "CONTAINER SETUP"
  echo "========================================================================"
  
  # Install uv if needed
  if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  source $HOME/.local/bin/env 2>/dev/null || true
  
  # Install dependencies
  uv sync --group cuda
  
  echo ""
  echo "GPU Information:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
  echo ""
  
  # Verify JAX sees all GPUs
  uv run python -c "
import jax
print(f\"JAX devices: {jax.devices()}\")
print(f\"Device count: {jax.device_count()}\")
print(f\"Local device count: {jax.local_device_count()}\")
"
  
  echo ""
  echo "========================================================================"
  echo "STARTING MULTI-GPU TRAINING"
  echo "========================================================================"
  echo ""
  
  # Run training
  uv run python -m src.main \
    --config configs/multi_dataset.toml \
    --output-dir runs \
    --experiment-name efficientnet-$(date +%Y%m%d) \
    --seed 42
  
  TRAIN_EXIT=$?
  
  echo ""
  if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ TRAINING COMPLETED SUCCESSFULLY"
    
    # Find and display final metrics
    LATEST_RUN=$(ls -td runs/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ] && [ -f "${LATEST_RUN}/metrics.toml" ]; then
      echo ""
      echo "Final Metrics:"
      cat "${LATEST_RUN}/metrics.toml"
    fi
  else
    echo "✗ TRAINING FAILED (exit code: $TRAIN_EXIT)"
  fi
  
  exit $TRAIN_EXIT
'

echo ""
echo "Job completed at: $(date)"
exit $?
