#!/bin/bash
#SBATCH --job-name=run-emotion-detection
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[017,018]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err

mkdir -p slurm

module purge
module use /projects/community/modulefiles
module load apptainer

# set cache dir to scratch so we don't run out of space
mkdir -p /scratch/${USER}/apptainer
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer

JAX_IMAGE=jax_25.10-py3.sif

# pull jax container if not pulled
if [ ! -f "${JAX_IMAGE}" ]; then
  echo "Cached apptainer image not found, pulling new one"
  apptainer pull "${JAX_IMAGE}" docker://nvcr.io/nvidia/jax:25.10-py3
else
  echo "Using cached apptainer image"
fi

# load container once it's been pulled
apptainer exec --nv "${JAX_IMAGE}" bash -c '
  set -euo pipefail
  source .venv/bin/activate
  if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --upgrade uv
  fi
  uv sync --group cuda
  uv run pytest --cov=src
'

