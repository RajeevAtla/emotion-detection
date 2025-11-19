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
module load git/2.35.1-ez82

# check apptainer version
apptainer --version

# set cache dir to scratch so we don't run out of space
mkdir -p /scratch/${USER}/apptainer
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer

# get current git commit
echo "Current git commit: "
git rev-parse --short HEAD

JAX_IMAGE=jax_25.10-py3.sif

# pull nvidia's jax container if not pulled
if [ ! -f "${JAX_IMAGE}" ]; then
  echo "Cached apptainer image not found, pulling new one"
  apptainer pull "${JAX_IMAGE}" docker://nvcr.io/nvidia/jax:25.10-py3
else
  echo "Using cached apptainer image"
fi

# load and execute commands on the container once it's been pulled
apptainer exec --nv "${JAX_IMAGE}" bash -c '
  echo "Container started"
  set -euo pipefail
  
  # uv setup
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  echo "uv installed!"
  uv --version
  
  # sync dependencies
  uv sync --group cuda
  
  # check jax+nvidia drivers+cuda are working correctly
  nvidia-smi
  nvcc --version
  uv run python -c "import jax; print(jax.devices())"
  
  # run training
  uv run python -m src.main \
  --config configs/example.toml \
  --output-dir runs \
  --experiment-name gh-full \
  --seed 42 \
  --num-epochs 2
'

