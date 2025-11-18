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

mkdir -p /scratch/${USER}/apptainer/
export APPTAINER_CACHEDIR = /scratch/${USER}/apptainer

if [! -e jax_25-py3.sif]; then
  echo "Cached apptainer image not found, pulling new one"
  apptainer pull docker://nvcr.io/nvidia/jax:25.10-py3
else
  echo "Using cached apptainer image"
fi

srun singularity run --nv jax_25.10-py3.sif

