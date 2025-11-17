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

srun singularity run --nv jax_25.10-py3.sif

