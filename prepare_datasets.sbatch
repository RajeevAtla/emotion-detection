#!/bin/bash
#SBATCH --job-name=prepare_datasets
#SBATCH --output=slurm/prepare_datasets_%j.out
#SBATCH --error=slurm/prepare_datasets_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================================
# Dataset Preparation Script for Emotion Detection
# ============================================================================
# 
# This script preprocesses CK+ and RAF-DB datasets to match FER-2013 format.
#
# BEFORE RUNNING:
# 1. Download CK+ from Kaggle: https://www.kaggle.com/datasets/shawon10/ckplus
# 2. Download RAF-DB from Kaggle: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset
# 3. Extract them to the paths below (or update the paths)
#
# Usage:
#   sbatch prepare_datasets.sbatch
#
# ============================================================================

# Exit on error
set -e

# Create output directory for logs
mkdir -p slurm

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo ""
echo "Setting up Python environment..."

# Load Python module (try different versions available on cluster)
module purge
module load python/3.9 2>/dev/null || \
module load python/3.8 2>/dev/null || \
module load python/3.7 2>/dev/null || \
module load python 2>/dev/null || true

# Print Python version
echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"

# Create virtual environment if it doesn't exist
VENV_DIR="$HOME/.venvs/dataset_prep"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "Installing dependencies..."
pip install numpy pillow

# Verify installations
echo ""
echo "Verifying installations:"
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python3 -c "from PIL import Image; import PIL; print(f'  pillow: {PIL.__version__}')"
echo ""

# ============================================================================
# CONFIGURE THESE PATHS
# ============================================================================

# Project root directory
PROJECT_DIR="/home/vss54/emotion/emotion-detection"

# Script location
SCRIPT="$PROJECT_DIR/scripts/prepare_datasets.py"

# Where you extracted the raw downloaded datasets
CKPLUS_RAW="$PROJECT_DIR/data/ckplustemp"
RAFDB_RAW="$PROJECT_DIR/data/rafdbtemp"

# Where to save processed datasets (should match your training config)
DATA_DIR="$PROJECT_DIR/data"
CKPLUS_OUT="$DATA_DIR/ckplus"
RAFDB_OUT="$DATA_DIR/rafdb"
FER2013_DIR="$DATA_DIR/fer2013"

# Image size (48 matches FER-2013)
IMAGE_SIZE=48

# ============================================================================
# VERIFY FER-2013 EXISTS
# ============================================================================

echo ""
echo "============================================"
echo "Verifying FER-2013 dataset..."
echo "============================================"

if [ -d "$FER2013_DIR" ]; then
    python3 "$SCRIPT" --dataset verify --input "$FER2013_DIR"
else
    echo "WARNING: FER-2013 not found at $FER2013_DIR"
    echo "Make sure your existing dataset is in place."
fi

# ============================================================================
# PROCESS CK+ DATASET
# ============================================================================

echo ""
echo "============================================"
echo "Processing CK+ dataset..."
echo "============================================"

if [ -d "$CKPLUS_RAW" ]; then
    python3 "$SCRIPT" \
        --dataset ckplus \
        --input "$CKPLUS_RAW" \
        --output "$CKPLUS_OUT" \
        --size "$IMAGE_SIZE"
    
    echo ""
    echo "CK+ verification:"
    python3 "$SCRIPT" --dataset verify --input "$CKPLUS_OUT"
else
    echo "WARNING: CK+ raw data not found at $CKPLUS_RAW"
    echo "Download from: https://www.kaggle.com/datasets/shawon10/ckplus"
    echo "Extract to: $CKPLUS_RAW"
fi

# ============================================================================
# PROCESS RAF-DB DATASET
# ============================================================================

echo ""
echo "============================================"
echo "Processing RAF-DB dataset..."
echo "============================================"

if [ -d "$RAFDB_RAW" ]; then
    # RAF-DB uses numbered folders (1-7) for emotions
    # Mapping: 1=surprise, 2=fear, 3=disgust, 4=happy, 5=sad, 6=angry, 7=neutral
    
    echo "Converting RAF-DB numbered folders to named emotion folders..."
    
    # Just copy the whole thing first, then rename
    echo "Copying RAF-DB dataset..."
    cp -r "$RAFDB_RAW" "$RAFDB_OUT"
    
    # Rename numbered folders to emotion names
    for split in train test; do
        if [ -d "$RAFDB_OUT/$split" ]; then
            echo "Renaming folders in $split..."
            cd "$RAFDB_OUT/$split"
            
            [ -d "1" ] && mv "1" "surprised"
            [ -d "2" ] && mv "2" "fearful"
            [ -d "3" ] && mv "3" "disgusted"
            [ -d "4" ] && mv "4" "happy"
            [ -d "5" ] && mv "5" "sad"
            [ -d "6" ] && mv "6" "angry"
            [ -d "7" ] && mv "7" "neutral"
            
            cd - > /dev/null
        fi
    done
    
    echo ""
    echo "RAF-DB verification:"
    python3 "$SCRIPT" --dataset verify --input "$RAFDB_OUT"
else
    echo "WARNING: RAF-DB raw data not found at $RAFDB_RAW"
    echo "Download from: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset"
    echo "Extract to: $RAFDB_RAW"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================"
echo "Dataset Preparation Complete"
echo "============================================"
echo ""
echo "Dataset locations:"
echo "  FER-2013: $FER2013_DIR"
echo "  CK+:      $CKPLUS_OUT"
echo "  RAF-DB:   $RAFDB_OUT"
echo ""
echo "Next steps:"
echo "  1. Update configs/multi_dataset.toml with correct paths"
echo "  2. Run training: sbatch gpu_efficientnet_multi.sbatch"
echo ""
echo "End time: $(date)"
echo "============================================"
