#!/bin/bash
# Workflow script to run face preprocessing followed by training
# 
# Usage:
#   ./run_workflow.sh [input_data] [experiment_name]
#
# Example:
#   ./run_workflow.sh data my-experiment
#
# This will:
#   1. Submit the preprocessing job to detect/align faces
#   2. Wait for preprocessing to complete
#   3. Submit the training job using preprocessed data

set -euo pipefail

INPUT_DATA="${1:-data}"
EXPERIMENT_NAME="${2:-emotion-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DATA="data_processed_${EXPERIMENT_NAME}"
NUM_EPOCHS="${3:-50}"

echo "=========================================="
echo "Emotion Detection Workflow"
echo "=========================================="
echo "Input data:     ${INPUT_DATA}"
echo "Output data:    ${OUTPUT_DATA}"
echo "Experiment:     ${EXPERIMENT_NAME}"
echo "Epochs:         ${NUM_EPOCHS}"
echo "=========================================="
echo ""

# Check input data exists
if [ ! -d "${INPUT_DATA}" ]; then
    echo "ERROR: Input data directory '${INPUT_DATA}' not found!"
    exit 1
fi

# Create slurm log directory
mkdir -p slurm

echo "Step 1: Submitting preprocessing job..."
PREPROCESS_JOB=$(sbatch --parsable preprocess_faces.sbatch "${INPUT_DATA}" "${OUTPUT_DATA}")
echo "  Preprocessing job ID: ${PREPROCESS_JOB}"

echo ""
echo "Step 2: Submitting training job (will wait for preprocessing)..."

# Create a temporary config with the correct data directory
TEMP_CONFIG="configs/workflow_${EXPERIMENT_NAME}.toml"
cat > "${TEMP_CONFIG}" << EOF
# Auto-generated config for workflow: ${EXPERIMENT_NAME}

[training]
output_dir = "runs"
model_depth = 34
width_multiplier = 1
dropout_rate = 0.0
num_epochs = ${NUM_EPOCHS}
batch_size = 256
learning_rate = 0.0006
min_learning_rate = 0.00001
warmup_epochs = 5
weight_decay = 0.0001
gradient_accumulation_steps = 1
label_smoothing = 0.1
log_every = 25
log_to_console = true
checkpoint_every = 5
max_checkpoints = 3
use_mixed_precision = true
patience = 15
experiment_name = "${EXPERIMENT_NAME}"

[training.data]
data_dir = "${OUTPUT_DATA}"
batch_size = 256
val_ratio = 0.1
augment = true

[training.data.augmentation]
horizontal_flip_prob = 0.5
rotation_degrees = 15.0
scale_range = [0.9, 1.1]
elastic_blur_sigma = 0.0
enabled = true

[training.data.insightface]
enabled = false
EOF

echo "  Created config: ${TEMP_CONFIG}"

# Submit training job with dependency on preprocessing
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB} \
    train_optimized.sbatch "${OUTPUT_DATA}" "${TEMP_CONFIG}" "${EXPERIMENT_NAME}" "${NUM_EPOCHS}")
echo "  Training job ID: ${TRAIN_JOB}"

echo ""
echo "=========================================="
echo "Workflow submitted!"
echo "=========================================="
echo ""
echo "Job chain:"
echo "  1. Preprocess (${PREPROCESS_JOB}): Running face detection"
echo "  2. Train (${TRAIN_JOB}): Waiting for preprocessing"
echo ""
echo "Monitor with:"
echo "  squeue -u \${USER}"
echo ""
echo "View logs:"
echo "  tail -f slurm/preprocess_${PREPROCESS_JOB}.out"
echo "  tail -f slurm/train_${TRAIN_JOB}.out"
echo ""
echo "Cancel workflow:"
echo "  scancel ${PREPROCESS_JOB} ${TRAIN_JOB}"
