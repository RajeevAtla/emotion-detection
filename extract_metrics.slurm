#!/bin/bash
#SBATCH --job-name=extract-metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=ampere
#SBATCH --output=slurm/extract_%j.out
#SBATCH --error=slurm/extract_%j.err

# ---------------- CONFIG ----------------
# Default to the specific run you mentioned, but allow override via command line
RUN_NAME="${1:-20251127-231051-regularized}"
BASE_DIR="${2:-runs}" 

# Full path to the run
RUN_DIR="/home/vss54/emotion/emotion-detection/runs/20251127-231051-regularized-20251127"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
OUTPUT_CSV="${RUN_DIR}/metrics_export.csv"

CONTAINER_DIR="/scratch/$USER/containers"
CONTAINER_NAME="pytorch_2.1.0-cuda11.8-runtime.sif"
CONTAINER_IMAGE="${CONTAINER_DIR}/${CONTAINER_NAME}"

echo "========== extract-metrics job =========="
echo "Host:   $(hostname)"
echo "Start:  $(date)"
echo "Run:    ${RUN_NAME}"
echo "TB Dir: ${TENSORBOARD_DIR}"
echo "Output: ${OUTPUT_CSV}"
echo "========================================="

# ---------------- MODULES ----------------
module purge
module use /projects/community/modulefiles 2>/dev/null || true

echo "[STEP] Loading singularity module..."
if ! module load singularity 2>/dev/null; then
    echo "ERROR: Could not load singularity module."
    exit 1
fi

# ---------------- PREP PATHS / ENVS ----------------
mkdir -p slurm
mkdir -p "${CONTAINER_DIR}"

export SINGULARITYENV_TENSORBOARD_DIR="${TENSORBOARD_DIR}"
export SINGULARITYENV_OUTPUT_CSV="${OUTPUT_CSV}"
export SINGULARITYENV_USER="${USER}"

# ---------------- CHECK CONTAINER ----------------
if [ ! -f "${CONTAINER_IMAGE}" ]; then
    echo "ERROR: Container ${CONTAINER_IMAGE} not found."
    echo "Please ensure the container is pulled (or use the pull logic from previous script)."
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

# ---------------- PYTHON WORKER SCRIPT ----------------
cat << 'EOF' > _extractor_script.py
import os
import sys
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tf_events(log_dir, output_csv):
    print(f"[Extractor] Reading events from: {log_dir}")
    
    # Initialize EventAccumulator to load all data (scalars)
    # size_guidance=0 loads all events without downsampling
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()

    # Get all scalar tags (e.g., 'Loss/train', 'Accuracy/val')
    tags = event_acc.Tags()['scalars']
    
    if not tags:
        print("[Extractor] No scalar tags found in TensorBoard logs.")
        return

    print(f"[Extractor] Found tags: {tags}")

    # Dictionary to hold dataframes for merging
    data_frames = []

    for tag in tags:
        events = event_acc.Scalars(tag)
        
        # Extract steps and values
        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]
        
        # Create a temp dataframe for this tag
        df = pd.DataFrame({
            'step': steps,
            tag: values,
            # 'wall_time': wall_times # Optional: uncomment if you need timestamps
        })
        
        # Set step as index for easy merging
        df.set_index('step', inplace=True)
        data_frames.append(df)

    # Merge all dataframes on 'step' (outer join to handle different logging frequencies)
    if data_frames:
        final_df = pd.concat(data_frames, axis=1)
        final_df.sort_index(inplace=True)
        final_df.reset_index(inplace=True)
        
        print(f"[Extractor] Saving merged data to {output_csv}")
        print(final_df.head())
        final_df.to_csv(output_csv, index=False)
        print("[Extractor] Success.")
    else:
        print("[Extractor] Failed to aggregate data.")

def main():
    tb_dir = os.environ.get("TENSORBOARD_DIR")
    out_csv = os.environ.get("OUTPUT_CSV")

    if not os.path.exists(tb_dir):
        print(f"ERROR: TensorBoard directory not found: {tb_dir}")
        sys.exit(1)

    extract_tf_events(tb_dir, out_csv)

if __name__ == "__main__":
    main()
EOF

# ---------------- RUN CONTAINER ----------------
echo "[STEP] Starting container + extraction..."

singularity exec \
    --nv \
    --cleanenv \
    --bind /scratch:/scratch \
    "${CONTAINER_IMAGE}" \
    bash -lc '
        echo "[STEP] Installing Pandas & TensorBoard..."
        # We need these to parse the logs
        unset PYTHONNOUSERSITE
        python3 -m pip install --user --no-cache-dir pandas tensorboard

        echo "[STEP] Running extractor..."
        python3 _extractor_script.py
    '

echo "=========================================="
echo "Job finished at $(date)"
echo "Check output at: ${OUTPUT_CSV}"
echo "=========================================="
