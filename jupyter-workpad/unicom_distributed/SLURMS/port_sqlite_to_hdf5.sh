#!/bin/bash
#SBATCH --job-name=sqlite_to_hdf5
#SBATCH --partition=bigmem          # CPU partition with more memory
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # 8 CPUs per task for data loading/processing
#SBATCH --mem=128gb                 # Memory for the node
#SBATCH --time=24:00:00             # Maximum runtime
#SBATCH --output=sqlite_to_hdf5_%j.log
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90
#SBATCH --requeue                   # Allow the job to be requeued
#SBATCH --open-mode=append          # Append to output files if restarted

# Print job information
echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Load required modules
module load conda

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Set path to conda environment
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

# Activate conda environment
conda activate ${CONDA_ENV_PATH}

# Set variables
SQLITE_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"
HDF5_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.h5"
LOG_DIR="/home/tdeatherage3.gatech/logs/sqlite_to_hdf5_${SLURM_JOB_ID}"
CHUNK_SIZE=100000  # Number of rows to process at once
TABLE_NAME="image_embeddings"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/checkpoints

# Set up environment variables for BLASes to use appropriate number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check if this is a restarted job by looking for checkpoint files
CHECKPOINT_EXISTS=0
if [ -d "${LOG_DIR}/checkpoints" ] && [ "$(ls -A ${LOG_DIR}/checkpoints)" ]; then
    echo "Checkpoint files found. This appears to be a restarted job."
    CHECKPOINT_EXISTS=1
fi

# Add resume flag if checkpoints exist
RESUME_FLAG=""
if [ "$CHECKPOINT_EXISTS" -eq 1 ]; then
    RESUME_FLAG="--resume"
    echo "Will attempt to resume from checkpoint"
fi

# Run the migration script
python /home/tdeatherage3.gatech/sqlite_to_hdf5.py \
    --sqlite-path "$SQLITE_PATH" \
    --hdf5-path "$HDF5_PATH" \
    --table-name "$TABLE_NAME" \
    --log-dir "$LOG_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    $RESUME_FLAG

RETURN_CODE=$?
if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    exit $RETURN_CODE
fi

echo "Job completed at $(date)"