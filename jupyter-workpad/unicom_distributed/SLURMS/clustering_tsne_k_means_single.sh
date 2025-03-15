#!/bin/bash
#SBATCH --job-name=single_cluster
#SBATCH --partition=gpu             # GPU partition
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # 8 CPUs per task for data loading/processing
#SBATCH --gres=gpu:a100:1           # Request 1 A100 GPU
#SBATCH --mem=128gb                 # Memory for the node
#SBATCH --time=48:00:00             # Maximum runtime
#SBATCH --output=single_cluster_%j.log
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
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"
TABLE_NAME="image_embeddings" 
COLUMN_NAME="single_node_cluster"
LOG_DIR="/home/tdeatherage3.gatech/logs/clustering_single_${SLURM_JOB_ID}"
MIN_SAMPLES=25
MIN_K=2
MAX_K=10
TSNE_DIMS=2
OUTLIER_THRESHOLD=2.0
STATS_OUTPUT="clustering_stats_single_${SLURM_JOB_ID}.json"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/checkpoints

# Set CUDA visible devices to match local GPU index
export CUDA_VISIBLE_DEVICES=0

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

# Run the single-node clustering script
python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/clustering_tsne_k_means_single.py \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --column-name "$COLUMN_NAME" \
    --log-dir "$LOG_DIR" \
    --min-samples-for-clustering "$MIN_SAMPLES" \
    --min-k "$MIN_K" \
    --max-k "$MAX_K" \
    --tsne-dims "$TSNE_DIMS" \
    --outlier-threshold "$OUTLIER_THRESHOLD" \
    --stats-output "$STATS_OUTPUT" \
    --use-gpu \
    $RESUME_FLAG

RETURN_CODE=$?
if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    exit $RETURN_CODE
fi

echo "Job completed at $(date)"