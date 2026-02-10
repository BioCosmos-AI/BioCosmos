#!/bin/bash
#SBATCH --job-name=multi_cluster_full
#SBATCH --partition=gpu             # GPU partition
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # 8 CPUs per task for data loading/processing
#SBATCH --gres=gpu:a100:1           # Request 1 A100 GPU
#SBATCH --mem=512gb                 # Memory for the node
#SBATCH --time=14-00:00:00           # Maximum runtime (14 days)
#SBATCH --output=multi_cluster_full%j.log
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90
#SBATCH --requeue                   # Allow the job to be requeued
#SBATCH --open-mode=append          # Append to output files if restarted

# Print job information
echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Current directory: $(pwd)"

# Load required modules
module load conda

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Set path to conda environment
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

# Activate conda environment
conda activate ${CONDA_ENV_PATH}
echo "Conda environment activated: $CONDA_ENV_PATH"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set variables
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings_backup.sqlite"
TABLE_NAME="image_embeddings"
LOG_DIR="$HOME/logs/multi_cluster_full_${SLURM_JOB_ID}"
STATS_OUTPUT="clustering_stats_multi_cluster_full_${SLURM_JOB_ID}.json"
MIN_SAMPLES=25
MIN_K=2
MAX_K=10
HDBSCAN_MIN_CLUSTER_SIZE=25
TSNE_DIMS=2
OUTLIER_THRESHOLD=2.0

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/checkpoints

# Set CUDA visible devices to match local GPU index
export CUDA_VISIBLE_DEVICES=0

# Set up environment variables for BLASes to use appropriate number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check if required Python packages are available
echo "Checking required Python packages:"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"
python -c "import cuml; print('cuML version:', cuml.__version__)"
python -c "import scipy; print('SciPy version:', scipy.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
if pip show hdbscan >/dev/null 2>&1; then
    echo "HDBSCAN version: $(pip show hdbscan | grep Version | cut -d ' ' -f 2)"
else
    echo "HDBSCAN not installed"
fi

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

# Define column names for each clustering method
KMEANS_COL="kmeans_cluster"
KMEANS_SCORE_COL="kmeans_silhouette_score"
HIER_SIL_COL="hierarchical_silhouette_cluster"
HIER_SIL_SCORE_COL="hierarchical_silhouette_score"
HIER_GAP_COL="hierarchical_gap_cluster"
HIER_GAP_SCORE_COL="hierarchical_gap_silhouette_score"
HDBSCAN_COL="hdbscan_cluster"
HDBSCAN_SCORE_COL="hdbscan_silhouette_score"

# Run the multi-method clustering script
echo "Running multiple clustering methods script"
python "$HOME/unicom/embedding_and_clustering/multi_cluster_full.py" \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --kmeans-column "$KMEANS_COL" \
    --kmeans-score-column "$KMEANS_SCORE_COL" \
    --hier-sil-column "$HIER_SIL_COL" \
    --hier-sil-score-column "$HIER_SIL_SCORE_COL" \
    --hier-gap-column "$HIER_GAP_COL" \
    --hier-gap-score-column "$HIER_GAP_SCORE_COL" \
    --hdbscan-column "$HDBSCAN_COL" \
    --hdbscan-score-column "$HDBSCAN_SCORE_COL" \
    --log-dir "$LOG_DIR" \
    --min-samples-for-clustering "$MIN_SAMPLES" \
    --min-k "$MIN_K" \
    --max-k "$MAX_K" \
    --hdbscan-min-cluster-size "$HDBSCAN_MIN_CLUSTER_SIZE" \
    --tsne-dims "$TSNE_DIMS" \
    --outlier-threshold "$OUTLIER_THRESHOLD" \
    --stats-output "$STATS_OUTPUT" \
    $RESUME_FLAG

RETURN_CODE=$?
if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    exit $RETURN_CODE
fi

# Check if stats file was created
if [ -f "$LOG_DIR/$STATS_OUTPUT" ]; then
    echo "Clustering statistics file created successfully"
    # Print summary info
    echo "Stats file size: $(du -h $LOG_DIR/$STATS_OUTPUT | cut -f1)"
else
    echo "WARNING: Clustering statistics file not found"
fi

echo "Job completed at $(date)"
echo "Results are available in the database and in $LOG_DIR/$STATS_OUTPUT"

# Exit with the same code as the Python script
exit $RETURN_CODE