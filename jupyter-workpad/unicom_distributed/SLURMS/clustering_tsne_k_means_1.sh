#!/bin/bash
#SBATCH --job-name=dist_cluster
#SBATCH --partition=gpu             # GPU partition
#SBATCH --nodes=4                   # Request 4 nodes
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=8           # 8 CPUs per task for data loading/processing
#SBATCH --gres=gpu:a100:1           # Request 1 A100 GPU per node
#SBATCH --mem=128gb                 # Memory per node
#SBATCH --time=48:00:00             # Maximum runtime
#SBATCH --output=dist_cluster_%j.log
#SBATCH --mail-type=END,FAIL

# Print job information
echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"

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
COLUMN_NAME="porto_suggested_cluster_exp_1"
LOG_DIR="/home/tdeatherage3.gatech/logs"
MIN_SAMPLES=25
MIN_K=2
MAX_K=10
TSNE_DIMS=2
OUTLIER_THRESHOLD=2.0
STATS_OUTPUT="clustering_tsne_k_means_1_$(date +%Y%m%d).json"

# Set CUDA visible devices to match local GPU index
export CUDA_VISIBLE_DEVICES=0

# Set master port for PyTorch distributed
export MASTER_PORT=12355

# Set up environment variables for BLASes to use appropriate number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the distributed clustering script using srun
srun python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/clustering_tsne_k_means_1.py \
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
    --use-gpu

echo "Job completed at $(date)"