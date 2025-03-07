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
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90
#SBATCH --requeue                   # Allow the job to be requeued
#SBATCH --open-mode=append          # Append to output files if restarted

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

# Get the first node to use as master
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Using master node: $MASTER_NODE"
export MASTER_ADDR=$MASTER_NODE

export GLOO_SOCKET_IFNAME=eth0
# Increase timeouts for better stability
export GLOO_TIMEOUT_SECONDS=3600  # 1 hour timeout


# Force IPv4 and set network parameters
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_IP_VERSION=4

export PYTORCH_DISTRIBUTED_SOCKET_TIMEOUT=3600  # 1 hour timeout


# Set variables
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"
TABLE_NAME="image_embeddings" 
COLUMN_NAME="porto_suggested_cluster_exp_1"
LOG_DIR="/home/tdeatherage3.gatech/logs/clustering_job_${SLURM_JOB_ID}"
MIN_SAMPLES=25
MIN_K=2
MAX_K=10
TSNE_DIMS=2
OUTLIER_THRESHOLD=2.0
STATS_OUTPUT="clustering_stats_${SLURM_JOB_ID}.json"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/checkpoints

# Set CUDA visible devices to match local GPU index
export CUDA_VISIBLE_DEVICES=0

# Set master port for PyTorch distributed
export MASTER_PORT=12355

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
    --use-gpu \
    $RESUME_FLAG

RETURN_CODE=$?
if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    exit $RETURN_CODE
fi

echo "Job completed at $(date)"