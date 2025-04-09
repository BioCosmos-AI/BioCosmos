#!/bin/bash
#SBATCH --job-name=unicom_train
#SBATCH --partition=gpu             # GPU partition
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks-per-node=2         # Use 2 GPUs on the node (adjust as needed)
#SBATCH --cpus-per-task=8           # 8 CPUs per GPU for data loading
#SBATCH --gres=gpu:a100:2           # Request 2 A100 GPUs
#SBATCH --mem=500gb                 # Memory for the node (high for SQLite DB in memory)
#SBATCH --time=48:00:00             # Maximum runtime
#SBATCH --output=/home/tdeatherage3.gatech/logs/unicom_train_%j.log
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90
#SBATCH --requeue                   # Allow the job to be requeued
#SBATCH --open-mode=append          # Append to output files if restarted

# Print job information
echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

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
WEBDATASET_PATH="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train"
OUTPUT_DIR="/home/tdeatherage3.gatech/unicom/output"
CHECKPOINT_DIR="/home/tdeatherage3.gatech/unicom/checkpoints"
LOG_DIR="/home/tdeatherage3.gatech/logs/unicom_train_${SLURM_JOB_ID}"

# Set training parameters
BATCH_SIZE=16  # Per GPU
EPOCHS=32
LR=0.0001
SAMPLE_RATE=0.1  # For random class selection (partial FC)
NUM_FEAT=768     # For random feature selection

# Create required directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

# Set up environment variables for BLASes
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set a random master port to avoid conflicts
export MASTER_PORT=$(( 30000 + RANDOM % 10000 ))
export MASTER_ADDR=$(hostname -s)

# Print distributed training settings
echo "Using MASTER_PORT: $MASTER_PORT"
echo "Using MASTER_ADDR: $MASTER_ADDR"

# Check for existing checkpoint
RESUME_FLAG=""
LATEST_CHECKPOINT=$(find ${CHECKPOINT_DIR} -name "unicom_checkpoint_*.pt" | sort -V | tail -n 1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    RESUME_FLAG="--resume ${LATEST_CHECKPOINT}"
fi

# Run distributed training script
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    /home/tdeatherage3.gatech/unicom/training/unicom_training_distributed.py \
    --db-path "$DB_PATH" \
    --webdataset-path "$WEBDATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --sample-rate "$SAMPLE_RATE" \
    --num-feat "$NUM_FEAT" \
    $RESUME_FLAG

RETURN_CODE=$?

# Clean up any background processes
cleanup() {
    echo "Cleaning up processes..."
    pkill -P $$ || true
    jobs -p | xargs kill -9 >/dev/null 2>&1 || true
}

# Handle exit
if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    cleanup
    exit $RETURN_CODE
fi

cleanup
echo "Job completed at $(date)"