#!/bin/bash
#SBATCH --job-name=fall_unicom_train
#SBATCH --partition=hpg-b200        
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=8         # 8 GPUs
#SBATCH --cpus-per-task=8           # 8 CPUs per GPU (64 total)
#SBATCH --gres=gpu:b200:8           # Request 8 B200 GPUs
#SBATCH --mem=500gb                 # Memory
#SBATCH --time=14-00:00:00          
#SBATCH --output=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/logs/fall_unicom_training_distributed_%j.log
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90
#SBATCH --requeue                   
#SBATCH --open-mode=append


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
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed_b200

# Activate conda environment
conda activate ${CONDA_ENV_PATH}

# Set variables
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"

WEBDATASET_PATH="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train"
# OUTPUT_DIR="/home/tdeatherage3.gatech/unicom/output"
OUTPUT_DIR="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/unicom/output"
# CHECKPOINT_DIR="/home/tdeatherage3.gatech/unicom/checkpoints"
CHECKPOINT_DIR="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/unicom/checkpoints"
# LOG_DIR="/home/tdeatherage3.gatech/logs/unicom_train_${SLURM_JOB_ID}"
LOG_DIR="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/logs/fall_unicom_training_distributed_${SLURM_JOB_ID}"
TEST_FILE_PATH="/home/tdeatherage3.gatech/unicom/test_eval_data/test_eval_vlm4bio.csv"

# Set training parameters
# BATCH_SIZE=64  # Per GPU
EPOCHS=32
# LR=0.0001
# SAMPLE_RATE=0.1  # For random class selection (partial FC)
# NUM_FEAT=768     # For random feature selection

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

# Set PyTorch to use expandable segments for CUDA memory allocation
# TO fix torch.outofmemoryerror
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print distributed training settings
echo "Using MASTER_PORT: $MASTER_PORT"
echo "Using MASTER_ADDR: $MASTER_ADDR"

# Check for existing checkpoint
RESUME_FLAG=""
LATEST_CHECKPOINT=$(find ${CHECKPOINT_DIR} -name "unicom_checkpoint_*.pt" | sort -V | tail -n 1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    RESUME_FLAG="--resume ${LATEST_CHECKPOINT} --use-checkpoint-classes"
fi

# Run distributed training script
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    /home/tdeatherage3.gatech/unicom/training/unicom_training_distributed.py \
    --db-path "$DB_PATH" \
    --webdataset-path "$WEBDATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --epochs "$EPOCHS" \
    --tester-path "$TEST_FILE_PATH" \
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