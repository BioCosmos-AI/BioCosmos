#!/bin/bash
#SBATCH --job-name=tsne_viz_img
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=4         # 4 CPUs per task for data loading/processing
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=128gb               # Increased memory for the node (same as original script)
#SBATCH --time=2:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --output=tsne_viz_img_%j.log  # Output log
#SBATCH --mail-type=END,FAIL      # Email notifications
#SBATCH --open-mode=append        # Append to output files if restarted

# Print job information
echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Load required modules
module load conda

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Set path to conda environment - use the same environment as your clustering script
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

# Activate conda environment
conda activate ${CONDA_ENV_PATH}

# Set variables
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"
CHECKPOINT_PATH="/home/tdeatherage3.gatech/logs/clustering_single_60302562/checkpoints/checkpoint.pkl"
DATA_DIR="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train"
OUTPUT_DIR="$HOME/tsne_visualizations_with_images"
LOG_DIR="$HOME/tsne_viz_logs"

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Set CUDA visible devices to match local GPU index
export CUDA_VISIBLE_DEVICES=0

# Set up environment variables for BLASes to use appropriate number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the script without passing species as command-line arguments
# Instead, use the default species list in the script
python /home/tdeatherage3.gatech/unicom/visualization/tsne_visualization_with_images.py \
    --db-path "$DB_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --use-gpu

RETURN_CODE=$?
if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    exit $RETURN_CODE
fi

echo "Job completed at $(date)"
echo "t-SNE visualizations with images are available in $OUTPUT_DIR"