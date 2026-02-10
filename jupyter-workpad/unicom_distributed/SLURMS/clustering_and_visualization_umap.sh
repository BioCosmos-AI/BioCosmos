#!/bin/bash
#SBATCH --job-name=umap_viz
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=4         # 4 CPUs per task for data loading/processing
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=64gb                # Memory for the node
#SBATCH --time=4:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --output=umap_viz_%j.log  # Output log
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

# Set path to conda environment
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

# Activate conda environment
conda activate ${CONDA_ENV_PATH}

# Set variables
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"
TABLE_NAME="image_embeddings"
DATA_DIR="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train"
LOG_DIR="$HOME/logs/umap_viz_${SLURM_JOB_ID}"
OUTPUT_DIR="$HOME/visualizations/umap_clustering"
MIN_K=2
MAX_K=10
N_NEIGHBORS=15
MIN_DIST=0

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Set CUDA visible devices to match local GPU index
export CUDA_VISIBLE_DEVICES=0

# Set up environment variables for BLASes to use appropriate number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the UMAP-based clustering and visualization script
python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/clustering_and_visualization_umap.py \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --data-dir "$DATA_DIR" \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --min-k "$MIN_K" \
    --max-k "$MAX_K" \
    --n-neighbors "$N_NEIGHBORS" \
    --min-dist "$MIN_DIST" \
    --use-gpu
    # --species "Abagrotis alternata" "Abaeis nicippe" "Hemicircus canente" "Hemigomphus comitatus" "Zyrphelis crenata" "Zygaena oxytropis"

RETURN_CODE=$?
if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    exit $RETURN_CODE
fi

echo "Job completed at $(date)"
echo "Visualizations are available in $OUTPUT_DIR"