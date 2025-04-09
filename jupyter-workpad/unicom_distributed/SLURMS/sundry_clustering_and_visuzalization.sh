#!/bin/bash
#SBATCH --job-name=tsne_debug
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=4         # 4 CPUs per task for data loading/processing
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=64gb                # Memory for the node
#SBATCH --time=4:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --output=tsne_debug_%j.log  # Output log
#SBATCH --mail-type=END,FAIL      # Email notifications
#SBATCH --open-mode=append        # Append to output files if restarted

# Turn on bash debugging
set -x

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
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"
TABLE_NAME="image_embeddings"
DATA_DIR="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train"
LOG_DIR="$HOME/logs/tsne_debug_${SLURM_JOB_ID}"
OUTPUT_DIR="$HOME/visualizations/sundry_tsne_clustering"
SCRIPT_PATH="$HOME/unicom/embedding_and_clustering/sundry_clustering_and_visuzalization.py"

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script file not found at $SCRIPT_PATH"
    exit 1
fi

# Check if the SQLite database exists
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: SQLite database not found at $DB_PATH"
    exit 1
fi

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}
echo "Created directories:"
echo "  Log dir: $LOG_DIR"
echo "  Output dir: $OUTPUT_DIR"

# Print the content of the output directory
echo "Output directory content before execution:"
ls -la ${OUTPUT_DIR}

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
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
python -c "import sqlite3; print('SQLite3 version:', sqlite3.version)"
python -c "try: import hdbscan; print('HDBSCAN version:', hdbscan.__version__); except ImportError: print('HDBSCAN not available')"

# Run with a single species first for debugging
echo "Running t-SNE-based clustering and visualization script with a single species for debugging"
python "$SCRIPT_PATH" \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --data-dir "$DATA_DIR" \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --min-k 2 \
    --max-k 5 \
    --dbscan-min-samples 25 \
    --hdbscan-min-cluster-size 25 \
    --species "Abagrotis alternata"

RETURN_CODE=$?
echo "Python script return code: $RETURN_CODE"

# Check output directory content after execution
echo "Output directory content after execution:"
ls -la ${OUTPUT_DIR}

# Check log directory content
echo "Log directory content:"
ls -la ${LOG_DIR}

# Print the latest log file
LATEST_LOG=$(ls -t ${LOG_DIR}/*.log | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Last 50 lines of the latest log file ($LATEST_LOG):"
    tail -50 "$LATEST_LOG"
else
    echo "No log files found in $LOG_DIR"
fi

if [ $RETURN_CODE -ne 0 ]; then
    echo "Job failed with return code $RETURN_CODE at $(date)"
    exit $RETURN_CODE
fi

echo "Job completed at $(date)"
echo "Visualizations should be available in $OUTPUT_DIR"