#!/bin/bash
#SBATCH --job-name=tsne_full
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=8         # 8 CPUs per task for data loading/processing
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=96gb                # Memory for the node
#SBATCH --time=12:00:00           # Maximum runtime (HH:MM:SS)
#SBATCH --output=tsne_full_GPU_%j.log # Output log
#SBATCH --mail-type=END,FAIL      # Email notifications
#SBATCH --open-mode=append        # Append to output files if restarted

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
LOG_DIR="$HOME/logs/tsne_full_GPU_${SLURM_JOB_ID}"
OUTPUT_DIR="$HOME/visualizations/sundry_tsne_clustering_GPU"
SCRIPT_PATH="$HOME/unicom/embedding_and_clustering/sundry_clustering_and_visuzalization.py"

# Define the species to analyze - use the default list from the Python script
SPECIES=(
    "Abagrotis alternata"
    "Abaeis nicippe"
    "Hemicircus canente"
    "Hemigomphus comitatus"
    "Zyrphelis crenata"
    "Zygaena oxytropis"
)

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
if pip show hdbscan >/dev/null 2>&1; then
    echo "HDBSCAN version: $(pip show hdbscan | grep Version | cut -d ' ' -f 2)"
else
    echo "HDBSCAN not installed"
fi

# Format the species list for command line
SPECIES_ARG=""
for species in "${SPECIES[@]}"; do
    SPECIES_ARG="$SPECIES_ARG \"$species\""
done

# Run the full analysis for all species
echo "Running t-SNE-based clustering and visualization script for all species"
echo "Species list: ${SPECIES_ARG}"

python "$SCRIPT_PATH" \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --data-dir "$DATA_DIR" \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --min-k 2 \
    --max-k 20 \
    --dbscan-min-samples 2 \
    --dbscan-min-eps 0.1 \
    --dbscan-max-eps 1.0 \
    --dbscan-eps-steps 20 \
    --hdbscan-min-cluster-size 2
    # --species ${SPECIES[@]}

RETURN_CODE=$?
echo "Python script return code: $RETURN_CODE"

# Check for visualizations and output files
echo "Checking output directory for results:"
ls -la "$OUTPUT_DIR"

# Count the number of visualization files created
NUM_VISUALIZATIONS=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
echo "Number of visualization files created: $NUM_VISUALIZATIONS"

# Check if stats file was created
if [ -f "$OUTPUT_DIR/clustering_stats.json" ]; then
    echo "Clustering statistics file created successfully"
    # Print the first few lines to verify content
    echo "First 10 lines of clustering_stats.json:"
    head -n 10 "$OUTPUT_DIR/clustering_stats.json"
else
    echo "WARNING: Clustering statistics file not found"
fi

echo "Job completed at $(date)"
echo "Visualizations are available in $OUTPUT_DIR"

# Exit with the same code as the Python script
exit $RETURN_CODE