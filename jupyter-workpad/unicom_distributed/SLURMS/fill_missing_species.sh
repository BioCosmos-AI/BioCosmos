#!/bin/bash
#SBATCH --job-name=fill_missing_species
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=fill_missing_species_%j.log
#SBATCH --mail-type=END,FAIL

# Load conda module
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
LOG_DIR="/home/tdeatherage3.gatech/logs"

# Run the script
python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/fill_missing_species.py \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --log-dir "$LOG_DIR"