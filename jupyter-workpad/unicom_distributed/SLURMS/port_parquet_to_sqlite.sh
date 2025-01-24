#!/bin/bash
#SBATCH --job-name=parquet2sqlite    # Job name
#SBATCH --ntasks=1                   # Run on a single CPU
#SBATCH --mem=64gb                   # Job memory request
#SBATCH --time=12:00:00             # Time limit hrs:min:sec
#SBATCH --output=parquet2sqlite_%j.log   # Standard output and error log
#SBATCH --mail-type=END,FAIL        # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Load conda module
module load conda

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Set path to conda environment
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

# Activate conda environment
conda activate ${CONDA_ENV_PATH}

# Set variables
BASE_DIR="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/embeddings"
DB_PATH="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite"  # Adjust this path as needed
TABLE_NAME="image_embeddings"

# Run the conversion script
python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/port_parquet_to_sqlite.py \
    --base-dir "$BASE_DIR" \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME"