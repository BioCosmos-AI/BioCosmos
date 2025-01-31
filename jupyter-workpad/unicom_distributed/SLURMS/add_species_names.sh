#!/bin/bash
#SBATCH --job-name=add_species_names
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=add_species_names_%j.log
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
TAR_DIR="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train"

# Run the script
python /home/tdeatherage3.gatech/add_species_names.py \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --tar-dir "$TAR_DIR"