#!/bin/bash
#SBATCH --job-name=species_names
#SBATCH --nodes=1                     # Single node
#SBATCH --ntasks=1                    # Single task
#SBATCH --cpus-per-task=1            # Single CPU
#SBATCH --mem=32gb                    # Memory for caching
#SBATCH --time=24:00:00              # Longer runtime since sequential
#SBATCH --output=species_names_%j.log
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
LOG_DIR="/home/tdeatherage3.gatech/logs"
BATCH_SIZE=50000

# Run the import script
srun python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/add_species_names.py \
    --db-path "$DB_PATH" \
    --table-name "$TABLE_NAME" \
    --tar-dir "$TAR_DIR" \
    --log-dir "$LOG_DIR" \
    --batch-size "$BATCH_SIZE"