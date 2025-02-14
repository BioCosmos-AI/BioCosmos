#!/bin/bash
#SBATCH --job-name=add_index    # Job name
#SBATCH --ntasks=1              # Run on a single CPU
#SBATCH --mem=32gb             # Job memory request
#SBATCH --time=4:00:00         # Time limit hrs:min:sec
#SBATCH --output=add_index_%j.log   # Standard output and error log
#SBATCH --mail-type=END,FAIL        # Mail events

# Load conda module
module load conda

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Set path to conda environment
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

# Activate conda environment
conda activate ${CONDA_ENV_PATH}

# Run the indexing script
python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/index_species_name.py \
    --db-path "/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite" \
    --table-name "image_embeddings"