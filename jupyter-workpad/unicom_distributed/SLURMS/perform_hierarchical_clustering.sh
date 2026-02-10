#!/bin/bash
#SBATCH --job-name=hierarchical_clustering
#SBATCH --partition=gpu
#SBATCH --nodes=2                    # Request 2 nodes
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=4            # Cores per task
#SBATCH --gres=gpu:a100:1            # One GPU per node
#SBATCH --mem=64G                    # Memory per node - increased from embeddings since we're doing clustering
#SBATCH --time=12:00:00             # Increased time allocation for clustering
#SBATCH --output=clustering_%j.out
#SBATCH --error=clustering_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tdeatherage3@gatech.edu

# Load conda module
module load conda

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Set path to conda environment
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

# Activate conda environment
conda activate ${CONDA_ENV_PATH}

# Set required environment variables
export MASTER_PORT=12355
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Debug vars
# Network configuration
export NCCL_SOCKET_FAMILY=v4
export NCCL_SOCKET_IFNAME=^docker0,lo
# Increase timeout for debugging - clustering might take longer than embeddings
export NCCL_TIMEOUT=2400000  # 40 minutes

# Define paths
BASE_EMBEDDING_DIR=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings
OUTPUT_DIR=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/clustering_results

# Run the clustering script
srun python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/perform_hierarchical_clustering.py \
    --embeddings_dir ${BASE_EMBEDDING_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_name "ViT-H-14-378-quickgelu" \
    --cohort "train" \
    --n_clusters 100 \
    --local_centroids 100
    # --n_clusters 1000000 \
    # --local_centroids 1000000