#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --partition=gpu
#SBATCH --nodes=2                    # Request 2 nodes
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1           # One GPU per node
#SBATCH --mem=32G                    # Memory per node
#SBATCH --time=04:30:00
#SBATCH --output=embeddings_%j.out
#SBATCH --error=embeddings_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tdeatherage3@gatech.edu

#module load cuda/11.8

# Load conda module
module load conda

# Initialize conda for bash
eval "$(conda shell.bash hook)"

CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed
# Activate your conda environment
# source mamba activate ${CONDA_ENV_PATH}  # the mamba documentation from the UFL wiki isn't working for me
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
# Increase timeout for debugging
export NCCL_TIMEOUT=1200000  # 20 minutes


# Define paths
BASE_PATH=/blue/arthur.porto-biocosmos/data/datasets/VLM4Bio/Butterfly
CSV_PATH=${BASE_PATH}/metadata/metadata_10k.csv
OUTPUT_DIR=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings

# Run the script with all required arguments
srun python /home/tdeatherage3.gatech/unicom/embedding_and_clustering/generate_embeddings.py \
    --csv_path ${CSV_PATH} \
    --base_path ${BASE_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_name "ViT-H-14-378-quickgelu" \
    --cohort "train" \
    --batch_size 32