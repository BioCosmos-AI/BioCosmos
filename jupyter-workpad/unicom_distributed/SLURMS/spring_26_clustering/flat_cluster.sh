#!/bin/bash
#SBATCH --job-name=
#SBATCH --partition=hpg-b200        
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1         # 1 GPUs
#SBATCH --cpus-per-task=4           # 4 CPUs per GPU
#SBATCH --gres=gpu:b200:1           # Request 1 B200 GPUs
#SBATCH --mem=128gb                 # Memory
#SBATCH --time=14-00:00:00          
#SBATCH --output=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/logs/spring_26_cluster/flat_cluster_%j.log
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90
#SBATCH --requeue                   
#SBATCH --open-mode=append

# Print job information
echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# Load required modules
module load conda

# Initialize conda for bash
# DO I NEED THIS?
eval "$(conda shell.bash hook)"

# Set path to conda environment
CONDA_ENV_PATH=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/messy_env

# Activate conda environment
conda activate ${CONDA_ENV_PATH}


TITLE="Flat Cluster Image-only embed"
PLOT_DIR="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/spring_26/clustering"
PLOT_DIR="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/embeddings"
COSINE=TRUE

python /home/tdeatherage3.gatech/unicom/spring_2026_clustering/flat_cluster.py \
    --title "$TITLE" \
    --embedding-dir "$PLOT_DIR" \
    --plot-dir "$EMBEDDING_DIR" \
    --cosine $COSINE


echo "Job completed at $(date)"