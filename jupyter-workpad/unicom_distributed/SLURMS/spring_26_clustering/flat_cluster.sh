#!/bin/bash
#SBATCH --job-name=
#SBATCH --partition=hpg-b200        
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1         # 1 GPUs
#SBATCH --cpus-per-task=4           # 4 CPUs per GPU
#SBATCH --gres=gpu:b200:1           # Request 1 B200 GPUs
#SBATCH --mem=512gb                 # Memory
#SBATCH --time=14-00:00:00          
#SBATCH --output=/blue/arthur.porto-biocosmos/tdeatherage3.gatech/logs/spring_26_cluster/flat_cluster_labelling_AVG_job_%j.log
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80,TIME_LIMIT_90
#SBATCH --requeue                   
#SBATCH --open-mode=append

TITLE="Flat Cluster Image+Text AVG (non-null species) embed COSINE"
echo "Starting: $TITLE"
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

PLOT_DIR="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/spring_26/clustering"
EMBEDDING_DIR="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/combined_embeds/average"
COSINE=TRUE

echo "Embedding dir: $EMBEDDING_DIR"

DROP_NULL_SPECIES=TRUE
# READS this (for image only)
TAXA_METADATA_CSV=/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/cleaned_train.csv

PLOT_ELBOW_ONLY=FALSE
# FOR Generating CSV Cluster Assignements
K=454000
CSV_OUTPUT_FILE="$PLOT_DIR/combined_cluster_labels.csv"

python /home/tdeatherage3.gatech/unicom/spring_2026_clustering/flat_cluster.py \
    --title "$TITLE" \
    --embedding-dir "$EMBEDDING_DIR" \
    --plot-dir "$PLOT_DIR" \
    --cosine $COSINE \
    --drop-null-species $DROP_NULL_SPECIES \
    --taxa-metadata-csv $TAXA_METADATA_CSV \
    --plot-elbow-only $PLOT_ELBOW_ONLY \
    --k $K \
    --csv-output-file $CSV_OUTPUT_FILE


echo "Job completed at $(date)"