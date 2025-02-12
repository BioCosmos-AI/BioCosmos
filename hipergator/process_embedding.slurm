#!/bin/bash
#SBATCH --job-name=process_embeddings          # Job name
#SBATCH --output=logs/process_embeddings_%j.out    # Standard output log
#SBATCH --error=logs/process_embeddings_%j.err     # Standard error log
#SBATCH --time=24:00:00                   # Time limit hrs:min:sec
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=300G                         # Memory per node
#SBATCH --partition=gpu               # Partition to submit to (adjust as per your cluster)
#SBATCH --gres=gpu:a100:1                      # Number of GPUs per node (if needed)

module load conda # Adjust as needed

conda activate bioclip-train

python run_process_embedding.py train_small
# python run_process_embedding.py val
# python run_process_embedding.py train


