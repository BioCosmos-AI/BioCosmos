#!/bin/bash
#SBATCH --job-name=validate_similarity          # Job name
#SBATCH --output=logs/validate_similarity_%j.out    # Standard output log
#SBATCH --error=logs/validate_similarity_%j.err     # Standard error log
#SBATCH --time=2:00:00                   # Time limit hrs:min:sec
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=150G                         # Memory per node
#SBATCH --partition=gpu               # Partition to submit to (adjust as per your cluster)
#SBATCH --gres=gpu:1                      # Number of GPUs per node (if needed)

module load conda # Adjust as needed

conda activate bioclip-train

python validate_embeddings_similarity.py train_small
# python run_process_embedding.py val
# python run_process_embedding.py train


