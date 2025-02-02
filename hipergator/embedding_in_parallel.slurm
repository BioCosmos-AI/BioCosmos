#!/bin/bash
#SBATCH --job-name=embedding_in_parallel          # Job name
#SBATCH --output=logs/embedding_in_parallel_%j.out    # Standard output log
#SBATCH --error=logs/embedding_in_parallel_%j.err     # Standard error log
#SBATCH --time=12:00:00                   # Time limit hrs:min:sec
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=2                 # Number of CPU cores per task
#SBATCH --mem=300G                         # Memory per node
#SBATCH --partition=gpu               # Partition to submit to (adjust as per your cluster)
#SBATCH --gres=gpu:a100:2                      # Number of GPUs per node (if needed)

module load conda # Adjust as needed

conda activate bioclip-train

python run_embedding_in_parallel.py train_small
# python run_embedding_in_parallel.py val
# python run_embedding_in_parallel.py train
