#!/bin/bash
#SBATCH --job-name=batch_nni_experiment  # Job name
#SBATCH --output=output/batch_output_%j.log      # Output file name (%j will be replaced by the job ID)
#SBATCH --error=output/batch_error_%j.log        # Error file name (%j will be replaced by the job ID)
#SBATCH --ntasks=1                        # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # Memory per node
#SBATCH --time=24:00:00                   # Maximum runtime (hh:mm:ss)

nnictl stop --all

nnictl create --config config.yml

EXPERIMENT_ID=$(nnictl get | awk 'NR==2 {print $1}')

# Output the best trial
nnictl top $EXPERIMENT_ID > best_parameters.log


