#!/bin/bash -l
#SBATCH --job-name=batch_nni_experiment  # Job name
#SBATCH --output=output/batch_output_%j.log      # Output file name (%j will be replaced by the job ID)
#SBATCH --error=output/batch_error_%j.log        # Error file name (%j will be replaced by the job ID)
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G

nnictl stop --all

nnictl create --config config.yml --port 8081
