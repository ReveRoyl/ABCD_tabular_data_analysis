#!/bin/bash
#SBATCH --job-name=batch_nni_experiment  # Job name
#SBATCH --output=output/batch_output_%j.log      # Output file name (%j will be replaced by the job ID)
#SBATCH --error=output/batch_error_%j.log        # Error file name (%j will be replaced by the job ID)
#SBATCH --ntasks=1                        # Number of tasks (usually 1 for single-node jobs)
#SBATCH --time=24:00:00                   # Maximum runtime (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G

cd /users/k21116947/Autoencoder

pwd

source ~/miniconda3/bin/activate ABCD

nnictl resume 5qtkvujf --port 8086

while true; do
    sleep infinity
done
