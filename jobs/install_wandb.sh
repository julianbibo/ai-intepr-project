#!/bin/bash
#SBATCH --output=~/ai-intepr-project/logs/%x/%j.log
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

cd ~/ai-intepr-project

module purge
module load 2023
module load Anaconda3/2023.07-2

# activate env
source activate myenv

# install deps
# pip install wandb
pip install python-dotenv
