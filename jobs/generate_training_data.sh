#!/bin/bash
#SBATCH --output=~/ai-intepr-project/logs/%x/%j.log
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

cd ~/ai-intepr-project

python your_script.py
