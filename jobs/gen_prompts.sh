#!/bin/bash
#SBATCH --output=/home/scur1188/ai-intepr-project/logs/%x/%j.log
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

cd ~/ai-intepr-project
# module purge
# module load 2023
# module load Anaconda3/2023.07-2
# source activate myenv

############################################################

python src/gen_prompts.py