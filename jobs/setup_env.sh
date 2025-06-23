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

# setup env
conda env remove -n myenv -y || true
conda create -n myenv python=3.10 -y
source activate myenv

# install deps
pip install --upgrade pip
pip install -r requirements.txt

# Ensure the environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: Virtual environment is not activated."
    exit 1
fi

